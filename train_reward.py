"""
Train the Autoregressive Dense Reward Model on top of a frozen Diffusion-Planner.

For each minibatch:
    1) Encode the scene with the frozen planner encoder.
    2) Sample G candidate trajectories around the GT (Gaussian neighborhood).
    3) Label each candidate with the proxy reward labeler.
    4) Predict rewards with the AD-RM and supervise with BCEWithLogits.

This script is intentionally minimal - users can plug a real PDM scorer into
``label_trajectory_rewards`` later for higher quality labels.
"""
import os
import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.model.latent_predictor import LatentWorldModel
from diffusion_planner.model.metric_weight_head import MetricWeightHead
from diffusion_planner.model.reward_model import AutoregressiveDenseRewardModel, METRIC_NAMES
from diffusion_planner.reward_labeling import label_trajectory_rewards, stack_metrics
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils.early_stopping import EarlyStopper
from diffusion_planner.utils.normalizer import StateNormalizer, ObservationNormalizer
from diffusion_planner.utils.train_utils import set_seed, load_state_dict_verbose
from diffusion_planner.utils.trajectory_vocabulary import total_reward_from_dense


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_set', type=str, required=True)
    p.add_argument('--train_set_list', type=str, required=True)
    p.add_argument('--planner_ckpt', type=str, required=True)
    p.add_argument('--save_dir', type=str, default='reward_runs')

    p.add_argument('--future_len', type=int, default=80)
    p.add_argument('--time_len', type=int, default=21)
    p.add_argument('--agent_state_dim', type=int, default=11)
    p.add_argument('--agent_num', type=int, default=32)
    p.add_argument('--static_objects_state_dim', type=int, default=10)
    p.add_argument('--static_objects_num', type=int, default=5)
    p.add_argument('--lane_len', type=int, default=20)
    p.add_argument('--lane_state_dim', type=int, default=12)
    p.add_argument('--lane_num', type=int, default=70)
    p.add_argument('--route_len', type=int, default=20)
    p.add_argument('--route_state_dim', type=int, default=12)
    p.add_argument('--route_num', type=int, default=25)
    p.add_argument('--predicted_neighbor_num', type=int, default=10)

    p.add_argument('--encoder_depth', type=int, default=3)
    p.add_argument('--decoder_depth', type=int, default=3)
    p.add_argument('--num_heads', type=int, default=6)
    p.add_argument('--hidden_dim', type=int, default=192)
    p.add_argument('--encoder_drop_path_rate', type=float, default=0.1)
    p.add_argument('--decoder_drop_path_rate', type=float, default=0.1)
    p.add_argument('--diffusion_model_type', type=str, default='x_start')
    p.add_argument('--use_shortcut', action='store_true',
                   help='Instantiate planner with Shortcut Forcing d_embedder so SFT shortcut '
                        'checkpoints load without unexpected_keys (planner stays frozen here).')

    p.add_argument('--n_horizons', type=int, default=8)
    p.add_argument('--n_candidates', type=int, default=16)
    p.add_argument('--sigma_xy', type=float, default=1.5)
    p.add_argument('--sigma_heading', type=float, default=0.15)
    p.add_argument('--use_latent_predictor', action='store_true',
                   help='Use a learned latent world model to imagine per-horizon scene tokens.')
    p.add_argument('--latent_layers', type=int, default=2)
    p.add_argument('--predict_uncertainty', action='store_true',
                   help='Emit (mu, log_var) per (horizon, metric) and add uncertainty loss.')
    p.add_argument('--w_uncertainty', type=float, default=0.1,
                   help='Weight of the log-variance fitting loss.')
    p.add_argument('--log_var_min', type=float, default=-2.0,
                   help='Lower clamp on log_var (prevents the uncertainty NLL '
                        'from going to -inf on dead metrics and yielding a '
                        'negative-loss exploit that masks mu-collapse).')
    p.add_argument('--log_var_max', type=float, default=2.0,
                   help='Upper clamp on log_var.')
    p.add_argument('--w_traj_margin', type=float, default=1.0,
                   help='Weight of the hard trajectory-discrimination margin '
                        'loss: forces sign(logit_GT - logit_perturbed) to '
                        'match sign(label_GT - label_perturbed) per metric. '
                        'Set to 0 to disable.')
    p.add_argument('--traj_margin', type=float, default=0.5,
                   help='Hinge margin (in logit space) for the discrimination '
                        'loss.')
    p.add_argument('--traj_label_eps', type=float, default=0.05,
                   help='Only enforce the margin on pairs whose label '
                        'difference exceeds this threshold.')
    p.add_argument('--use_metric_weights', action='store_true',
                   help='Train a context-conditional MetricWeightHead alongside the AD-RM.')
    p.add_argument('--w_metric_margin', type=float, default=0.1,
                   help='Weight of the GT-beats-perturbed margin loss for the metric weight head.')
    p.add_argument('--metric_loss_weights', type=str, default='equal',
                   choices=['equal', 'pdms'],
                   help='Per-metric scaling of BCE / NLL / hinge losses. '
                        '"equal" (default) = uniform 1.0 per metric (original behaviour). '
                        '"pdms" = upweight by nuPlan PDMS importance so the reward model '
                        'does not give up on TTC and comfort. Weights aligned to METRIC_NAMES '
                        '(nc, dac, ttc, ep, comfort, ddc, mp, sl) = (1,1,5,5,2,1,1,4), '
                        'normalized to mean=1.0 to keep the overall loss scale comparable.')
    p.add_argument('--adaptive_horizons', action='store_true',
                   help='Sample horizons at uniform cumulative distance along the candidate '
                        'trajectory and condition on continuous per-horizon (time, distance).')
    p.add_argument('--dt', type=float, default=0.1,
                   help='Seconds per trajectory step (used by adaptive horizons for tau_time).')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--learning_rate', type=float, default=3e-4)
    p.add_argument('--early_stop_patience', type=int, default=0,
                   help='Stop if epoch loss does not improve for this many epochs (0 = disabled).')
    p.add_argument('--early_stop_min_delta', type=float, default=0.0,
                   help='Minimum decrease in epoch loss to count as improvement.')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--normalization_file_path', default='configs/normalization.json')
    args = p.parse_args()

    args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(args)
    args.guidance_fn = None
    args.device = args.device if torch.cuda.is_available() else 'cpu'
    return args


def sample_candidates(gt_traj: torch.Tensor, n_candidates: int, sigma_xy: float, sigma_h: float):
    """gt_traj: (B, T, 4). Returns (B, G, T, 4) candidates with added Gaussian noise."""
    B, T, D = gt_traj.shape
    base = gt_traj.unsqueeze(1).expand(B, n_candidates, T, D).clone()
    noise = torch.randn_like(base)
    noise[..., :2] *= sigma_xy
    noise[..., 2:] *= sigma_h
    base[:, 1:] = base[:, 1:] + noise[:, 1:]  # leave one "ground-truth" anchor
    return base


def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    planner = Diffusion_Planner(args).to(args.device)
    ckpt = torch.load(args.planner_ckpt, map_location=args.device, weights_only=False)
    load_state_dict_verbose(planner, ckpt.get('model', ckpt), strict=False, label="planner")
    planner.eval()
    for p_ in planner.parameters():
        p_.requires_grad_(False)

    reward_model = AutoregressiveDenseRewardModel(
        hidden_dim=args.hidden_dim,
        traj_dim=4,
        n_horizons=args.n_horizons,
        metric_names=METRIC_NAMES,
        predict_uncertainty=args.predict_uncertainty,
        adaptive_horizons=args.adaptive_horizons,
        dt=args.dt,
    ).to(args.device)

    latent_predictor = None
    if args.use_latent_predictor:
        latent_predictor = LatentWorldModel(
            hidden_dim=args.hidden_dim,
            traj_dim=4,
            n_horizons=args.n_horizons,
            n_layers=args.latent_layers,
            adaptive_horizons=args.adaptive_horizons,
            dt=args.dt,
        ).to(args.device)

    metric_weight_head = None
    if args.use_metric_weights:
        metric_weight_head = MetricWeightHead(
            hidden_dim=args.hidden_dim, n_metrics=len(METRIC_NAMES),
        ).to(args.device)

    train_set = DiffusionPlannerData(args.train_set, args.train_set_list,
                                     args.agent_num, args.predicted_neighbor_num, args.future_len)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True)

    params = list(reward_model.parameters())
    if latent_predictor is not None:
        params += list(latent_predictor.parameters())
    if metric_weight_head is not None:
        params += list(metric_weight_head.parameters())
    opt = optim.AdamW(params, lr=args.learning_rate, weight_decay=5e-2)
    bce = nn.BCEWithLogitsLoss()
    bce_none = nn.BCEWithLogitsLoss(reduction='none')

    early_stopper = EarlyStopper(patience=args.early_stop_patience,
                                 min_delta=args.early_stop_min_delta, mode="min")

    # Per-metric loss reweighting. PDMS scored-term weights (ttc=5, ep=5,
    # comfort=2, sl=4) drive closed-loop PDMS; gates (nc, dac, ddc, mp) carry
    # weight 1 in the multiplicative tier. Mean-normalize so the overall BCE/
    # NLL scale stays comparable to the equal-weight baseline.
    if args.metric_loss_weights == 'pdms':
        _raw = torch.tensor([1.0, 1.0, 5.0, 5.0, 2.0, 1.0, 1.0, 4.0])  # nc,dac,ttc,ep,comfort,ddc,mp,sl
        w_metric_loss = (_raw / _raw.mean()).to(args.device)            # mean = 1.0
    else:
        w_metric_loss = torch.ones(len(METRIC_NAMES), device=args.device)
    print(f"metric_loss_weights={args.metric_loss_weights}: " +
          "  ".join(f"{n}={float(v):.2f}" for n, v in zip(METRIC_NAMES, w_metric_loss.cpu())),
          flush=True)

    print(f"Dataset Prepared: {len(train_set)} train data ({len(loader)} batches/epoch)", flush=True)
    n_metrics = len(METRIC_NAMES)
    for epoch in range(args.epochs):
        epoch_loss_sum, epoch_batches = 0.0, 0
        epoch_bce_per_metric = torch.zeros(n_metrics)
        epoch_disc_per_metric = torch.zeros(n_metrics)
        epoch_disc_count_per_metric = torch.zeros(n_metrics)
        print(f"\nEpoch {epoch+1}/{args.epochs}", flush=True)
        pbar = tqdm(loader, desc="Training", unit="batch")
        for batch in pbar:
            ego_future = batch[1].to(args.device)         # (B, T, 3)
            neighbors_future = batch[3].to(args.device)   # (B, P, T, 3+)
            lanes = batch[4].to(args.device)
            inputs = {
                'ego_current_state': batch[0].to(args.device),
                'neighbor_agents_past': batch[2].to(args.device),
                'lanes': lanes,
                'lanes_speed_limit': batch[5].to(args.device),
                'lanes_has_speed_limit': batch[6].to(args.device),
                'route_lanes': batch[7].to(args.device),
                'route_lanes_speed_limit': batch[8].to(args.device),
                'route_lanes_has_speed_limit': batch[9].to(args.device),
                'static_objects': batch[10].to(args.device),
            }
            inputs = args.observation_normalizer(inputs)
            with torch.no_grad():
                enc = planner.encoder(inputs)
                context_tokens = enc['encoding']
            ego_future4 = torch.cat([ego_future[..., :2], ego_future[..., 2:3].cos(), ego_future[..., 2:3].sin()], dim=-1)
            cands = sample_candidates(ego_future4, args.n_candidates, args.sigma_xy, args.sigma_heading)

            B, G, T, _ = cands.shape
            cands_flat = cands.reshape(B * G, T, 4)
            ctx_rep = context_tokens.unsqueeze(1).expand(-1, G, -1, -1).reshape(B * G, *context_tokens.shape[1:])

            neigh_rep = neighbors_future.unsqueeze(1).expand(-1, G, -1, -1, -1).reshape(B * G, *neighbors_future.shape[1:])
            valid_mask = torch.sum(torch.ne(neigh_rep[..., :3], 0), dim=-1) != 0
            lanes_rep = lanes.unsqueeze(1).expand(-1, G, -1, -1, -1).reshape(B * G, *lanes.shape[1:])
            route_lanes_rep = batch[7].to(args.device).unsqueeze(1).expand(-1, G, -1, -1, -1).reshape(B * G, *batch[7].shape[1:])

            labels = stack_metrics(label_trajectory_rewards(cands_flat, neigh_rep, valid_mask, lanes_rep, route_lanes=route_lanes_rep, n_horizons=args.n_horizons))

            if latent_predictor is not None:
                ctx_input = latent_predictor(ctx_rep, cands_flat)   # (B*G, H, N, D)
            else:
                ctx_input = ctx_rep

            # Broadcast per-metric weights along last dim (K) for BCE / NLL / hinge.
            wK = w_metric_loss.view(*([1] * (labels.dim() - 1)), -1)

            if args.predict_uncertainty:
                mu, log_var = reward_model(ctx_input, cands_flat)
                log_var = log_var.clamp(args.log_var_min, args.log_var_max)
                bce_elem = bce_none(mu, labels)                       # (B*G, H, K)
                bce_loss = (bce_elem * wK).mean()
                with torch.no_grad():
                    err2 = (torch.sigmoid(mu).detach() - labels) ** 2
                var = log_var.exp()
                nll = 0.5 * (log_var + err2 / var)
                loss = bce_loss + args.w_uncertainty * (nll * wK).mean()
                logits = mu
                preds_prob = torch.sigmoid(mu)
            else:
                preds = reward_model(ctx_input, cands_flat)
                bce_elem = bce_none(preds, labels)
                bce_loss = (bce_elem * wK).mean()
                loss = bce_loss
                logits = preds
                preds_prob = torch.sigmoid(preds)

            # Hard trajectory-discrimination margin: candidate 0 is the GT
            # anchor (see sample_candidates); for every other candidate, force
            # sign(logit[0]-logit[g]) to match sign(label[0]-label[g]) by a
            # margin, but only on pairs whose labels actually differ. Without
            # this term the model can ignore the trajectory and still drive
            # BCE+NLL low by predicting the per-metric label mean.
            logits_bg = logits.reshape(B, G, *logits.shape[1:])         # (B, G, H, K)
            labels_bg = labels.reshape(B, G, *labels.shape[1:])
            d_logit = logits_bg[:, 0:1] - logits_bg[:, 1:]
            d_label = labels_bg[:, 0:1] - labels_bg[:, 1:]
            sign = torch.sign(d_label)
            pair_mask = (d_label.abs() > args.traj_label_eps).float()
            hinge = torch.relu(args.traj_margin - sign * d_logit) * pair_mask  # (B, G-1, H, K)
            # PDMS-weighted hinge: keep the per-metric scale-preserving mean
            # by normalizing weights to sum to K (mean=1) and dividing by their
            # sum so the equal-weight case is bit-identical.
            wK_hinge = w_metric_loss.view(*([1] * (hinge.dim() - 1)), -1)
            denom = (pair_mask * wK_hinge).sum().clamp_min(1.0)
            disc_loss = (hinge * wK_hinge).sum() / denom
            if args.w_traj_margin > 0.0:
                loss = loss + args.w_traj_margin * disc_loss

            with torch.no_grad():
                epoch_bce_per_metric += bce_elem.mean(dim=(0, 1)).detach().cpu()
                epoch_disc_per_metric += hinge.sum(dim=(0, 1, 2)).detach().cpu()
                epoch_disc_count_per_metric += pair_mask.sum(dim=(0, 1, 2)).detach().cpu()

            if metric_weight_head is not None:
                weights = metric_weight_head(ctx_rep)                     # (B*G, K)
                preds_bg = preds_prob.reshape(B, G, *preds_prob.shape[1:])
                w_bg = weights.reshape(B, G, -1)
                # the first candidate (index 0) is the GT anchor (see sample_candidates)
                r_all = torch.stack([
                    total_reward_from_dense(preds_bg[:, g], metric_weights=w_bg[:, g])
                    for g in range(G)
                ], dim=1)                                                 # (B, G)
                margin = (r_all[:, 1:] - r_all[:, :1]).clamp(min=0.0).mean()
                loss = loss + args.w_metric_margin * margin

            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss_sum += float(loss.detach()); epoch_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", disc=f"{disc_loss.item():.3f}")
        ckpt = {'model': reward_model.state_dict(),
                'predict_uncertainty': args.predict_uncertainty,
                'adaptive_horizons': args.adaptive_horizons,
                'dt': args.dt}
        if latent_predictor is not None:
            ckpt['latent_predictor'] = latent_predictor.state_dict()
            ckpt['latent_layers'] = args.latent_layers
        if metric_weight_head is not None:
            ckpt['metric_weight_head'] = metric_weight_head.state_dict()
        torch.save(ckpt, os.path.join(args.save_dir, f'reward_epoch_{epoch+1}.pth'))

        mean_loss = epoch_loss_sum / max(epoch_batches, 1)
        print(f"epoch {epoch+1} train loss: {mean_loss:.4f}", flush=True)
        per_metric_bce = (epoch_bce_per_metric / max(epoch_batches, 1)).tolist()
        per_metric_disc = (epoch_disc_per_metric /
                           epoch_disc_count_per_metric.clamp_min(1.0)).tolist()
        per_metric_active = epoch_disc_count_per_metric.tolist()
        print(f"  per-metric BCE :  " +
              "  ".join(f"{n}={v:.3f}" for n, v in zip(METRIC_NAMES, per_metric_bce)),
              flush=True)
        print(f"  per-metric hinge: " +
              "  ".join(f"{n}={v:.3f}" for n, v in zip(METRIC_NAMES, per_metric_disc)),
              flush=True)
        print(f"  pairs w/ |dlabel|>{args.traj_label_eps}: " +
              "  ".join(f"{n}={int(c)}" for n, c in zip(METRIC_NAMES, per_metric_active)),
              flush=True)
        should_stop = early_stopper.step(mean_loss, epoch + 1)
        if early_stopper.improved and not early_stopper.disabled:
            torch.save(ckpt, os.path.join(args.save_dir, 'reward_best.pth'))
            print(f"[epoch {epoch+1}] best loss={mean_loss:.4f}", flush=True)
        if should_stop:
            print(f"Early stopping at epoch {epoch+1}: no improvement for "
                  f"{early_stopper.bad_epochs} epochs "
                  f"(best={early_stopper.best:.4f} @ epoch {early_stopper.best_epoch}).")
            break


if __name__ == "__main__":
    main()
