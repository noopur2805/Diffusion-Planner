"""
GRPO RL fine-tuning of the Diffusion-Planner using:
    * a trajectory vocabulary (filtered + uniform-spaced) built once from the GT corpus
    * the AD-RM (frozen) as the reward source
    * a frozen reference copy of the SFT planner for KL regularization

Run after ``train_predictor.py`` (SFT) and ``train_reward.py`` (AD-RM).
"""
import os
import argparse
import copy

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.model.latent_predictor import LatentWorldModel
from diffusion_planner.model.metric_weight_head import MetricWeightHead
from diffusion_planner.model.reward_model import AutoregressiveDenseRewardModel
from diffusion_planner.reward_labeling import drift_augmented_rewards, stack_metrics
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils.trajectory_vocabulary import (
    DynamicVocabulary,
    gaussian_vocab_sample,
    total_reward_from_dense,
)
from diffusion_planner.utils.early_stopping import EarlyStopper
from diffusion_planner.utils.normalizer import StateNormalizer, ObservationNormalizer
from diffusion_planner.grpo import grpo_total_loss, diag_gauss_logprob, group_advantage
from diffusion_planner.utils.train_utils import set_seed, load_state_dict_verbose


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_set', required=True)
    p.add_argument('--train_set_list', required=True)
    p.add_argument('--planner_ckpt', required=True)
    p.add_argument('--reward_ckpt', required=True)
    p.add_argument('--vocab_path', required=True, help='.pt file with (V, T, 3) tensor')
    p.add_argument('--save_dir', default='grpo_runs')

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
    p.add_argument('--encoder_drop_path_rate', type=float, default=0.0)
    p.add_argument('--decoder_drop_path_rate', type=float, default=0.0)
    p.add_argument('--diffusion_model_type', type=str, default='x_start')
    p.add_argument('--use_shortcut', action='store_true')
    p.add_argument('--shortcut_steps', type=int, default=1)

    p.add_argument('--g1', type=int, default=8)
    p.add_argument('--g2', type=int, default=8)
    p.add_argument('--sigma_xy', type=float, default=1.5)
    p.add_argument('--sigma_h', type=float, default=0.2)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--w_bc', type=float, default=1.0)
    p.add_argument('--w_kl', type=float, default=0.1)
    p.add_argument('--w_ent', type=float, default=0.0,
                   help='Weight on the trajectory-dispersion entropy bonus '
                        '(see trajectory_dispersion in grpo.py). 0 = original '
                        'behaviour. Raise (e.g. 0.05-0.5) to penalise mode '
                        'collapse: the diffusion-policy log-prob has constant '
                        'Gaussian entropy under fixed sigma, so we instead '
                        'subtract a per-batch dispersion of policy_mean from '
                        'the loss, pushing per-scene predictions to stay '
                        'diverse across the batch.')
    p.add_argument('--clip_eps', type=float, default=0.2)
    p.add_argument('--grad_clip', type=float, default=1.0,
                   help='max grad norm; <=0 disables clipping')
    p.add_argument('--uncertainty_temp', type=float, default=1.0,
                   help='Scale on per-candidate uncertainty when weighting advantages (only used if reward ckpt has an uncertainty head).')
    p.add_argument('--horizon_uncertainty_temp', type=float, default=0.0,
                   help='Scale on per-horizon uncertainty in the reward aggregator '
                        '(0.0 = uniform-horizon sum, the original behaviour).')
    p.add_argument('--cumulative_uncertainty', action='store_true',
                   help='Replace per-horizon sigma with its running max along the '
                        'horizon axis (cummax) before damping. Enforces a monotone '
                        'non-increasing damping factor.')
    p.add_argument('--use_metric_weights', action='store_true',
                   help='If a MetricWeightHead is present in the reward ckpt, load it and '
                        'use scene-conditional per-metric weights in the aggregator.')
    p.add_argument('--use_dynamic_vocab', action='store_true',
                   help='Wrap the static vocabulary in a DynamicVocabulary that admits new '
                        'winning candidates each step and evicts low-utility entries.')
    p.add_argument('--dynamic_vocab_capacity', type=int, default=8192)
    p.add_argument('--dynamic_vocab_age_decay', type=float, default=0.0)
    p.add_argument('--dynamic_vocab_add_per_step', type=int, default=2,
                   help='Top-advantage candidates per batch element to push back into the buffer.')
    p.add_argument('--drift_aug_K', type=int, default=1,
                   help='Tier-C closed-loop-aware reward: split each candidate '
                        'into K equal segments and apply per-segment xy drift '
                        '(epsilon_k ~ N(0, sigma*sqrt(k))) before proxy scoring. '
                        '1 = original AD-RM single-shot reward.')
    p.add_argument('--drift_aug_sigma', type=float, default=0.5,
                   help='Per-segment drift sigma in meters. Ignored when '
                        'drift_aug_K <= 1.')
    p.add_argument('--sft_anchored_advantage', action='store_true',
                   help='Lift the group advantage baseline to max(group_mean, R_SFT), '
                        'so candidates worse than the SFT mean get non-positive advantage. '
                        'Anchors GRPO to the SFT prior (off-course fix).')
    p.add_argument('--bc_horizon_alpha', type=float, default=0.0,
                   help='Quadratic late-horizon weight on the BC term: '
                        'w(h) = (1 + alpha * h/(H-1))**2. 0 = uniform (original).')
    p.add_argument('--reward_gate_floor', type=float, default=0.0,
                   help='Soften multiplicative safety gates in the reward aggregator '
                        'by mapping safety probabilities p -> floor + (1-floor)*p. '
                        '0.0 = original PDMS-style hard gates; 0.3 is a sensible '
                        'stability setting when adding sparse gates (e.g. mp, ddc).')
    p.add_argument('--adv_std_floor', type=float, default=1e-6,
                   help='Floor on per-group reward std before advantage normalization. '
                        'Raise (e.g. 0.05) to prevent Z-score blow-ups when most '
                        'candidates in a group have near-zero reward.')
    p.add_argument('--adv_clip', type=float, default=0.0,
                   help='If > 0, clamp normalized advantages to [-adv_clip, +adv_clip]. '
                        '0.0 disables (original behaviour); 3.0 is a sensible cap.')
    p.add_argument('--reward_safety_set', type=str, default='pdms',
                   choices=['pdms', 'minimal'],
                   help='Partition of the 8 PDMS metrics into multiplicative (safety) '
                        'and additive (task) terms inside total_reward_from_dense. '
                        '"pdms" = nuPlan-faithful: nc/dac/ddc/mp gates, ttc/ep/comfort/sl '
                        'task. "minimal" = only nc/dac gate; ddc/mp join the task sum, '
                        'so sparse mp/ddc predictions cannot zero out the reward and '
                        'GRPO advantages stay bounded. Only affects K=8.')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--early_stop_patience', type=int, default=0,
                   help='Stop if epoch loss does not improve for this many epochs (0 = disabled).')
    p.add_argument('--early_stop_min_delta', type=float, default=0.0,
                   help='Minimum decrease in epoch loss to count as improvement.')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--normalization_file_path', default='configs/normalization.json')
    args = p.parse_args()

    args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(args)
    args.guidance_fn = None
    args.device = args.device if torch.cuda.is_available() else 'cpu'
    if args.reward_safety_set == 'minimal':
        args.safety_idx = (0, 1)               # nc, dac
        args.task_idx = (2, 3, 4, 5, 6, 7)     # ttc, ep, comfort, ddc, mp, sl
    else:
        args.safety_idx = None                 # auto-detect (pdms-style)
        args.task_idx = None
    return args


def _to_xy_heading(traj4: torch.Tensor) -> torch.Tensor:
    """(B, T, 4 [x,y,cos,sin]) -> (B, T, 3 [x,y,theta])"""
    theta = torch.atan2(traj4[..., 3], traj4[..., 2])
    return torch.stack([traj4[..., 0], traj4[..., 1], theta], dim=-1)


def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    planner = Diffusion_Planner(args).to(args.device)
    ckpt = torch.load(args.planner_ckpt, map_location=args.device, weights_only=False)
    load_state_dict_verbose(planner, ckpt.get('model', ckpt), strict=False, label="planner")

    ref_planner = copy.deepcopy(planner).to(args.device).eval()
    for p_ in ref_planner.parameters():
        p_.requires_grad_(False)

    r_ckpt = torch.load(args.reward_ckpt, map_location=args.device, weights_only=False)
    predict_uncertainty = bool(r_ckpt.get('predict_uncertainty', False))
    adaptive_horizons = bool(r_ckpt.get('adaptive_horizons', False))
    dt = float(r_ckpt.get('dt', 0.1))
    reward_model = AutoregressiveDenseRewardModel(
        hidden_dim=args.hidden_dim, traj_dim=4,
        predict_uncertainty=predict_uncertainty,
        adaptive_horizons=adaptive_horizons, dt=dt,
    ).to(args.device).eval()
    load_state_dict_verbose(reward_model, r_ckpt.get('model', r_ckpt), strict=False, label="reward_model")
    for p_ in reward_model.parameters():
        p_.requires_grad_(False)

    latent_predictor = None
    if 'latent_predictor' in r_ckpt:
        latent_predictor = LatentWorldModel(
            hidden_dim=args.hidden_dim, traj_dim=4,
            n_layers=int(r_ckpt.get('latent_layers', 2)),
            adaptive_horizons=adaptive_horizons, dt=dt,
        ).to(args.device).eval()
        load_state_dict_verbose(latent_predictor, r_ckpt['latent_predictor'], strict=False, label="latent_predictor")
        for p_ in latent_predictor.parameters():
            p_.requires_grad_(False)

    metric_weight_head = None
    if args.use_metric_weights and 'metric_weight_head' in r_ckpt:
        metric_weight_head = MetricWeightHead(hidden_dim=args.hidden_dim).to(args.device).eval()
        load_state_dict_verbose(metric_weight_head, r_ckpt['metric_weight_head'], strict=False, label="metric_weight_head")
        for p_ in metric_weight_head.parameters():
            p_.requires_grad_(False)

    vocab = torch.load(args.vocab_path, map_location=args.device, weights_only=False)  # (V, T, 3)
    assert vocab.dim() == 3 and vocab.shape[-1] == 3
    dyn_vocab = None
    if args.use_dynamic_vocab:
        dyn_vocab = DynamicVocabulary(
            capacity=args.dynamic_vocab_capacity,
            T=vocab.shape[1], traj_dim=vocab.shape[2],
            age_decay=args.dynamic_vocab_age_decay, device=args.device,
        )
        dyn_vocab.add(vocab[: args.dynamic_vocab_capacity])

    train_set = DiffusionPlannerData(args.train_set, args.train_set_list,
                                     args.agent_num, args.predicted_neighbor_num, args.future_len)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True)
    n_batches = len(loader)
    print(f"Dataset Prepared: {len(train_set)} train data ({n_batches} batches/epoch)", flush=True)

    opt = optim.AdamW(planner.parameters(), lr=args.learning_rate, weight_decay=1e-2)

    early_stopper = EarlyStopper(patience=args.early_stop_patience,
                                 min_delta=args.early_stop_min_delta, mode="min")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}", flush=True)
        epoch_loss_sum, epoch_batches = 0.0, 0
        epoch_actor_sum = epoch_bc_sum = epoch_kl_sum = epoch_ent_sum = 0.0
        pbar = tqdm(loader, total=n_batches, desc="GRPO", unit="batch", dynamic_ncols=True)
        for batch in pbar:
            lanes_raw = batch[4].to(args.device)
            route_lanes_raw = batch[7].to(args.device)
            inputs = {
                'ego_current_state': batch[0].to(args.device),
                'neighbor_agents_past': batch[2].to(args.device),
                'lanes': lanes_raw,
                'lanes_speed_limit': batch[5].to(args.device),
                'lanes_has_speed_limit': batch[6].to(args.device),
                'route_lanes': batch[7].to(args.device),
                'route_lanes_speed_limit': batch[8].to(args.device),
                'route_lanes_has_speed_limit': batch[9].to(args.device),
                'static_objects': batch[10].to(args.device),
            }
            gt_future = batch[1].to(args.device)
            neigh_future_raw = batch[3].to(args.device) if args.drift_aug_K > 1 else None
            inputs = args.observation_normalizer(inputs)

            planner.eval()
            with torch.no_grad():
                _, out_ref = ref_planner(inputs)
                ref_traj = out_ref['prediction'][:, 0]  # (B, T, 4)
                ref_traj3 = _to_xy_heading(ref_traj)

            # NOTE: stay in eval() so the decoder takes its inference path
            # and emits 'prediction'. eval() does not disable autograd, so
            # gradients still flow from the GRPO loss back through the
            # denoising chain into planner parameters. The samplers default
            # to a torch.no_grad() context for inference; we opt in to grad
            # tracking only for this policy forward.
            inputs_policy = {**inputs, "enable_grad_sampling": True}
            _, out_new = planner(inputs_policy)
            policy_mean = out_new['prediction'][:, 0]   # (B, T, 4) ego
            policy_mean3 = _to_xy_heading(policy_mean)

            if dyn_vocab is not None:
                cands3, vocab_idx, _ = dyn_vocab.sample(ref_traj3,
                                                        g1=args.g1, g2=args.g2,
                                                        sigma_xy=args.sigma_xy, sigma_h=args.sigma_h)
            else:
                cands3, vocab_idx, _ = gaussian_vocab_sample(vocab, ref_traj3,
                                                             g1=args.g1, g2=args.g2,
                                                             sigma_xy=args.sigma_xy, sigma_h=args.sigma_h)
            B, G, T, _ = cands3.shape
            cands4 = torch.cat([cands3[..., :2], cands3[..., 2:3].cos(), cands3[..., 2:3].sin()], dim=-1)

            with torch.no_grad():
                if args.drift_aug_K > 1:
                    # Tier-C: bypass the frozen AD-RM and score each candidate
                    # group under K-segment drift on the raw proxy metrics.
                    neigh_valid = (neigh_future_raw.abs().sum(dim=-1) > 0)     # (B, P, T_n)
                    n_per_seg = max(1, 8 // args.drift_aug_K)
                    rewards_list = []
                    for g in range(G):
                        m_dict = drift_augmented_rewards(
                            cands4[:, g], neigh_future_raw, neigh_valid, lanes_raw,
                            K=args.drift_aug_K, sigma_drift=args.drift_aug_sigma,
                            n_horizons_per_segment=n_per_seg,
                            route_lanes=route_lanes_raw,
                        )
                        preds_g = stack_metrics(m_dict)                        # (B, T_h, K_m)
                        rewards_list.append(total_reward_from_dense(
                            preds_g,
                            sigma=None,
                            metric_weights=None,
                            horizon_uncertainty_temp=args.horizon_uncertainty_temp,
                            cumulative_uncertainty=args.cumulative_uncertainty,
                            gate_floor=args.reward_gate_floor,
                            safety_idx=args.safety_idx,
                            task_idx=args.task_idx,
                        ))
                    rewards = torch.stack(rewards_list, dim=1)                 # (B, G)
                    sigma_grouped = None
                    r_sft = None
                    if args.sft_anchored_advantage:
                        m_sft = drift_augmented_rewards(
                            ref_traj, neigh_future_raw, neigh_valid, lanes_raw,
                            K=args.drift_aug_K, sigma_drift=args.drift_aug_sigma,
                            n_horizons_per_segment=n_per_seg,
                            route_lanes=route_lanes_raw,
                        )
                        preds_sft = stack_metrics(m_sft)                       # (B, T_h, K_m)
                        r_sft = total_reward_from_dense(
                            preds_sft,
                            sigma=None,
                            metric_weights=None,
                            horizon_uncertainty_temp=args.horizon_uncertainty_temp,
                            cumulative_uncertainty=args.cumulative_uncertainty,
                            gate_floor=args.reward_gate_floor,
                            safety_idx=args.safety_idx,
                            task_idx=args.task_idx,
                        )
                    advantages = group_advantage(rewards, r_sft=r_sft,
                                                 std_floor=args.adv_std_floor,
                                                 clip=args.adv_clip)
                else:
                    enc = planner.encoder(inputs)
                    ctx = enc['encoding']
                    ctx_rep = ctx.unsqueeze(1).expand(-1, G, -1, -1).reshape(B * G, *ctx.shape[1:])
                    cands_flat = cands4.reshape(B * G, T, 4)
                    if latent_predictor is not None:
                        ctx_in = latent_predictor(ctx_rep, cands_flat)
                    else:
                        ctx_in = ctx_rep
                    if predict_uncertainty:
                        mu, log_var = reward_model(ctx_in, cands_flat)
                        preds = torch.sigmoid(mu)
                        sigma = (0.5 * log_var).exp()
                    else:
                        preds = torch.sigmoid(reward_model(ctx_in, cands_flat))
                        sigma = None
                    preds = preds.reshape(B, G, *preds.shape[1:])
                    sigma_grouped = sigma.reshape(B, G, *sigma.shape[1:]) if sigma is not None else None

                    if metric_weight_head is not None:
                        weights = metric_weight_head(ctx)                      # (B, K)
                        weights_bg = weights.unsqueeze(1).expand(-1, G, -1)
                    else:
                        weights_bg = None

                    rewards = torch.stack([
                        total_reward_from_dense(
                            preds[:, g],
                            sigma=sigma_grouped[:, g] if sigma_grouped is not None else None,
                            metric_weights=weights_bg[:, g] if weights_bg is not None else None,
                            horizon_uncertainty_temp=args.horizon_uncertainty_temp,
                            cumulative_uncertainty=args.cumulative_uncertainty,
                            gate_floor=args.reward_gate_floor,
                            safety_idx=args.safety_idx,
                            task_idx=args.task_idx,
                        )
                        for g in range(G)
                    ], dim=1)
                    r_sft = None
                    if args.sft_anchored_advantage:
                        ctx_in_sft = latent_predictor(ctx, ref_traj) if latent_predictor is not None else ctx
                        if predict_uncertainty:
                            mu_s, log_var_s = reward_model(ctx_in_sft, ref_traj)
                            preds_sft = torch.sigmoid(mu_s)
                            sigma_sft = (0.5 * log_var_s).exp()
                        else:
                            preds_sft = torch.sigmoid(reward_model(ctx_in_sft, ref_traj))
                            sigma_sft = None
                        w_sft = weights if metric_weight_head is not None else None
                        r_sft = total_reward_from_dense(
                            preds_sft,
                            sigma=sigma_sft,
                            metric_weights=w_sft,
                            horizon_uncertainty_temp=args.horizon_uncertainty_temp,
                            cumulative_uncertainty=args.cumulative_uncertainty,
                            gate_floor=args.reward_gate_floor,
                            safety_idx=args.safety_idx,
                            task_idx=args.task_idx,
                        )
                    advantages = group_advantage(rewards, r_sft=r_sft,
                                                 std_floor=args.adv_std_floor,
                                                 clip=args.adv_clip)
                    if sigma_grouped is not None:
                        # average uncertainty per (B, G) candidate, then down-weight advantages
                        cand_unc = sigma_grouped.mean(dim=(-1, -2))
                        advantages = advantages / (1.0 + args.uncertainty_temp * cand_unc)
                old_logp = diag_gauss_logprob(cands4, ref_traj, args.sigma_xy, args.sigma_h)

            new_logp = diag_gauss_logprob(cands4, policy_mean, args.sigma_xy, args.sigma_h)
            gt4 = torch.cat([gt_future[..., :2], gt_future[..., 2:3].cos(), gt_future[..., 2:3].sin()], dim=-1)
            loss, log = grpo_total_loss(new_logp, old_logp, advantages,
                                        policy_mean, gt4, ref_traj,
                                        args.sigma_xy, args.sigma_h,
                                        w_bc=args.w_bc, w_kl=args.w_kl, clip_eps=args.clip_eps,
                                        bc_horizon_alpha=args.bc_horizon_alpha,
                                        w_ent=args.w_ent)
            # Skip non-finite losses (rare upstream NaN) instead of poisoning weights.
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                pbar.set_postfix(loss="nan-skip-loss",
                                 actor=f"{float(log['actor']):.3f}",
                                 bc=f"{float(log['bc']):.3f}",
                                 kl=f"{float(log['kl']):.3f}",
                                 ent=f"{float(log['ent']):.3f}")
                continue
            opt.zero_grad()
            loss.backward()
            # Defensive: grad_clip alone does not stop NaN gradients (NaN propagates
            # through total_norm). Explicitly inspect gradient finiteness before stepping.
            bad_grad = False
            for p_ in planner.parameters():
                if p_.grad is not None and not torch.isfinite(p_.grad).all():
                    bad_grad = True
                    break
            if bad_grad:
                opt.zero_grad(set_to_none=True)
                pbar.set_postfix(loss=f"{float(loss.detach()):.4f}", note="nan-skip-grad")
                continue
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(planner.parameters(), max_norm=args.grad_clip)
            opt.step()
            l = float(loss.detach())
            epoch_loss_sum += l; epoch_batches += 1
            epoch_actor_sum += float(log['actor'])
            epoch_bc_sum    += float(log['bc'])
            epoch_kl_sum    += float(log['kl'])
            epoch_ent_sum   += float(log['ent'])
            pbar.set_postfix(loss=f"{l:.4f}",
                             actor=f"{float(log['actor']):.3f}",
                             bc=f"{float(log['bc']):.3f}",
                             kl=f"{float(log['kl']):.3f}",
                             ent=f"{float(log['ent']):.3f}")

            if dyn_vocab is not None:
                with torch.no_grad():
                    k_add = min(args.dynamic_vocab_add_per_step, G)
                    top_a, top_g = torch.topk(advantages, k=k_add, dim=-1)      # (B, k_add)
                    # bump utility of pre-existing buffer entries that won (before add reshuffles)
                    flat_vocab_idx = vocab_idx.gather(1, top_g).reshape(-1)
                    dyn_vocab.update_utility(flat_vocab_idx, top_a.reshape(-1))
                    flat_idx = (torch.arange(B, device=args.device)[:, None] * G + top_g).reshape(-1)
                    winners = cands3.reshape(B * G, T, cands3.shape[-1])[flat_idx]
                    dyn_vocab.add(winners, top_a.reshape(-1))
                    dyn_vocab.tick()
        ckpt_out = {'model': planner.state_dict()}
        if dyn_vocab is not None:
            ckpt_out['dynamic_vocab'] = dyn_vocab.state_dict()
        torch.save(ckpt_out, os.path.join(args.save_dir, f'grpo_epoch_{epoch+1}.pth'))

        nb = max(epoch_batches, 1)
        mean_loss = epoch_loss_sum / nb
        mean_bc = epoch_bc_sum / nb
        print(f"epoch {epoch+1} train loss: {mean_loss:.4f}  "
              f"(actor={epoch_actor_sum/nb:.4f}, bc={mean_bc:.4f}, "
              f"kl={epoch_kl_sum/nb:.4f}, ent={epoch_ent_sum/nb:.4f})",
              flush=True)
        # Use bc (L1 to GT) as the convergence criterion: bounded and immune to
        # clipped-surrogate explosions that pollute mean_loss on outlier batches.
        should_stop = early_stopper.step(mean_bc, epoch + 1)
        if early_stopper.improved and not early_stopper.disabled:
            torch.save(ckpt_out, os.path.join(args.save_dir, 'grpo_best.pth'))
            print(f"[epoch {epoch+1}] best bc={mean_bc:.4f}", flush=True)
        if should_stop:
            print(f"Early stopping at epoch {epoch+1}: no improvement for "
                  f"{early_stopper.bad_epochs} epochs "
                  f"(best={early_stopper.best:.4f} @ epoch {early_stopper.best_epoch}).",
                  flush=True)
            break


if __name__ == "__main__":
    main()
