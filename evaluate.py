"""
Open-loop evaluation of a Diffusion-Planner checkpoint.

Reports ADE / FDE / heading-error (in meters / radians) against the GT
trajectory, plus a NavSim-style proxy-PDMS aggregated from the vectorized
``label_trajectory_rewards`` metrics. Run twice (SFT and GRPO) and the delta
quantifies the GRPO improvement without needing NavSim closed-loop sim.

Example:
    python evaluate.py \
        --planner_ckpt runs/sft/.../best.pth \
        --train_set $TRAIN_SET --train_set_list $TRAIN_LIST \
        --n_batches 64 --batch_size 4
"""
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils.normalizer import StateNormalizer, ObservationNormalizer
from diffusion_planner.utils.train_utils import set_seed, load_state_dict_verbose
from diffusion_planner.reward_labeling import label_trajectory_rewards, METRIC_ORDER


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--planner_ckpt', required=True)
    p.add_argument('--train_set', required=True)
    p.add_argument('--train_set_list', required=True)
    p.add_argument('--n_batches', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--n_horizons', type=int, default=8)
    # Model / data dims must match the planner config used at training time.
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
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--normalization_file_path', default='configs/normalization.json')
    args = p.parse_args()
    args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(args)
    args.guidance_fn = None
    args.device = args.device if torch.cuda.is_available() else 'cpu'
    return args


def _wrap_pi(x):
    return torch.atan2(x.sin(), x.cos())


@torch.no_grad()
def main():
    args = get_args()
    set_seed(args.seed)

    planner = Diffusion_Planner(args).to(args.device).eval()
    ckpt = torch.load(args.planner_ckpt, map_location=args.device, weights_only=False)
    load_state_dict_verbose(planner, ckpt.get('model', ckpt), strict=False, label='planner')

    ds = DiffusionPlannerData(args.train_set, args.train_set_list,
                              args.agent_num, args.predicted_neighbor_num, args.future_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=True)

    n_batches = min(args.n_batches, len(loader))
    ade_sum = fde_sum = herr_sum = 0.0
    metric_sums = {k: 0.0 for k in METRIC_ORDER}
    # Horizon-binned ADE (assumes 10 Hz future at 8 s -> 80 steps).
    horizon_edges = [(0, 20), (20, 40), (40, 60), (60, 80)]   # 0-2 s, 2-4 s, 4-6 s, 6-8 s
    horizon_sums = [0.0 for _ in horizon_edges]
    n = 0

    for bi, batch in enumerate(tqdm(loader, total=n_batches, desc='eval', unit='batch')):
        if bi >= n_batches:
            break
        inputs = {
            'ego_current_state': batch[0].to(args.device),
            'neighbor_agents_past': batch[2].to(args.device),
            'lanes': batch[4].to(args.device),
            'lanes_speed_limit': batch[5].to(args.device),
            'lanes_has_speed_limit': batch[6].to(args.device),
            'route_lanes': batch[7].to(args.device),
            'route_lanes_speed_limit': batch[8].to(args.device),
            'route_lanes_has_speed_limit': batch[9].to(args.device),
            'static_objects': batch[10].to(args.device),
        }
        gt_future = batch[1].to(args.device)                  # (B, T, 3) [x, y, theta] meters
        neigh_future = batch[3].to(args.device)               # (B, P, T, D)
        lanes = batch[4].to(args.device)
        inputs = args.observation_normalizer(inputs)

        _, out = planner(inputs)
        pred = out['prediction'][:, 0]                         # (B, T, 4) [x, y, cos, sin] meters
        B = pred.shape[0]

        # Open-loop displacement / heading errors vs GT.
        dxy = (pred[..., :2] - gt_future[..., :2])
        dist_t = (dxy * dxy).sum(-1).sqrt()                    # (B, T)
        ade_sum += dist_t.mean(dim=-1).sum().item()
        fde_sum += dist_t[..., -1].sum().item()
        pred_theta = torch.atan2(pred[..., 3], pred[..., 2])
        herr_sum += _wrap_pi(pred_theta[..., -1] - gt_future[..., -1, 2]).abs().sum().item()
        for hi, (lo, hi_) in enumerate(horizon_edges):
            horizon_sums[hi] += dist_t[..., lo:hi_].mean(dim=-1).sum().item()

        # Proxy-PDMS components via the vectorized labeler.
        valid = torch.sum(torch.ne(neigh_future[..., :3], 0), dim=-1) != 0
        metrics = label_trajectory_rewards(pred, neigh_future, valid, lanes,
                                           n_horizons=args.n_horizons)
        for k in METRIC_ORDER:
            metric_sums[k] += metrics[k].mean(dim=-1).sum().item()
        n += B

    ade, fde, herr = ade_sum / n, fde_sum / n, herr_sum / n
    means = {k: metric_sums[k] / n for k in METRIC_ORDER}
    # nuPlan-faithful PDMS:
    #   gates  : nc, dac, ddc, mp           (multiplicative)
    #   scored : ttc, ep, comfort, sl       (weighted sum, weights 5/5/2/4 of 16)
    pdms = (means['nc'] * means['dac'] * means['ddc'] * means['mp']) * (
        5.0 * means['ttc'] + 5.0 * means['ep']
        + 2.0 * means['comfort'] + 4.0 * means['sl']
    ) / 16.0

    print()
    print(f"Checkpoint:    {args.planner_ckpt}")
    print(f"Samples seen:  {n} ({n_batches} batches x {args.batch_size})")
    print()
    print(f"  ADE (m):       {ade:.4f}")
    print(f"  FDE (m):       {fde:.4f}")
    print(f"  Heading err:   {herr:.4f}  rad (final step)")
    for hi, (lo, hi_) in enumerate(horizon_edges):
        print(f"  ADE@{lo//10}-{hi_//10}s:    {horizon_sums[hi] / n:.4f}")
    for k in METRIC_ORDER:
        print(f"  {k:14s} {means[k]:.4f}")
    print(f"  proxy-PDMS:    {pdms:.4f}")


if __name__ == '__main__':
    main()
