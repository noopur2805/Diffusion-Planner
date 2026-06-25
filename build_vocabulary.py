"""
Build a trajectory vocabulary from the preprocessed nuPlan/NavSim corpus.

The vocabulary is a (V, T, 3) tensor of (x, y, heading) future trajectories,
collected across all training scenes and (optionally) sub-sampled so the file
size stays small. ``train_grpo.py`` consumes the output via ``--vocab_path``.

Usage::

    python build_vocabulary.py \
        --train_set /path/to/processed \
        --train_set_list diffusion_planner_training.json \
        --out_path vocabulary.pt \
        --max_size 8192

Note: ``--train_set_list`` must point at the list of cached scenario ``.npz``
files (produced by ``data_process.py`` as ``diffusion_planner_training.json``),
NOT the nuPlan log-name filter (``nuplan_train.json``).
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from diffusion_planner.utils.dataset import DiffusionPlannerData


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_set', required=True)
    p.add_argument('--train_set_list', required=True)
    p.add_argument('--out_path', required=True)
    p.add_argument('--future_len', type=int, default=80)
    p.add_argument('--agent_num', type=int, default=32)
    p.add_argument('--predicted_neighbor_num', type=int, default=10)
    p.add_argument('--max_size', type=int, default=8192)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=256)
    args = p.parse_args()

    ds = DiffusionPlannerData(args.train_set, args.train_set_list,
                              args.agent_num, args.predicted_neighbor_num, args.future_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    collected = []
    for batch in loader:
        ego_future = batch[1]  # (B, T, 3) -> x, y, heading
        collected.append(ego_future)
        if sum(c.shape[0] for c in collected) >= args.max_size:
            break

    vocab = torch.cat(collected, dim=0)[: args.max_size]
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)) or ".", exist_ok=True)
    torch.save(vocab, args.out_path)
    print(f"Saved vocabulary of shape {tuple(vocab.shape)} to {args.out_path}")


if __name__ == "__main__":
    main()
