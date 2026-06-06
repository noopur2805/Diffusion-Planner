# Diffusion-Planner — Extension On Top Of Upstream

This fork extends the official **Diffusion-Planner** (ICLR 2025) with a
**DreamerAD-style** RL stack and two original research contributions:

1. **Vectorized Latent World Model** — per-horizon "imagined" scene tokens
   that replace DreamerAD's pixel-based video-DiT rollout.
2. **Uncertainty-Aware Dense Reward Model** — heteroscedastic head
   predicting `(μ, log σ²)` per (horizon, metric) so GRPO can down-weight
   advantages from uncertain rewards.

All upstream files (the DiT planner, SDE math, encoder/decoder, data
loaders) are preserved. New code is additive and feature-flagged.

---

## Pipeline overview

```
SFT (train_predictor.py) ─► AD-RM (train_reward.py) ─► GRPO (train_grpo.py)
       │                          │                         │
       └── Shortcut Forcing  ─────┤                         │
                                  └── + LatentWorldModel ───┤
                                  └── + Uncertainty head  ──┘
```

---

## What was added

### Model

| File | Purpose |
|---|---|
| `diffusion_planner/model/latent_predictor.py` | `LatentWorldModel` — cross-attention transformer that conditions on `(scene_tokens, candidate_trajectory)` and emits `(B, H, N, D)` per-horizon scene latents. |
| `diffusion_planner/model/reward_model.py` | `AutoregressiveDenseRewardModel` — DreamerAD AD-RM with 5 metrics × 8 horizons; optional `predict_uncertainty=True` flag emits `(μ, log σ²)`. Supports both 3-D and 4-D context. |
| `diffusion_planner/model/module/camera_encoder.py` | `CameraEncoder` — timm-backbone 8-view encoder for NavSim, fused as extra scene tokens. |
| `diffusion_planner/model/module/dit.py` | Added `StepEmbedder` to encode the Shortcut step-size `d ∈ (0, 1]`; `DiTBlock` already supports the extra conditioning. |
| `diffusion_planner/model/module/decoder.py` | `use_shortcut` flag → `shortcut_sampler` at inference (1–few step sampling). |
| `diffusion_planner/model/diffusion_utils/sampling.py` | New `shortcut_sampler` — anchored re-noising via `marginal_prob` between dyadic time steps. |

### Loss / training

| File | Purpose |
|---|---|
| `diffusion_planner/loss.py` | New `shortcut_loss_func` — Shortcut Forcing self-distillation on a dyadic grid `d ∈ {1/16, …, 1}` (Frans et al., 2024), adapted to the VP-SDE / x_start parameterization. |
| `diffusion_planner/train_epoch.py` | Branches on `args.use_shortcut`; pipes camera tensors when `args.use_camera`. |
| `diffusion_planner/grpo.py` | `grpo_actor_loss`, `policy_kl`, `group_advantage`, `diag_gauss_logprob`, `grpo_total_loss` (clipped surrogate + BC + KL to a frozen SFT ref policy). |
| `diffusion_planner/reward_labeling.py` | Vectorized **proxy PDM** scorer: `nc`, `dac`, `ttc`, `ep`, `comfort` straight from the input tensors. Pluggable for real PDM later. |
| `diffusion_planner/utils/trajectory_vocabulary.py` | `filter_by_endstate`, `build_vocabulary`, `gaussian_vocab_sample` (g1 discriminative + g2 neighborhood), `total_reward_from_dense` aggregation. |
| `diffusion_planner/utils/navsim_dataset.py` | `NavSimMultiModalData` — adds 8-camera images + valid-mask to each sample. |
| `diffusion_planner/utils/ddp.py` | Non-DDP early-return so `torch.distributed.barrier()` is skipped on single-GPU/CPU runs. |

### Top-level scripts

| File | Purpose |
|---|---|
| `train_reward.py` | Train AD-RM on a frozen SFT planner. Flags: `--use_latent_predictor`, `--latent_layers`, `--predict_uncertainty`, `--w_uncertainty`. Saves `{model, predict_uncertainty, latent_predictor?, latent_layers?}`. |
| `train_grpo.py` | GRPO fine-tune. Auto-detects uncertainty/latent flags from the reward checkpoint. When σ is available: `A ← A / (1 + τ·σ̄_cand)`. |
| `build_vocabulary.py` | One-shot vocabulary builder from GT futures of the training corpus. |
| `scripts/visualize_uncertainty.py` | Diagnostic — per-scene heatmaps of mean reward and uncertainty + candidate trajectories colored by mean σ. |

### Tests

`tests/` — 49 tests across 6 suites, all passing:

| Suite | Tests |
|---|---|
| `test_shortcut_forcing` | 12 |
| `test_grpo` | 10 |
| `test_latent_predictor` | 9 |
| `test_reward_model` | 6 |
| `test_camera_encoder` | 6 |
| `test_trajectory_vocabulary` | 6 |

---

## Reproducing the pipeline

```bash
# 0. Tests
pytest tests/ -v

# 1. SFT (add --use_shortcut for 1-step inference)
python train_predictor.py --train_set $TRAIN_SET --train_set_list $LIST \
    --use_shortcut --shortcut_k_max 16 --save_dir runs/sft

# 2. AD-RM with both novelties
python train_reward.py --planner_ckpt runs/sft/latest.pth \
    --train_set $TRAIN_SET --train_set_list $LIST \
    --use_latent_predictor --latent_layers 2 \
    --predict_uncertainty --w_uncertainty 0.1 \
    --save_dir runs/reward_full

# 3. Vocabulary
python build_vocabulary.py --train_set $TRAIN_SET --train_set_list $LIST \
    --out_path runs/vocab.pt --max_size 8192

# 4. GRPO (uncertainty-aware advantage scaling auto-enabled)
python train_grpo.py --planner_ckpt runs/sft/latest.pth \
    --reward_ckpt runs/reward_full/reward_epoch_12.pth \
    --vocab_path runs/vocab.pt --uncertainty_temp 1.0 \
    --train_set $TRAIN_SET --train_set_list $LIST \
    --save_dir runs/grpo

# 5. Visualize the calibrated reward model
python scripts/visualize_uncertainty.py \
    --planner_ckpt runs/sft/latest.pth \
    --reward_ckpt runs/reward_full/reward_epoch_12.pth \
    --train_set $TRAIN_SET --train_set_list $LIST \
    --n_scenes 4 --n_candidates 16 --out_dir runs/vis
```

---

## Backward compatibility

* All new flags default to **off**. Running `train_predictor.py` /
  `train_reward.py` without them reproduces the original behavior.
* Reward checkpoints without uncertainty have `predict_uncertainty=False`
  and no `latent_predictor` key in the saved dict — `train_grpo.py`
  auto-detects this and falls back to plain GRPO.
* The CPU path is fully supported (`--device cpu`); was used to validate
  the end-to-end smoke run on an RTX 5060 (sm_120) machine where the
  installed PyTorch build does not yet support the GPU.

---

## Notes

The proxy reward in `reward_labeling.py` is intentionally lightweight so
the AD-RM can be trained without the NavSim PDM simulator. Swap the
function for a real PDM scorer to match the paper's numbers.
