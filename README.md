# Diffusion-Planner — Extension On Top Of Upstream

This fork extends the official **Diffusion-Planner** (ICLR 2025) with a
**DreamerAD-style** RL fine-tuning stack and adds three extensions on top:

1. **Vectorized (token-space) latent world model** — engineering port of
   DreamerAD's pixel-space video-DiT into the planner's scene-token
   space, ~485 K params, runs at the planner's native rate.
2. **Heteroscedastic AD-RM head** — `(μ, log σ²)` per (horizon, metric)
   via the Kendall–Gal (NIPS 2017) recipe, giving every reward a
   calibrated confidence.
3. **Uncertainty-weighted GRPO advantage** — `A ← A / (1 + τ · σ̄)`,
   the smooth regularized form of inverse-variance weighting. This is
   the cleanest algorithmic contribution and is testable in isolation
   against vanilla GRPO via a `τ = 0` ablation.

All upstream files (DiT planner, SDE math, encoder/decoder, data
loaders) are preserved. Every extension is gated by a flag and defaults
to off, so the upstream behavior is recovered when nothing is enabled.

> **Honest accounting.** The pipeline as a whole is a re-implementation
> of DreamerAD on top of the upstream Diffusion-Planner. See
> [§ Honest accounting](#honest-accounting) below for what is borrowed
> from prior work versus what is genuinely original.

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

<a id="honest-accounting"></a>

## Honest accounting — what is borrowed, what is mine

The full pipeline is divided into three layers: borrowed from prior
work, engineering work I did to wire it together, and genuinely new
contributions.

### Borrowed (faithful re-implementation)

| Component | Origin |
|---|---|
| Shortcut Forcing self-distillation on a dyadic grid | Frans et al., NeurIPS 2024 |
| AD-RM architecture (5 metrics × 8 horizons, BCE) | DreamerAD |
| GRPO (clipped surrogate + group z-score advantage) | DeepSeekMath 2024, adopted by DreamerAD |
| Trajectory vocabulary (filter + uniform sample, g1+g2 sampling, log-σ + log-sum aggregation) | DreamerAD |
| BC anchor + KL-to-frozen-SFT-ref regularization | DreamerAD + standard PPO practice |
| Concept of a learned world model feeding the reward critic | DreamerAD (theirs is pixel-space) |
| Heteroscedastic Gaussian NLL `½(log σ² + e²/σ²)` | Kendall & Gal, NIPS 2017 |

### Engineering work (not research contributions on their own)

- Wiring all of the above into the upstream Diffusion-Planner codebase
  without breaking its behavior when the new flags are disabled.
- The proxy PDM labeler in `reward_labeling.py` — a practical
  convenience so the AD-RM can be trained without the NavSim PDM
  simulator. This is arguably a *limitation* of the prototype, not an
  advance.
- NavSim 8-camera fusion encoder, CPU-validated end-to-end smoke
  pipeline, the 49-test suite, the uncertainty visualizer.

### Original contributions

#### A. Uncertainty-weighted GRPO advantage (strongest)

```python
# diffusion_planner/grpo.py — one line that changes the GRPO update
advantages = group_advantage(rewards)
if sigma is not None:
    advantages = advantages / (1.0 + uncertainty_temp * cand_unc)
```

Where `cand_unc` is the mean of the AD-RM's predicted σ over
(horizon, metric) per candidate.

**Why this matters.** Vanilla GRPO's advantage is the group z-score of
the predicted reward. When the reward model is noisy and
heteroscedastic — confident on some candidates, unreliable on others —
a single high-σ candidate with a spuriously inflated reward produces a
large advantage and corrupts the policy gradient. PPO clipping does
not fix this, because it bounds the importance ratio, not the
advantage magnitude.

The proposed update damps the advantage of each candidate by its
predicted σ. The form `1 / (1 + τσ)` is sign-preserving (the policy
still knows better/worse), bounded in `(0, 1]`, smooth everywhere, and
reduces to vanilla GRPO at `τ = 0`. It is the smooth, regularized
form of **inverse-variance weighting** — the classical BLUE estimator
under Gaussian noise.

**What it does:** addresses the well-known noisy-reward failure mode
of vanilla GRPO. **What it does *not* do:** fix reward-model bias,
reward hacking, or purely epistemic uncertainty (σ here is aleatoric).

**Caveats.**
- It is one line of code. The empirical effect must be demonstrated by
  an ablation (`τ = 0` vs `τ > 0`, ≥ 3 seeds, mean ± std PDMS).
- The exact functional form is one choice among several plausible ones
  (`exp(−τσ)`, `1 / (τ + σ)`, etc.); picking it without sweeping is a
  weakness.
- This claim is contingent on DreamerAD not already doing a
  σ-weighted advantage. The reader should verify this directly in the
  paper before publishing.

#### B. Heteroscedastic head on the AD-RM (conditional)

The output head goes from 1 channel (BCE logit `μ`) to 2 channels
(`μ`, `log σ²`). `μ` is BCE-supervised; `log σ²` is fit by the
Kendall–Gal Gaussian NLL on the squared residual of `sigmoid(μ)`,
giving a calibrated `(B, H, 5)` σ-map per trajectory.

```
L = BCE(μ, y)  +  w_unc · ½(log σ²  +  (sigmoid(μ).detach() − y)² / exp(log σ²))
```

The Kendall–Gal recipe is standard; the contribution is its
**application** to the per-(horizon × metric) dense reward model in
the context of driving RL. It is the prerequisite for contribution A
— without σ, there is no advantage scaling — so the two stand or fall
together.

This claim is also contingent on DreamerAD's AD-RM not already
emitting `(μ, log σ²)`; the reader should check the paper for
"uncertainty", "aleatoric", "variance", "calibration".

#### C. Vectorized (token-space) world model (engineering port)

`LatentWorldModel` is a 2-layer cross-attention transformer
(~485 K params) operating in the planner's scene-token space. It
takes the planner's `(B, N, D)` scene tokens and a candidate
trajectory, builds per-horizon `(action_emb + horizon_emb)`
conditioning, and refines per-horizon copies of the scene by
cross-attending back to the *original* scene tokens. Output:
`(B, H, N, D)`. The AD-RM's 4-D context path then attends to these
per-horizon imagined latents instead of a static present-time scene.

The cross-attention-back-to-original-context is the inductive bias
that keeps the predictor anchored — each horizon's tokens can evolve
in time but must keep referencing the observed scene, so the predictor
cannot hallucinate freely.

**Honest framing.** The concept ("imagine futures with a learned world
model and feed them to the reward critic") is DreamerAD's, not mine.
The contribution is the realization in token space rather than pixel
space, which makes the approach runnable at the planner's native
inference rate. This is an engineering adaptation, not a research
novelty. To upgrade it, an ablation must show per-horizon imagined
latents beat a shared-token baseline.

---

## What this solves, and how

| Problem | How it is addressed | Origin |
|---|---|---|
| Imitation-learning ceiling | Add GRPO RL fine-tuning on top of SFT | Integration of DreamerAD's recipe |
| Multi-step diffusion is too slow for 20 Hz planning | Shortcut Forcing self-distillation → 1-step inference | Frans et al., integrated here |
| Pixel-space world models are too heavy for vectorized planners | Vectorized token-space `LatentWorldModel` | Original engineering port (C) |
| Noisy reward labels destabilize GRPO via spurious advantages | Heteroscedastic σ on the AD-RM (B) used to scale GRPO advantages (A) | Original contributions A + B |

---

## What it would take to upgrade these to publication-strength

1. **Verify the DreamerAD paper** for any prior use of variance- or
   σ-weighted advantage, or any `(μ, log σ²)` head on the AD-RM. If
   either exists, the corresponding contribution must be withdrawn.
2. **Run the four-variant ablation** on real PDMS:
   - SFT only
   - + GRPO (no extensions)
   - + GRPO + uncertainty
   - + GRPO + uncertainty + latent predictor
   Across ≥ 3 seeds, report mean ± std.
3. **Add a reliability diagram** for the AD-RM (bucket predictions by
   σ, plot empirical accuracy per bucket) to verify calibration.
4. **Sweep `τ`** over `{0, 0.5, 1, 2, 4}` and plot the PDMS-vs-τ curve.

---

## Notes

The proxy reward in `reward_labeling.py` is intentionally lightweight so
the AD-RM can be trained without the NavSim PDM simulator. Swap the
function for a real PDM scorer to match the paper's numbers.
