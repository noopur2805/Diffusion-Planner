# Diffusion-Planner — Extension On Top Of Upstream

This fork extends the official **Diffusion-Planner** (ICLR 2025) with a
**DreamerAD-style** RL fine-tuning stack and adds seven extensions on
top:

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
4. **Uncertainty-weighted horizon aggregation** — per-horizon reward
   damped by `1 / (1 + τ_h · σ_h)` before the sum, concentrating signal
   on horizons where the AD-RM is confident.
5. **Context-conditional reward priorities** — a `MetricWeightHead`
   emits per-scene per-metric weights `w ∈ R^K` (e.g. down-weight comfort
   in tight intersections).
6. **Dynamic trajectory vocabulary** — wraps the static vocabulary so
   winning candidates are admitted and low-utility entries evicted, with
   a utility tracker.
7. **Speed-adaptive horizons (A1 + B)** — horizon indices picked by
   uniform fractions of cumulative path *distance* (A1), and a continuous
   `Mlp(2 → D)` embedding on `(τ_time_sec, τ_dist_m)` so the network
   sees the physical meaning of each horizon (B).
8. **σ → horizon coupling (7a)** — under (4), replace `σ_h` by
   `cummax(σ_h)` along the horizon axis; damping becomes monotone
   non-increasing, so any later horizon is damped at least as strongly
   as the most uncertain prior one.

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
                                  ├── + LatentWorldModel    │
                                  ├── + Uncertainty head    │
                                  ├── + MetricWeightHead    │
                                  ├── + Adaptive horizons ──┤
                                  │                         ├── + σ-weighted advantage
                                  │                         ├── + horizon-σ damping
                                  │                         ├── + cumulative-max σ
                                  │                         └── + dynamic vocabulary
```

---

## What was added

### Model

| File | Purpose |
|---|---|
| `diffusion_planner/model/latent_predictor.py` | `LatentWorldModel` — cross-attention transformer that conditions on `(scene_tokens, candidate_trajectory)` and emits `(B, H, N, D)` per-horizon scene latents. `adaptive_horizons=True` switches sampling to cumulative-distance and adds the continuous `(τ_t, τ_d)` MLP. |
| `diffusion_planner/model/reward_model.py` | `AutoregressiveDenseRewardModel` — DreamerAD AD-RM with 5 metrics × 8 horizons; optional `predict_uncertainty=True` flag emits `(μ, log σ²)`. Supports both 3-D and 4-D context, and the same `adaptive_horizons` / `dt` knobs. |
| `diffusion_planner/model/metric_weight_head.py` | `MetricWeightHead` — pools the scene encoding and emits per-metric weights `w ∈ R^K` consumed by `total_reward_from_dense`. |
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
| `diffusion_planner/utils/trajectory_vocabulary.py` | `filter_by_endstate`, `build_vocabulary`, `gaussian_vocab_sample` (g1 discriminative + g2 neighborhood), `total_reward_from_dense` aggregation (now supports `metric_weights`, `horizon_uncertainty_temp`, `cumulative_uncertainty`), and `DynamicVocabulary` with utility-based eviction. |
| `diffusion_planner/utils/navsim_dataset.py` | `NavSimMultiModalData` — adds 8-camera images + valid-mask to each sample. |
| `diffusion_planner/utils/ddp.py` | Non-DDP early-return so `torch.distributed.barrier()` is skipped on single-GPU/CPU runs. |

### Top-level scripts

| File | Purpose |
|---|---|
| `train_reward.py` | Train AD-RM on a frozen SFT planner. Flags: `--use_latent_predictor`, `--latent_layers`, `--predict_uncertainty`, `--w_uncertainty`, `--use_metric_weights`, `--w_metric_margin`, `--adaptive_horizons`, `--dt`. Saves `{model, predict_uncertainty, adaptive_horizons, dt, latent_predictor?, latent_layers?, metric_weight_head?}`. |
| `train_grpo.py` | GRPO fine-tune. Auto-detects uncertainty / latent / metric-weight / adaptive-horizon flags from the reward checkpoint. Flags: `--uncertainty_temp`, `--horizon_uncertainty_temp`, `--cumulative_uncertainty`, `--use_metric_weights`, `--use_dynamic_vocab`. When σ is available: `A ← A / (1 + τ·σ̄_cand)`. |
| `build_vocabulary.py` | One-shot vocabulary builder from GT futures of the training corpus. |
| `scripts/visualize_uncertainty.py` | Diagnostic — per-scene heatmaps of mean reward and uncertainty + candidate trajectories colored by mean σ. |
| `scripts/run_ablation.py` | Orchestrate the {baseline, +adaptive, +horizon-σ, +both, +cummax} ablation matrix across seeds. Dry-run by default; `--execute` to launch. |
| `scripts/diagnose_horizon_error.py` | AD-RM mean `\|p − target\|` as a function of `(ego_speed, horizon)`; produces per-checkpoint heatmaps and an optional side-by-side fixed-vs-adaptive comparison. |

---

## Reproducing the pipeline

```bash
# 0. Tests
pytest tests/ -v

# 1. SFT (add --use_shortcut for 1-step inference)
python train_predictor.py --train_set $TRAIN_SET --train_set_list $LIST \
    --use_shortcut --shortcut_k_max 16 --save_dir runs/sft

# 2. AD-RM with all reward-side novelties
python train_reward.py --planner_ckpt runs/sft/latest.pth \
    --train_set $TRAIN_SET --train_set_list $LIST \
    --use_latent_predictor --latent_layers 2 \
    --predict_uncertainty --w_uncertainty 0.1 \
    --use_metric_weights --w_metric_margin 0.1 \
    --adaptive_horizons --dt 0.1 \
    --save_dir runs/reward_full

# 3. Vocabulary
python build_vocabulary.py --train_set $TRAIN_SET --train_set_list $LIST \
    --out_path runs/vocab.pt --max_size 8192

# 4. GRPO (uncertainty-aware advantage + horizon damping + cumulative σ)
python train_grpo.py --planner_ckpt runs/sft/latest.pth \
    --reward_ckpt runs/reward_full/reward_epoch_12.pth \
    --vocab_path runs/vocab.pt \
    --uncertainty_temp 1.0 --horizon_uncertainty_temp 1.0 \
    --cumulative_uncertainty \
    --use_metric_weights --use_dynamic_vocab \
    --train_set $TRAIN_SET --train_set_list $LIST \
    --save_dir runs/grpo

# 5. Visualize the calibrated reward model
python scripts/visualize_uncertainty.py \
    --planner_ckpt runs/sft/latest.pth \
    --reward_ckpt runs/reward_full/reward_epoch_12.pth \
    --train_set $TRAIN_SET --train_set_list $LIST \
    --n_scenes 4 --n_candidates 16 --out_dir runs/vis

# 6. Ablation matrix + horizon-error diagnostic
python scripts/run_ablation.py \
    --planner_ckpt runs/sft/latest.pth \
    --predictor_ckpt runs/sft/latest.pth \
    --train_set $TRAIN_SET --train_set_list $LIST \
    --vocab_path runs/vocab.pt --seeds 0 1 2 \
    --out_root runs/ablation                # add --execute to launch

python scripts/diagnose_horizon_error.py \
    --planner_ckpt runs/sft/latest.pth \
    --reward_ckpt runs/ablation/baseline/reward/reward_epoch_12.pth \
    --reward_ckpt_adaptive runs/ablation/adaptive/reward/reward_epoch_12.pth \
    --train_set $TRAIN_SET --train_set_list $LIST \
    --out_dir runs/horizon_diag
```

---

## Backward compatibility

* All new flags default to **off**. Running `train_predictor.py` /
  `train_reward.py` / `train_grpo.py` without any of them reproduces the
  original behavior, verified by suite-wide bit-identity tests.
* Reward checkpoints without uncertainty have `predict_uncertainty=False`
  and no `latent_predictor` key in the saved dict — `train_grpo.py`
  auto-detects this and falls back to plain GRPO.
* `adaptive_horizons` and `dt` are persisted into the reward checkpoint
  and auto-read by `train_grpo.py`; the same applies to the optional
  `metric_weight_head` state dict.
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
  pipeline, the 87-test suite, the uncertainty visualizer, the
  ablation runner, and the horizon-error diagnostic.

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

#### D. Uncertainty-weighted horizon aggregation

The aggregator `total_reward_from_dense` is extended so each per-horizon
term is damped by `1 / (1 + τ_h · σ_h)` before summation, where `σ_h`
is the AD-RM's per-horizon σ (averaged over metrics). `τ_h = 0`
recovers the original uniform-horizon sum. This is the *horizon-axis*
analogue of contribution A and shares its motivation: place more weight
on the prediction targets the reward critic is calibrated to.

**Caveat.** The mechanism is symmetric across candidates within a
batch, so it changes the relative scale of horizons, not the relative
scale of candidates. It does not by itself address noisy-reward
failure of GRPO — A is still required.

#### E. σ → horizon coupling via cumulative-max damping

Under (D), replace `σ_h` by `cummax(σ_h)` along the horizon axis before
the damping factor is applied. The resulting damping factor is
monotone non-increasing in horizon index: any time the AD-RM becomes
uncertain at horizon `h`, all later horizons are damped at least as
strongly. This enforces the natural prior that long-horizon predictions
should not be trusted more than the intermediate state on which they
are conditioned.

**What it does:** removes the "uncertain spike followed by a confident
distant horizon" pathology that otherwise leaks back into the gradient.
**What it does not do:** improve calibration of the AD-RM itself — it
is a post-hoc safety net on the aggregator, not on the reward head.

#### F. Context-conditional reward priorities (`MetricWeightHead`)

A small head on the pooled scene encoding emits per-scene per-metric
weights `w ∈ R^K` that multiply the safety log-terms and the task-sum
inside the aggregator. Trained with a GT-beats-perturbed margin loss,
so the weights are constrained to make the ground-truth trajectory
score higher than its Gaussian-perturbed neighbours. Lets the network
locally re-weight (e.g.) comfort vs. drivable-area compliance based on
scene context.

**Caveat.** The margin loss is a weak supervision signal; the head can
underfit on small batches. An ablation against fixed uniform weights
is required to claim it helps.

#### G. Dynamic trajectory vocabulary with utility-based eviction

The static vocabulary (build once, freeze) is wrapped in a
`DynamicVocabulary` that, at each GRPO step, admits the top-reward
candidate as a new entry and evicts the lowest-utility entry once the
buffer is full. Utility is an EMA over the candidate's recent
selection frequency. The g1 / g2 sampling protocol is unchanged.

**Caveat.** Risks reward hacking if the AD-RM is miscalibrated — the
buffer will accumulate entries that please the critic rather than
entries that improve PDMS. Should be paired with (A) + (E) for safety.

#### H. Speed-adaptive horizons (A1 + B)

Two coupled changes inside `reward_model.py` and `latent_predictor.py`:

- **A1** — horizon indices are picked by uniform fractions of the
  candidate's cumulative path *distance* instead of uniform fractions
  of the time axis. At highway speed the same `H` samples cover more
  meters; at low speed they bunch where the motion happens.
- **B** — the discrete `nn.Embedding(H, D)` step embedding is augmented
  with a continuous `Mlp(2 → D)` ingesting per-horizon
  `(τ_time_sec, τ_dist_m)`. The reward model and the world model now
  see the physical meaning of each horizon, not just its index.

This is the classical-control intuition `L_d = k · v + L_f` from pure
pursuit (Coulter 1992; Macenski 2023) lifted into the AD-RM /
world-model conditioning. Backward-compatible behind `--adaptive_horizons`
(verified bit-identical when off).

**Caveat.** A1 changes *what* the network is asked to predict; B
changes *what conditioning* it gets. Either alone has an obvious
failure mode; the contribution is the pairing. The empirical case
must be made by the (fixed vs adaptive) `(ego_speed, horizon)` error
heatmap produced by `scripts/diagnose_horizon_error.py`.

---

## What this solves, and how

| Problem | How it is addressed | Origin |
|---|---|---|
| Imitation-learning ceiling | Add GRPO RL fine-tuning on top of SFT | Integration of DreamerAD's recipe |
| Multi-step diffusion is too slow for 20 Hz planning | Shortcut Forcing self-distillation → 1-step inference | Frans et al., integrated here |
| Pixel-space world models are too heavy for vectorized planners | Vectorized token-space `LatentWorldModel` | Original engineering port (C) |
| Noisy reward labels destabilize GRPO via spurious advantages | Heteroscedastic σ on the AD-RM (B) used to scale GRPO advantages (A) | Original contributions A + B |
| Distant horizons dominate the reward sum despite being least trusted | Per-horizon damping by σ (D) with monotone-cummax safety net (E) | Original contributions D + E |
| Scene-blind metric weighting (e.g. comfort in tight intersections) | Context-conditional `MetricWeightHead` (F) | Original contribution F |
| Static vocabulary becomes stale as the policy drifts | `DynamicVocabulary` with utility-based eviction (G) | Original contribution G |
| Fixed-time horizons mis-allocate samples across speeds | Adaptive horizons + continuous `(τ_t, τ_d)` conditioning (H) | Original contribution H |

---

## What it would take to upgrade these to publication-strength

1. **Verify the DreamerAD paper** for any prior use of variance- or
   σ-weighted advantage, or any `(μ, log σ²)` head on the AD-RM. If
   either exists, the corresponding contribution must be withdrawn.
2. **Run the ablation matrix** on real PDMS using `scripts/run_ablation.py`:
   - SFT only
   - + GRPO (no extensions)
   - + GRPO + advantage σ-weighting (A)
   - + GRPO + advantage + horizon-σ (A + D)
   - + GRPO + above + cumulative σ (A + D + E)
   - + GRPO + above + adaptive horizons (A + D + E + H)
   - + GRPO + above + metric weights + dynamic vocab (A–G + H)
   Across ≥ 3 seeds, report mean ± std PDMS.
3. **Add a reliability diagram** for the AD-RM (bucket predictions by
   σ, plot empirical accuracy per bucket) to verify calibration.
4. **Sweep `τ` and `τ_h`** independently over `{0, 0.5, 1, 2, 4}` and
   plot the PDMS surfaces.
5. **Run the `(ego_speed, horizon)` heatmap** from
   `scripts/diagnose_horizon_error.py` on a held-out validation split
   for the fixed-vs-adaptive comparison.
6. **Probe reward hacking under (G)** by tracking the fraction of
   admitted vocabulary entries whose PDMS is below the cohort median.

---

## Notes

The proxy reward in `reward_labeling.py` is intentionally lightweight so
the AD-RM can be trained without the NavSim PDM simulator. Swap the
function for a real PDM scorer to match the paper's numbers.
