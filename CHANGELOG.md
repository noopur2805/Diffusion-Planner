# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Repo restructure: replaced `setup.py` with PEP 621 `pyproject.toml`;
  added `[project.scripts]` console-script entrypoints (`dp-train-sft`,
  `dp-train-reward`, `dp-train-grpo`, `dp-eval`, `dp-build-vocab`,
  `dp-preprocess`); split deps into `dev` / `dashboard` / `onnx` extras.
- Moved JSON configs and split manifests from repo root into `configs/`.
- Moved long-form architecture reference from `ARCHITECTURE.md` to
  `docs/architecture.md`.
- Moved PyTorch+CUDA pin file to `requirements/torch-cu118.txt`.

### Added
- `LICENSE` (MIT), `CITATION.cff`, `CHANGELOG.md`.
- `Makefile` with reproduction targets (`make sft`, `make reward`,
  `make vocab`, `make grpo`, `make eval`, `make figures`, `make test`,
  `make lint`).
- `.github/workflows/ci.yml` — ruff + pytest on push / PR.

### Removed
- `diffusion_planner_training.json.bak` (stale backup file).
- `scripts/_tmp_per_type_reactive.py` (one-off diagnostic).
- `onnx/*.log` (build logs do not belong under VCS).

## [1.1.0] — 2026-06-24 — `v8 fix3c` champion

### Added
- Route-masked Drivable-Area Compliance (DAC) in
  `diffusion_planner/reward_labeling.py`: score DAC only against the
  expert-route corridor, threshold tightened 6.0 → 3.0 m.
- Route-tangent Driving-Direction Compliance (DDC): lane tangents
  finite-differenced from the same route-lane set.

### Changed
- Closed-loop reactive PDMS on the 47-scenario nuPlan-mini filter:
  **0.305 (`v7_ent`) → 0.366 (`v8 fix3c`)**, **+72 % vs the SFT
  baseline of 0.213**.
- Champion checkpoint: `runs/grpo_cl8_v7_ent_fix3c/grpo_epoch_1.pth`
  (one GRPO epoch off the new reward, `w_kl=0.02`).

### Known regressions
- `starting_unprotected_cross_turn` (0.153 → 0.000) — 3 m route
  corridor is too tight for turns that legally cross the centreline.
  Variable corridor width by scenario type is the v9 follow-up.

## [1.0.0] — 2026-06-17 — `v7_ent` recovery suite

### Added
- PDMS-weighted reward BCE loss (`TTC=5, EP=5, comfort=2`).
- Trajectory-dispersion entropy regulariser (`w_ent=0.05`).
- Policy-mode CRC recalibration.

### Changed
- 47-scenario reactive PDMS: **0.244 (`v6_r3`) → 0.305 (`v7_ent`)**.

## [0.9.0] — 2026-06-15 — `v6_r3` first closed-loop-aware reward

### Added
- Dense AD-RM critic (8 sub-metrics × 8 horizons) with heteroscedastic
  `(μ, log σ²)` heads.
- Drift-augmented reward labelling (`σ=0.5 m / 0.5 rad`).
- GRPO with σ-damped advantage.

### Changed
- 47-scenario reactive PDMS: **0.213 (SFT) → 0.244**.

## [0.1.0] — upstream baseline

- Forked from Zheng et al., ICLR 2025
  ([2501.15564](https://arxiv.org/abs/2501.15564)).
