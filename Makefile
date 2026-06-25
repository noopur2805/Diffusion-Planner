# Diffusion-Planner — closed-loop RL fine-tune
#
# Reproduction targets. All paths are configurable via environment
# variables (see the README §Reproduce-the-full-pipeline for the full
# matrix). Defaults assume the layout produced by `dp-preprocess`.

SHELL          := /bin/bash

NUPLAN_DATA   ?= $(HOME)/Documents/nuplan
RUN_ROOT      ?= runs
CONFIG_ROOT   ?= configs
CACHE_ROOT    ?= cache

ARGS_SFT      ?= $(RUN_ROOT)/grpo_cl8_v7_ent_fix3c/args.json
CKPT_CHAMPION ?= $(RUN_ROOT)/grpo_cl8_v7_ent_fix3c/grpo_epoch_1.pth

.DEFAULT_GOAL := help

# -----------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------
.PHONY: help
help:
	@echo "Diffusion-Planner — make targets"
	@echo ""
	@echo "  setup       install package + dev extras (editable)"
	@echo "  preprocess  run nuPlan preprocessing → cache/mini_train"
	@echo "  sft         Stage 1: SFT with Shortcut Forcing"
	@echo "  reward      Stage 2: dense reward critic (PDMS-weighted BCE)"
	@echo "  vocab       Stage 3: trajectory vocabulary"
	@echo "  grpo        Stage 4: GRPO fine-tune (champion recipe, v8 fix3c)"
	@echo "  eval        Stage 5: closed-loop reactive (47-scn) simulation"
	@echo "  figures     regenerate per-scenario PDMS bar chart"
	@echo "  onnx        Stage 7: export DiT to ONNX + INT8 quantise"
	@echo "  test        run pytest suite"
	@echo "  lint        run ruff check + format --check"
	@echo "  fmt         run ruff format (writes)"
	@echo "  clean       remove __pycache__, .egg-info, *.pyc"

# -----------------------------------------------------------------------
# Install
# -----------------------------------------------------------------------
.PHONY: setup
setup:
	pip install -r requirements/torch-cu118.txt
	pip install -e ".[dev,dashboard,onnx]"

# -----------------------------------------------------------------------
# Pipeline (thin wrappers; full CLIs documented in README)
# -----------------------------------------------------------------------
.PHONY: preprocess sft reward vocab grpo eval onnx figures

preprocess:
	python data_process.py

sft:
	python train_predictor.py \
	    --normalization_file_path $(CONFIG_ROOT)/normalization.json \
	    --use_shortcut True

reward:
	python train_reward.py \
	    --normalization_file_path $(CONFIG_ROOT)/normalization.json \
	    --use_metric_weights --metric_loss_weights TTC=5,EP=5,comfort=2

vocab:
	python build_vocabulary.py \
	    --train_set_list $(CONFIG_ROOT)/splits/diffusion_planner_training.json

grpo:
	python train_grpo.py \
	    --normalization_file_path $(CONFIG_ROOT)/normalization.json \
	    --epochs 1 --w_bc 1.0 --w_kl 0.02 --w_ent 0.05 \
	    --use_shortcut --use_metric_weights --use_dynamic_vocab \
	    --drift_aug_K 4 --drift_aug_sigma 0.5

eval:
	bash scripts/sh/sim_diffusion_planner_runner.sh

figures:
	python scripts/plot_pdms_breakdown.py
	python scripts/plot_ceiling_exploration.py 2>/dev/null || true

onnx:
	python scripts/export_onnx.py --target encoder
	python scripts/export_onnx.py --target dit
	python scripts/quantize_onnx.py

# -----------------------------------------------------------------------
# Dev
# -----------------------------------------------------------------------
.PHONY: test lint fmt clean
test:
	pytest -q

lint:
	ruff check .
	ruff format --check .

fmt:
	ruff format .

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
