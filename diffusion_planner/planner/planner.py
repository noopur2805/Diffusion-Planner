
import math
import sys
import time
import warnings
import torch
import numpy as np
from typing import Deque, Dict, List, Type

warnings.filterwarnings("ignore")

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.observation_type import Observation, DetectionsTracks
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput
)

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.model.reward_model import AutoregressiveDenseRewardModel, METRIC_NAMES
from diffusion_planner.reward_labeling import METRIC_ORDER
from diffusion_planner.data_process.data_processor import DataProcessor
from diffusion_planner.planner.momentum import (
    TrajectoryMomentumBuffer,
    select_by_hausdorff,
    route_dac_mask,
    route_heading_mask,
)
from diffusion_planner.safety import ConformalPredictor, constant_velocity_fallback, pdms_from_metrics
from diffusion_planner.utils.config import Config

def identity(ego_state, predictions):
    return predictions


class DiffusionPlanner(AbstractPlanner):
    # Class-level counter so per-scenario instances share the same numbering.
    _scenario_counter = 0

    def __init__(
            self,
            config: Config,
            ckpt_path: str,

            past_trajectory_sampling: TrajectorySampling,
            future_trajectory_sampling: TrajectorySampling,

            enable_ema: bool = True,
            device: str = "cpu",
            # Novelty 8 -- momentum-aware inference (all default off)
            use_ttm: bool = False,
            warm_start_alpha: float = 0.0,
            n_samples: int = 1,
            # Conformal Risk Control safety tier (opt-in; default off).
            use_conformal_safety: bool = False,
            reward_ckpt: str = None,
            conformal_calibration_path: str = None,
            tau_safe: float = 0.5,
            fallback_decel: float = 0.0,
            # Phase 2 -- ONNX Runtime backends (opt-in; PyTorch by default).
            use_onnx: bool = False,
            onnx_encoder_path: str = None,
            onnx_dit_path: str = None,
            # Route-DAC veto (Fix #2): drop off-route candidates before TTM rerank.
            use_route_dac_veto: bool = False,
            route_dac_threshold_m: float = 2.5,
            route_dac_min_frac: float = 0.75,
            # Route heading veto (Fix #3a): drop wrong-way candidates whose
            # local heading disagrees with the nearest route-lane tangent.
            use_route_heading_veto: bool = False,
            route_heading_tol_deg: float = 60.0,
            route_heading_min_frac: float = 0.75,
            route_heading_dist_m: float = 5.0,
        ):

        assert device in ["cpu", "cuda"], f"device {device} not supported"
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"

        self._future_horizon = future_trajectory_sampling.time_horizon # [s]
        self._step_interval = future_trajectory_sampling.time_horizon / future_trajectory_sampling.num_poses # [s]

        self._config = config
        self._ckpt_path = ckpt_path

        self._past_trajectory_sampling = past_trajectory_sampling
        self._future_trajectory_sampling = future_trajectory_sampling

        self._ema_enabled = enable_ema
        self._device = device

        self._planner = Diffusion_Planner(config)

        self.data_processor = DataProcessor(config)

        self.observation_normalizer = config.observation_normalizer

        # Momentum-aware inference state (Novelty 8). The buffer is created in
        # ``initialize`` so a fresh scenario starts with no prior plan.
        self._use_ttm = bool(use_ttm)
        self._warm_start_alpha = float(warm_start_alpha)
        self._n_samples = max(1, int(n_samples))
        self._momentum_buffer: TrajectoryMomentumBuffer = None

        # Per-instance progress state (nuPlan spawns one planner per scenario).
        self._scenario_steps = 0
        self._scenario_start_t = 0.0

        # Conformal Risk Control state. Heavy artifacts (reward model, YAML)
        # are loaded in ``initialize`` so unit tests that construct the
        # planner without checkpoints still work.
        self._use_conformal_safety = bool(use_conformal_safety)
        self._reward_ckpt = reward_ckpt
        self._conformal_calibration_path = conformal_calibration_path
        self._tau_safe = float(tau_safe)
        self._fallback_decel = float(fallback_decel)
        self._reward_model: AutoregressiveDenseRewardModel = None
        self._conformal: ConformalPredictor = None
        self._safety_tier_counts = {"dit": 0, "fallback": 0}
        # Running log of ``lb_pdms`` for monitor-only diagnostics.
        self._lb_pdms_log: List[float] = []

        # Phase 2 -- ONNX Runtime backends. Paths are kept here and resolved
        # in ``initialize`` after the PyTorch state_dict has been loaded.
        self._use_onnx = bool(use_onnx)
        self._onnx_encoder_path = onnx_encoder_path
        self._onnx_dit_path = onnx_dit_path

        # Route-DAC veto state.
        self._use_route_dac_veto = bool(use_route_dac_veto)
        self._route_dac_threshold_m = float(route_dac_threshold_m)
        self._route_dac_min_frac = float(route_dac_min_frac)
        self._route_dac_counts = {"kept": 0, "vetoed": 0, "skipped": 0}

        # Route heading veto state (Fix #3a).
        self._use_route_heading_veto = bool(use_route_heading_veto)
        self._route_heading_tol_rad = float(route_heading_tol_deg) * math.pi / 180.0
        self._route_heading_min_frac = float(route_heading_min_frac)
        self._route_heading_dist_m = float(route_heading_dist_m)
        self._route_heading_counts = {"kept": 0, "vetoed": 0, "skipped": 0}

    def name(self) -> str:
        """
        Inherited.
        """
        return "diffusion_planner"
    
    def observation_type(self) -> Type[Observation]:
        """
        Inherited.
        """
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        Inherited.
        """
        # Print a per-scenario header so long closed-loop runs show progress.
        # Counter is class-level because nuPlan creates a fresh planner per scenario.
        DiffusionPlanner._scenario_counter += 1
        self._scenario_steps = 0
        self._scenario_start_t = time.time()
        sys.stdout.write(
            f"\n[diffusion_planner] scenario #{DiffusionPlanner._scenario_counter} starting "
        )
        sys.stdout.flush()

        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids

        if self._ckpt_path is not None:
            state_dict:Dict = torch.load(self._ckpt_path, map_location=self._device, weights_only=False)
            
            if self._ema_enabled:
                state_dict = state_dict['ema_state_dict']
            else:
                if "model" in state_dict.keys():
                    state_dict = state_dict['model']
            # Strip "module." DDP prefix only when present; otherwise pass keys through.
            model_state_dict = {
                (k[len("module."):] if k.startswith("module.") else k): v
                for k, v in state_dict.items()
            }
            self._planner.load_state_dict(model_state_dict)
        else:
            print("load random model")
        
        self._planner.eval()
        self._planner = self._planner.to(self._device)

        # Phase 2 -- swap PyTorch submodules for ORT-backed adapters. Done
        # after load_state_dict so the exported graphs replace fully-trained
        # weights rather than the randomly initialised slots.
        if self._use_onnx and (self._onnx_encoder_path or self._onnx_dit_path):
            from diffusion_planner.inference import wire_ort_into_planner
            wired = wire_ort_into_planner(
                self._planner,
                encoder_onnx=self._onnx_encoder_path,
                dit_onnx=self._onnx_dit_path,
                device=self._device,
            )
            print(f"[diffusion_planner] ONNX runtime on: {wired}", flush=True)

        self._initialization = initialization

        # Fresh momentum buffer per scenario.
        self._momentum_buffer = TrajectoryMomentumBuffer(
            predicted_neighbor_num=self._config.predicted_neighbor_num,
            future_len=self._config.future_len,
        )

        # Lazy-load the conformal safety stack on the first scenario; reuse
        # across subsequent scenarios (the reward model and YAML never change).
        if self._use_conformal_safety and self._reward_model is None:
            self._load_conformal_safety()

    def _load_conformal_safety(self) -> None:
        """Build the reward model and load its weights + calibration YAML.

        Called once on the first ``initialize`` when
        ``use_conformal_safety=True``. Missing artifacts raise immediately
        so misconfigured runs fail fast rather than silently skip the gate.
        """
        if not self._reward_ckpt or not self._conformal_calibration_path:
            raise ValueError(
                "use_conformal_safety=True requires both reward_ckpt and "
                "conformal_calibration_path to be set."
            )
        cfg = self._config
        ck = torch.load(self._reward_ckpt, map_location=self._device, weights_only=False)
        sd = ck.get("model", ck)
        sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
        # Autodetect arch flags from the checkpoint so the constructor matches
        # the shapes on disk. The reward head's final linear is shape (2, D)
        # when trained with aleatoric uncertainty (mu + log_var) and (1, D)
        # otherwise; adaptive_horizons adds a step_proj submodule.
        head_w = sd.get("head.3.weight")
        predict_uncertainty = bool(head_w is not None and head_w.shape[0] == 2)
        adaptive_horizons = any(k.startswith("step_proj.") for k in sd.keys())
        rm = AutoregressiveDenseRewardModel(
            hidden_dim=cfg.hidden_dim,
            traj_dim=4,
            n_horizons=getattr(cfg, "n_horizons", 8),
            metric_names=METRIC_NAMES,
            predict_uncertainty=predict_uncertainty,
            adaptive_horizons=adaptive_horizons,
            dt=getattr(cfg, "dt", 0.1),
        )
        rm.load_state_dict(sd, strict=False)
        rm.eval().to(self._device)
        self._reward_model = rm
        self._conformal = ConformalPredictor.from_yaml(self._conformal_calibration_path)
        print(
            f"[diffusion_planner] conformal safety on: alpha={self._conformal.alpha} "
            f"delta_pdms={self._conformal.delta_pdms:.4f} tau_safe={self._tau_safe}",
            flush=True,
        )

    @torch.no_grad()
    def _conformal_is_unsafe(
        self,
        inputs: Dict[str, torch.Tensor],
        prediction: torch.Tensor,
    ) -> bool:
        """Score ``prediction`` with the reward model and apply the CRC gate.

        ``prediction`` is the planner's ``(K, P, T, 4)`` output. We score
        the ego candidate only (P index 0), take the mean over the
        horizon axis, aggregate to PDMS, and compare the lower confidence
        bound to ``tau_safe``. The raw ``lb_pdms`` is also appended to a
        rolling log for monitor-only diagnostics (independent of whether
        the gate fires).
        """
        if self._reward_model is None or self._conformal is None:
            return False
        ctx = self._planner.encoder(inputs)["encoding"]
        ego_traj = prediction[:1, 0]                                 # (1, T, 4)
        out = self._reward_model(ctx[:1], ego_traj)
        mu = out[0] if isinstance(out, tuple) else out
        p_hat = torch.sigmoid(mu).mean(dim=1)                        # (1, K)
        per_metric = {m: p_hat[:, i] for i, m in enumerate(METRIC_ORDER)}
        p_pdms = pdms_from_metrics(per_metric)                       # (1,)
        lb_pdms = float(self._conformal.lower_bound_pdms(p_pdms).item())
        self._lb_pdms_log.append(lb_pdms)
        return lb_pdms < float(self._tau_safe)

    def planner_input_to_model_inputs(self, planner_input: PlannerInput) -> Dict[str, torch.Tensor]:
        history = planner_input.history
        traffic_light_data = list(planner_input.traffic_light_data)
        model_inputs = self.data_processor.observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)

        return model_inputs

    def outputs_to_trajectory(self, outputs: Dict[str, torch.Tensor], ego_state_history: Deque[EgoState]) -> List[InterpolatableState]:    

        predictions = outputs['prediction'][0, 0].detach().cpu().numpy().astype(np.float64) # T, 4
        heading = np.arctan2(predictions[:, 3], predictions[:, 2])[..., None]
        predictions = np.concatenate([predictions[..., :2], heading], axis=-1) 

        states = transform_predictions_to_states(predictions, ego_state_history, self._future_horizon, self._step_interval)

        return states
    
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Inherited.
        """
        # Lightweight progress dot every 20 steps (~2 s of sim) so the console
        # confirms the run is alive during 30+ min closed-loop sweeps.
        self._scenario_steps += 1
        if self._scenario_steps % 20 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
        if (
            self._use_conformal_safety
            and self._scenario_steps % 100 == 0
            and len(self._lb_pdms_log) >= 10
        ):
            lb = torch.tensor(self._lb_pdms_log[-100:])
            n_fire = self._safety_tier_counts["fallback"]
            n_tot = self._safety_tier_counts["dit"] + n_fire
            sys.stdout.write(
                f"\n[crc] step={self._scenario_steps} lb_pdms "
                f"p5={lb.quantile(0.05).item():.3f} p50={lb.quantile(0.50).item():.3f} "
                f"p95={lb.quantile(0.95).item():.3f}   "
                f"fallback={n_fire}/{n_tot} ({(n_fire/max(n_tot,1)):.1%})\n"
            )
            sys.stdout.flush()
        if self._use_route_dac_veto and self._scenario_steps % 100 == 0:
            c = self._route_dac_counts
            tot = c["kept"] + c["vetoed"]
            sys.stdout.write(
                f"\n[dac] step={self._scenario_steps} kept={c['kept']} "
                f"vetoed={c['vetoed']} ({(c['vetoed']/max(tot,1)):.1%}) "
                f"skipped={c['skipped']}\n"
            )
            sys.stdout.flush()
        if self._use_route_heading_veto and self._scenario_steps % 100 == 0:
            c = self._route_heading_counts
            tot = c["kept"] + c["vetoed"]
            sys.stdout.write(
                f"\n[head] step={self._scenario_steps} kept={c['kept']} "
                f"vetoed={c['vetoed']} ({(c['vetoed']/max(tot,1)):.1%}) "
                f"skipped={c['skipped']}\n"
            )
            sys.stdout.flush()

        inputs = self.planner_input_to_model_inputs(current_input)

        # Capture pre-normalisation route lanes (ego frame, metres) for the
        # route-DAC veto below. ``observation_normalizer`` returns a copy, so
        # this reference stays in physical units.
        route_lanes_raw = None
        if self._use_route_dac_veto and "route_lanes" in inputs:
            route_lanes_raw = inputs["route_lanes"].detach().clone()

        inputs = self.observation_normalizer(inputs)

        # Momentum-aware inference (Novelty 8). When all three flags are at
        # their defaults the inputs dict is untouched and behaviour matches
        # upstream.
        ego_state = current_input.history.ego_states[-1]
        ego_pose_world = (
            float(ego_state.rear_axle.x),
            float(ego_state.rear_axle.y),
            float(ego_state.rear_axle.heading),
        )
        momentum_on = self._warm_start_alpha > 0.0 or (self._use_ttm and self._n_samples > 1)
        anchor = None
        if momentum_on and self._momentum_buffer is not None and self._momentum_buffer.has_anchor:
            anchor = self._momentum_buffer.get_anchor(ego_pose_world, device=torch.device(self._device))
            if anchor is not None and self._warm_start_alpha > 0.0:
                inputs["prev_plan_anchor"] = anchor
                inputs["warm_start_alpha"] = self._warm_start_alpha
        if self._use_ttm and self._n_samples > 1:
            inputs["n_samples"] = self._n_samples

        _, outputs = self._planner(inputs)

        prediction = outputs["prediction"]  # (K, P, T, 4) physical units, current ego frame
        if prediction.shape[0] > 1:
            cands_xy = prediction[:, 0, :, :2]
            # Route-DAC veto (Fix #2): drop candidates that leave the route
            # corridor for more than (1 - min_frac) of the horizon. Skip when
            # no route lanes were extracted (mask all-zero) so we never veto
            # the full set.
            keep_mask = None
            if (self._use_route_dac_veto or self._use_route_heading_veto) and route_lanes_raw is not None:
                rl = route_lanes_raw[0, :, :, :2].to(cands_xy.device)  # (M, P, 2)
                K = cands_xy.shape[0]
                dist_mask = torch.ones(K, dtype=torch.bool, device=cands_xy.device)
                head_mask = torch.ones(K, dtype=torch.bool, device=cands_xy.device)
                if self._use_route_dac_veto:
                    m = route_dac_mask(
                        cands_xy, rl,
                        threshold_m=self._route_dac_threshold_m,
                        min_frac=self._route_dac_min_frac,
                    )
                    if bool(m.any()):
                        dist_mask = m
                        self._route_dac_counts["kept"] += int(m.sum().item())
                        self._route_dac_counts["vetoed"] += int((~m).sum().item())
                    else:
                        self._route_dac_counts["skipped"] += 1
                if self._use_route_heading_veto:
                    m = route_heading_mask(
                        cands_xy, rl,
                        heading_tol_rad=self._route_heading_tol_rad,
                        min_frac=self._route_heading_min_frac,
                        dist_threshold_m=self._route_heading_dist_m,
                    )
                    if bool(m.any()):
                        head_mask = m
                        self._route_heading_counts["kept"] += int(m.sum().item())
                        self._route_heading_counts["vetoed"] += int((~m).sum().item())
                    else:
                        self._route_heading_counts["skipped"] += 1
                combined = dist_mask & head_mask
                if bool(combined.any()):
                    keep_mask = combined
            if anchor is not None and self._use_ttm:
                anchor_xy = anchor[0, 0, :, :2].to(prediction.device)
                if keep_mask is not None:
                    kept_idx = torch.nonzero(keep_mask, as_tuple=False).flatten()
                    sub_idx = select_by_hausdorff(cands_xy[kept_idx], anchor_xy)
                    best_idx = int(kept_idx[sub_idx].item())
                else:
                    best_idx = select_by_hausdorff(cands_xy, anchor_xy)
            else:
                if keep_mask is not None:
                    best_idx = int(torch.nonzero(keep_mask, as_tuple=False)[0].item())
                else:
                    best_idx = 0
            prediction = prediction[best_idx:best_idx + 1]
            outputs = {**outputs, "prediction": prediction}

        # Conformal safety gate (opt-in). Scores the chosen DiT plan with
        # the calibrated reward model and swaps in a defensive fallback
        # when the PDMS lower confidence bound drops below ``tau_safe``.
        safety_tier = "dit"
        if self._use_conformal_safety and self._reward_model is not None:
            if self._conformal_is_unsafe(inputs, prediction):
                prediction = constant_velocity_fallback(
                    ego_state=ego_state,
                    future_len=self._config.future_len,
                    dt=self._step_interval,
                    decel=self._fallback_decel,
                    device=prediction.device,
                    dtype=prediction.dtype,
                )
                outputs = {**outputs, "prediction": prediction}
                safety_tier = "fallback"
            self._safety_tier_counts[safety_tier] += 1

        # Persist the chosen ego plan for the next call.
        if self._momentum_buffer is not None:
            ego_plan = prediction[0, 0].detach().cpu().numpy().astype(np.float32)
            self._momentum_buffer.update(ego_plan, ego_pose_world)

        trajectory = InterpolatedTrajectory(
            trajectory=self.outputs_to_trajectory(outputs, current_input.history.ego_states)
        )

        return trajectory
    