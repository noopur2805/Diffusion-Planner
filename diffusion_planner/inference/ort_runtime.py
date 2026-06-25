"""ONNX Runtime adapters that drop into the PyTorch model at inference time.

Phase 2 wiring. The exported ``encoder_*.onnx`` / ``dit_*.onnx`` graphs are
loaded into ``onnxruntime.InferenceSession`` instances and exposed as ``nn.Module``
shims that match the original PyTorch signatures. Swapping a shim in for the
real submodule lets the rest of the planner stack (samplers, conformal gate,
trajectory post-processing) run unchanged.

The adapters are intentionally CPU-side: tensors are moved to host, fed through
ORT, and the result is returned on the planner's device. This keeps the contract
simple and matches how the closed-loop sim runs (single ego, batch=1, CPU EP).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def _to_numpy(t: torch.Tensor, dtype: np.dtype) -> np.ndarray:
    return t.detach().cpu().numpy().astype(dtype, copy=False)


def _make_session(onnx_path: str, device: str):
    import onnxruntime as ort
    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        # CUDAExecutionProvider is optional; fall back to CPU if not built in.
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(onnx_path, providers=providers)


class OrtEncoderAdapter(nn.Module):
    """Drop-in replacement for ``Diffusion_Planner_Encoder``.

    Consumes the same input dict used by the PyTorch encoder and returns
    ``{"encoding": Tensor[B, T, D]}``. ``scenario_id`` is propagated when
    present so the downstream decoder behaviour is preserved.
    """

    def __init__(self, onnx_path: str, device: str = "cpu") -> None:
        super().__init__()
        self.session = _make_session(onnx_path, device)
        self._device = torch.device(device)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feeds = {
            "neighbor_agents_past": _to_numpy(inputs["neighbor_agents_past"], np.float32),
            "static_objects":       _to_numpy(inputs["static_objects"],       np.float32),
            "lanes":                _to_numpy(inputs["lanes"],                np.float32),
            "lanes_speed_limit":    _to_numpy(inputs["lanes_speed_limit"],    np.float32),
            "lanes_has_speed_limit": _to_numpy(inputs["lanes_has_speed_limit"], np.bool_),
        }
        (encoding,) = self.session.run(["encoding"], feeds)
        out: Dict[str, torch.Tensor] = {
            "encoding": torch.from_numpy(encoding).to(self._device),
        }
        if "scenario_id" in inputs:
            out["scenario_id"] = inputs["scenario_id"].long()
        return out


class OrtDiTAdapter(nn.Module):
    """Drop-in replacement for the ``DiT`` block (single denoising step).

    The shortcut / dpm sampler keeps running in Python; each iteration calls
    this adapter, which dispatches a single ORT inference. Attributes the
    samplers read (``use_shortcut``, ``model_type``, ``_sde``) are mirrored
    from the reference DiT so the surrounding code is untouched.
    """

    def __init__(self, onnx_path: str, ref_dit: nn.Module, device: str = "cpu") -> None:
        super().__init__()
        self.session = _make_session(onnx_path, device)
        self._device = torch.device(device)
        self._use_shortcut = bool(getattr(ref_dit, "use_shortcut", False))
        self._model_type = str(getattr(ref_dit, "model_type", "x_start"))
        self._sde = getattr(ref_dit, "_sde", None)

    @property
    def use_shortcut(self) -> bool:
        return self._use_shortcut

    @property
    def model_type(self) -> str:
        return self._model_type

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cross_c: torch.Tensor,
        route_lanes: torch.Tensor,
        neighbor_current_mask: torch.Tensor,
        d: Optional[torch.Tensor] = None,
        scenario_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if d is None:
            d = torch.ones((t.shape[0],), dtype=t.dtype, device=t.device)
        feeds = {
            "x":                     _to_numpy(x,                     np.float32),
            "t":                     _to_numpy(t,                     np.float32),
            "cross_c":               _to_numpy(cross_c,               np.float32),
            "route_lanes":           _to_numpy(route_lanes,           np.float32),
            "neighbor_current_mask": _to_numpy(neighbor_current_mask, np.bool_),
            "d":                     _to_numpy(d,                     np.float32),
        }
        (out,) = self.session.run(["x_pred"], feeds)
        return torch.from_numpy(out).to(self._device)


def wire_ort_into_planner(
    planner_model: nn.Module,
    encoder_onnx: Optional[str] = None,
    dit_onnx: Optional[str] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Swap PyTorch submodules of a loaded ``Diffusion_Planner`` for ORT adapters.

    Call this AFTER ``load_state_dict`` -- the original tensors are no longer
    needed and we explicitly drop them so the GPU/CPU memory is freed.

    Returns a dict describing which slots were replaced (for logging / tests).
    """
    wired: Dict[str, Any] = {"encoder": False, "dit": False}
    if encoder_onnx is not None:
        planner_model.encoder = OrtEncoderAdapter(encoder_onnx, device=device)
        wired["encoder"] = encoder_onnx
    if dit_onnx is not None:
        ref_dit = planner_model.decoder.decoder.dit
        planner_model.decoder.decoder.dit = OrtDiTAdapter(dit_onnx, ref_dit, device=device)
        wired["dit"] = dit_onnx
    return wired
