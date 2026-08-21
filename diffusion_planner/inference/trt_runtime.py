"""TensorRT adapters that drop into the PyTorch model at inference time.

Mirrors ``ort_runtime.py``'s interface exactly (same forward signatures,
same ``wire_*_into_planner`` shape) so the two backends are interchangeable.
Engines are built by ``scripts/build_tensorrt_engines.py`` from the same
``encoder_*.onnx`` / ``dit_*.onnx`` graphs the ORT adapters use.

Unlike the ORT adapters (CPU-side, numpy round-trip), these adapters bind
CUDA device pointers directly from torch tensors (``Tensor.data_ptr()``)
into the TensorRT execution context, so tensors never leave the GPU.
fp16 engines expect fp16 I/O (built with ``keep_io_types=False``); the
adapter casts to/from the engine's declared per-tensor dtype automatically.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import tensorrt as trt
import torch
import torch.nn as nn

_TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
_RUNTIME = trt.Runtime(_TRT_LOGGER)

_TRT_TO_TORCH_DTYPE = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.BOOL: torch.bool,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
}


def _load_engine(engine_path: str) -> trt.ICudaEngine:
    with open(engine_path, "rb") as f:
        engine = _RUNTIME.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
    return engine


class _TrtSession:
    """Thin wrapper: one engine + one execution context, CUDA-only."""

    def __init__(self, engine_path: str) -> None:
        self.engine = _load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.input_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
        ]

    def tensor_dtype(self, name: str) -> torch.dtype:
        return _TRT_TO_TORCH_DTYPE[self.engine.get_tensor_dtype(name)]

    def run(self, feeds: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        stream = torch.cuda.Stream()
        outputs: Dict[str, torch.Tensor] = {}
        for name, tensor in feeds.items():
            if -1 in tuple(self.engine.get_tensor_shape(name)):
                self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            out = torch.empty(shape, dtype=self.tensor_dtype(name), device="cuda")
            outputs[name] = out
            self.context.set_tensor_address(name, out.data_ptr())
        with torch.cuda.stream(stream):
            ok = self.context.execute_async_v3(stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        stream.synchronize()
        return outputs


def _cast_feed(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return t.to(device="cuda", dtype=dtype).contiguous()


class TrtEncoderAdapter(nn.Module):
    """Drop-in replacement for ``Diffusion_Planner_Encoder`` (TensorRT backend)."""

    def __init__(self, engine_path: str, device: str = "cuda") -> None:
        super().__init__()
        assert device == "cuda", "TensorRT adapters require a CUDA device"
        self.session = _TrtSession(engine_path)
        self._device = torch.device(device)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        names = [
            "neighbor_agents_past", "static_objects", "lanes",
            "lanes_speed_limit", "lanes_has_speed_limit",
        ]
        feeds = {n: _cast_feed(inputs[n], self.session.tensor_dtype(n)) for n in names}
        outs = self.session.run(feeds)
        out: Dict[str, torch.Tensor] = {
            "encoding": outs["encoding"].to(self._device),
        }
        if "scenario_id" in inputs:
            out["scenario_id"] = inputs["scenario_id"].long()
        return out


class TrtDiTAdapter(nn.Module):
    """Drop-in replacement for the ``DiT`` block (TensorRT backend, single step)."""

    def __init__(self, engine_path: str, ref_dit: nn.Module, device: str = "cuda") -> None:
        super().__init__()
        assert device == "cuda", "TensorRT adapters require a CUDA device"
        self.session = _TrtSession(engine_path)
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
        names = ["x", "t", "cross_c", "route_lanes", "neighbor_current_mask", "d"]
        values = [x, t, cross_c, route_lanes, neighbor_current_mask, d]
        feeds = {n: _cast_feed(v, self.session.tensor_dtype(n)) for n, v in zip(names, values)}
        outs = self.session.run(feeds)
        return outs["x_pred"].to(dtype=x.dtype, device=self._device)


def wire_trt_into_planner(
    planner_model: nn.Module,
    encoder_engine: Optional[str] = None,
    dit_engine: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Swap PyTorch submodules of a loaded ``Diffusion_Planner`` for TensorRT adapters.

    Call this AFTER ``load_state_dict`` -- mirrors ``wire_ort_into_planner``.
    Returns a dict describing which slots were replaced (for logging / tests).
    """
    wired: Dict[str, Any] = {"encoder": False, "dit": False}
    if encoder_engine is not None:
        planner_model.encoder = TrtEncoderAdapter(encoder_engine, device=device)
        wired["encoder"] = encoder_engine
    if dit_engine is not None:
        ref_dit = planner_model.decoder.decoder.dit
        planner_model.decoder.decoder.dit = TrtDiTAdapter(dit_engine, ref_dit, device=device)
        wired["dit"] = dit_engine
    return wired
