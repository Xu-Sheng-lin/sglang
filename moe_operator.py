#!/usr/bin/env python3
"""FlashInfer-compatible SM103 CuTeDSL NVFP4 MoE operator.

The public function mirrors `flashinfer.trtllm_fp4_block_scale_moe`, with an
SM103 arch guard and CuTeDSL custom path enabled by default.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import List, Optional

import torch


SUPPORTED_HIDDEN = 4096
FOCUS_M_VALUES = (8576, 8320, 8064, 7680, 9500)
SUPPORTED_INTERMEDIATES = frozenset({1024})
SUPPORTED_EXPERTS = 512
SUPPORTED_TOP_K = 10
SUPPORTED_V3_FAMILY_CONFIGS = frozenset(
    {
        (SUPPORTED_HIDDEN, SUPPORTED_EXPERTS, SUPPORTED_TOP_K),
        (3072, 256, 8),
    }
)
SUPPORTED_ROUTING_METHOD = 1
SUPPORTED_ACTIVATION = 3
PIPELINES = ("v1", "v2", "v3", "v4")
DEFAULT_PIPELINE = "v3"
V3_FAMILY_PIPELINES = ("v3", "v4")
_V2_C2_MMA_TILER_MN = (128, 192)
_V2_C2_CLUSTER_SHAPE_MN = (1, 1)
_V2_C2_SM_COUNT = 128
_V3_C1_MMA_TILER_MN = (256, 256)
_V3_C2_MMA_TILER_MN = (128, 256)
_V3_C1_CLUSTER_SHAPE_MN = (2, 1)
_V3_C2_CLUSTER_SHAPE_MN = (1, 2)
_V3_C1_FALLBACK_MMA_TILER_MN = (128, 256)
_V3_C1_FALLBACK_CLUSTER_SHAPE_MN = (1, 1)
_V3_C1_M128_MAX_M = 7168


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class CachedCutedslWeights:
    c1_weights: torch.Tensor
    c1_weights_scale_linear: torch.Tensor
    c1_weights_scale: torch.Tensor
    c2_weights: torch.Tensor
    c2_weights_scale: torch.Tensor
    c1_weights_gate_up_block_pair: Optional[torch.Tensor] = None
    c1_weights_gate_up_block_pair_scale: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class V3CutedslWeights:
    c1_weights_gate_up_block_pair: torch.Tensor
    c1_weights_gate_up_block_pair_scale: torch.Tensor
    c2_weights: torch.Tensor
    c2_weights_scale: torch.Tensor


@dataclass(frozen=True)
class V4CutedslWeights:
    c1_weights_trt_preshuffled: torch.Tensor
    c1_weights_scale_trt_preshuffled: torch.Tensor
    c2_weights_trt_preshuffled: torch.Tensor
    c2_weights_scale_trt_preshuffled: torch.Tensor


_WEIGHT_CACHE: dict[tuple[object, ...], CachedCutedslWeights] = {}
_V3_WEIGHT_CACHE: dict[tuple[object, ...], V3CutedslWeights] = {}
_V4_WEIGHT_CACHE: dict[tuple[object, ...], V4CutedslWeights] = {}
_TENSOR_CACHE: dict[tuple[object, ...], torch.Tensor] = {}
_UNIT_SCALE_CACHE: dict[tuple[object, ...], bool] = {}


_ENV_PREFIX = "ATREX_NVFP4_FUSED_MOE_SM103"
_DEFAULT_ENV = {
    "CUDA_COMPACT": "1",
    "CUDA_COMPACT_MAX_M": "512",
    "CUDA_COMPACT_PHYSICAL_SCALE": "1",
    "CUDA_SCALE": "1",
    "CUDA_FINALIZE": "1",
    "CACHE_WEIGHTS": "1",
    "PRESWAP_C1": "1",
    "FUSED_C1": "1",
    "FINALIZE_VEC4": "1",
    "COMPACT_UNINIT": "1",
    "CUDA_TOPK": "1",
    "V3_FUSED_ROUTE": "1",
}

_ENV_SUFFIXES = (
    "ENABLE",
    "CUDA_COMPACT",
    "CUDA_COMPACT_MAX_M",
    "CUDA_COMPACT_PHYSICAL_SCALE",
    "CUDA_SCALE",
    "CUDA_FINALIZE",
    "CACHE_WEIGHTS",
    "PRESWAP_C1",
    "FLASHINFER_TOPK",
    "FUSED_C1",
    "FINALIZE_VEC4",
    "COMPACT_UNINIT",
    "CUDA_TOPK",
    "GEMM_MMA_MN",
    "GEMM_CLUSTER_MN",
    "C1_MMA_MN",
    "C1_CLUSTER_MN",
    "C2_MMA_MN",
    "C2_CLUSTER_MN",
    "CONTIGUOUS_DEBUG_SYNC_CHECKS",
    "V3_FUSED_ROUTE",
    "V3_C1_M128_MAX_M",
)


def _env_key(suffix: str) -> str:
    return f"{_ENV_PREFIX}_{suffix}"


def _env_get(suffix: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(_env_key(suffix), _DEFAULT_ENV.get(suffix, default))


def _env_flag(suffix: str) -> bool:
    return _env_get(suffix, "0") == "1"


_LEGACY_ENV_SYNCED = False


def _sync_legacy_env() -> None:
    """One-time sync of legacy env var names to canonical names."""
    global _LEGACY_ENV_SYNCED
    if _LEGACY_ENV_SYNCED:
        return
    for suffix in _ENV_SUFFIXES:
        value = _env_get(suffix)
        if value is not None:
            os.environ[_env_key(suffix)] = value
    _LEGACY_ENV_SYNCED = True


def _cache_first_ok(func):
    """Cache the first GuardResult with ok=True so later calls short-circuit."""
    _cached: list = [None]

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if _cached[0] is not None:
            return _cached[0]
        r = func(*args, **kwargs)
        if getattr(r, "ok", False):
            _cached[0] = r
        return r

    return wrapper


def _flashinfer_call(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_bias: Optional[torch.Tensor],
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm2_bias: Optional[torch.Tensor],
    output1_scale_scalar: Optional[torch.Tensor],
    output1_scale_gate_scalar: Optional[torch.Tensor],
    output2_scale_scalar: Optional[torch.Tensor],
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    activation_type: int = SUPPORTED_ACTIVATION,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
    norm_topk_prob: bool = True,
    routing_replay_out: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    import flashinfer

    return flashinfer.trtllm_fp4_block_scale_moe(
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_bias,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        gemm2_weights,
        gemm2_weights_scale,
        gemm2_bias,
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        do_finalize,
        enable_pdl,
        activation_type,
        output,
        tune_max_num_tokens,
        norm_topk_prob,
        routing_replay_out,
    )


@_cache_first_ok
def guard_supported_cutedsl_moe_call(
    routing_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    num_experts: int,
    top_k: int,
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routing_method_type: int,
    do_finalize: bool,
    activation_type: int,
    *,
    routing_bias: Optional[torch.Tensor] = None,
    output1_scale_scalar: Optional[torch.Tensor] = None,
    output1_scale_gate_scalar: Optional[torch.Tensor] = None,
    output2_scale_scalar: Optional[torch.Tensor] = None,
    n_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    enable_pdl: Optional[bool] = None,
    output: Optional[torch.Tensor] = None,
    norm_topk_prob: bool = True,
    routing_replay_out: Optional[torch.Tensor] = None,
    pipeline: str = DEFAULT_PIPELINE,
    v3_prepared_weights: Optional[object] = None,
) -> GuardResult:
    if not torch.cuda.is_available():
        return GuardResult(False, "CUDA is not available")
    if hidden_states.device.type != "cuda":
        return GuardResult(False, "hidden_states must be CUDA")
    if torch.cuda.get_device_capability(hidden_states.device) != (10, 3):
        return GuardResult(False, "only L20D/SM103 is enabled")
    if hidden_states.dtype != torch.uint8:
        return GuardResult(False, "hidden_states must be packed NVFP4 uint8")
    if hidden_states_scale is None or hidden_states_scale.device.type != "cuda":
        return GuardResult(False, "hidden_states_scale must be a CUDA tensor")
    if hidden_states_scale.dtype != torch.float8_e4m3fn:
        return GuardResult(False, "hidden_states_scale must be FP8 E4M3")
    if routing_logits.device.type != "cuda":
        return GuardResult(False, "routing_logits must be CUDA")
    if hidden_states.ndim != 2:
        return GuardResult(False, f"hidden_states must be rank 2, got {hidden_states.ndim}")
    hidden_size = int(hidden_states.shape[1]) * 2
    config = (hidden_size, int(num_experts), int(top_k))
    if pipeline in V3_FAMILY_PIPELINES:
        if config not in SUPPORTED_V3_FAMILY_CONFIGS:
            return GuardResult(
                False,
                "unsupported v3/v4 (hidden_size,num_experts,top_k) configuration: "
                f"{config}; supported: {sorted(SUPPORTED_V3_FAMILY_CONFIGS)}",
            )
    elif config != (SUPPORTED_HIDDEN, SUPPORTED_EXPERTS, SUPPORTED_TOP_K):
        return GuardResult(
            False,
            f"pipeline={pipeline!r} only supports "
            f"({SUPPORTED_HIDDEN},{SUPPORTED_EXPERTS},{SUPPORTED_TOP_K})",
        )
    if routing_bias is not None:
        if routing_bias.device.type != "cuda" or routing_bias.shape != (num_experts,):
            return GuardResult(
                False,
                f"routing_bias must be CUDA [{num_experts}] when provided",
            )
    if pipeline == "v4":
        if not _env_flag("V3_FUSED_ROUTE"):
            return GuardResult(False, "pipeline='v4' requires V3_FUSED_ROUTE=1")
        if routing_bias is not None:
            return GuardResult(False, "pipeline='v4' requires routing_bias=None")
        if routing_logits.dtype != torch.bfloat16:
            return GuardResult(False, "pipeline='v4' requires bfloat16 routing_logits")
    if routing_logits.shape != (hidden_states.shape[0], num_experts):
        return GuardResult(False, f"routing_logits shape is unsupported: {tuple(routing_logits.shape)}")
    if hidden_states_scale.shape != (hidden_states.shape[0], hidden_size // 16):
        return GuardResult(False, f"hidden scale shape is unsupported: {tuple(hidden_states_scale.shape)}")
    if local_num_experts != num_experts:
        return GuardResult(
            False,
            f"only full local {num_experts}-expert scope is enabled",
        )
    if local_expert_offset != 0:
        return GuardResult(False, "local_expert_offset must be 0")
    if intermediate_size not in SUPPORTED_INTERMEDIATES:
        return GuardResult(False, f"intermediate_size must be one of {sorted(SUPPORTED_INTERMEDIATES)}")
    if routing_method_type != SUPPORTED_ROUTING_METHOD:
        return GuardResult(False, "routing_method_type must be 1")
    if activation_type != SUPPORTED_ACTIVATION:
        return GuardResult(False, "activation_type must be SwiGLU=3")
    if not do_finalize:
        return GuardResult(False, "do_finalize=False parity is not implemented yet")
    if n_group not in (None, 0) or topk_group not in (None, 0):
        return GuardResult(False, "grouped top-k routing is not enabled for the custom path")
    if enable_pdl not in (None, False):
        return GuardResult(False, "enable_pdl=True is not enabled for the custom path")
    if not norm_topk_prob:
        return GuardResult(False, "norm_topk_prob=False is not enabled for routing_method_type=1")
    for name, scalar in (
        ("output1_scale_scalar", output1_scale_scalar),
        ("output1_scale_gate_scalar", output1_scale_gate_scalar),
        ("output2_scale_scalar", output2_scale_scalar),
    ):
        if scalar is None or scalar.device.type != "cuda" or scalar.shape != (num_experts,):
            return GuardResult(False, f"{name} must be CUDA [{num_experts}]")
        if scalar.dtype != torch.float32:
            return GuardResult(False, f"{name} must be float32")
        if not scalar.is_contiguous():
            return GuardResult(False, f"{name} must be contiguous")
    if pipeline not in V3_FAMILY_PIPELINES and not (
        _is_unit_scale_tensor(output1_scale_scalar)
        and _is_unit_scale_tensor(output1_scale_gate_scalar)
        and _is_unit_scale_tensor(output2_scale_scalar)
    ):
        return GuardResult(
            False,
            f"pipeline={pipeline!r} currently requires unit output scale tensors; "
            "use pipeline='v3'/'v4' or pass fallback=True for non-unit output scales",
        )
    if output is not None:
        if output.device.type != "cuda" or output.shape != (hidden_states.shape[0], hidden_size):
            return GuardResult(
                False,
                f"output must be CUDA [M,{hidden_size}] when provided",
            )
        if output.dtype != torch.bfloat16:
            return GuardResult(False, "output must be bfloat16")
        if pipeline == "v4" and not output.is_contiguous():
            return GuardResult(False, "pipeline='v4' output must be contiguous")
    if routing_replay_out is not None:
        if routing_replay_out.device.type != "cuda":
            return GuardResult(False, "routing_replay_out must be CUDA")
        if routing_replay_out.shape != (hidden_states.shape[0], top_k):
            return GuardResult(False, "routing_replay_out must be [M,top_k]")
        if routing_replay_out.dtype != torch.int16:
            return GuardResult(False, "routing_replay_out must be int16")
    if v3_prepared_weights is not None:
        if pipeline not in V3_FAMILY_PIPELINES:
            return GuardResult(False, "v3_prepared_weights requires pipeline='v3' or 'v4'")
        return _validate_v3_family_prepared_weights(
            v3_prepared_weights,
            pipeline=pipeline,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            num_experts=num_experts,
            device=hidden_states.device,
        )
    if gemm1_weights.shape != (num_experts, 2 * intermediate_size, hidden_size // 2):
        return GuardResult(False, f"gemm1_weights shape is unsupported: {tuple(gemm1_weights.shape)}")
    if gemm1_weights_scale.shape != (num_experts, 2 * intermediate_size, hidden_size // 16):
        return GuardResult(False, f"gemm1_weights_scale shape is unsupported: {tuple(gemm1_weights_scale.shape)}")
    if gemm1_weights.dtype != torch.uint8:
        return GuardResult(False, "gemm1_weights must be packed NVFP4 uint8")
    if gemm1_weights_scale.dtype != torch.float8_e4m3fn:
        return GuardResult(False, "gemm1_weights_scale must be FP8 E4M3")
    if gemm2_weights.shape != (num_experts, hidden_size, intermediate_size // 2):
        return GuardResult(False, f"gemm2_weights shape is unsupported: {tuple(gemm2_weights.shape)}")
    if gemm2_weights_scale.shape != (num_experts, hidden_size, intermediate_size // 16):
        return GuardResult(False, f"gemm2_weights_scale shape is unsupported: {tuple(gemm2_weights_scale.shape)}")
    if gemm2_weights.dtype != torch.uint8:
        return GuardResult(False, "gemm2_weights must be packed NVFP4 uint8")
    if gemm2_weights_scale.dtype != torch.float8_e4m3fn:
        return GuardResult(False, "gemm2_weights_scale must be FP8 E4M3")
    return GuardResult(True)


def _custom_enabled(explicit: Optional[bool]) -> bool:
    if explicit is not None:
        return explicit
    return True


def _check_pipeline(pipeline: str) -> str:
    if pipeline not in PIPELINES:
        raise ValueError(f"unsupported SM103 NVFP4 pipeline {pipeline!r}; valid values: {', '.join(PIPELINES)}")
    return pipeline


def _parse_pair_env(name: str, default: tuple[int, int]) -> tuple[int, int]:
    suffix = name.removeprefix(f"{_ENV_PREFIX}_")
    value = _env_get(suffix)
    if not value:
        return default
    parts = value.replace("x", ",").split(",")
    if len(parts) != 2:
        raise ValueError(f"{name} must be formatted as M,N or MxN")
    return (int(parts[0]), int(parts[1]))


def _interleave_gate_up_blocks(
    tensor: torch.Tensor,
    *,
    block: int = 64,
    preswapped_c1: bool,
    output_order: str = "gate_up",
) -> torch.Tensor:
    if output_order not in {"gate_up", "up_gate"}:
        raise ValueError(f"unsupported C1 block-pair output_order={output_order}")
    rows = int(tensor.shape[1])
    if rows % 2 != 0:
        raise ValueError(f"row dimension must be even, got {rows}")
    half = rows // 2
    if half % block != 0:
        raise ValueError(f"half row dimension {half} must be a multiple of {block}")
    first_half = tensor[:, :half]
    second_half = tensor[:, half:]
    if preswapped_c1:
        gate, up = first_half, second_half
    else:
        gate, up = second_half, first_half
    chunks: list[torch.Tensor] = []
    for start in range(0, half, block):
        if output_order == "gate_up":
            chunks.append(gate[:, start : start + block])
            chunks.append(up[:, start : start + block])
        else:
            chunks.append(up[:, start : start + block])
            chunks.append(gate[:, start : start + block])
    return torch.cat(chunks, dim=1).contiguous()


@functools.lru_cache(maxsize=32)
def _gemm_tuning(
    stage: str,
    *,
    default_mma_tiler_mn: tuple[int, int] = (128, 128),
    default_cluster_shape_mn: tuple[int, int] = (1, 1),
) -> tuple[tuple[int, int], tuple[int, int]]:
    default_mma = _parse_pair_env(f"{_ENV_PREFIX}_GEMM_MMA_MN", default_mma_tiler_mn)
    default_cluster = _parse_pair_env(
        f"{_ENV_PREFIX}_GEMM_CLUSTER_MN",
        default_cluster_shape_mn,
    )
    mma = _parse_pair_env(f"{_ENV_PREFIX}_{stage}_MMA_MN", default_mma)
    cluster = _parse_pair_env(f"{_ENV_PREFIX}_{stage}_CLUSTER_MN", default_cluster)
    return mma, cluster


def _has_c1_tuning_override() -> bool:
    return any(
        os.environ.get(_env_key(suffix)) is not None
        for suffix in (
            "GEMM_MMA_MN",
            "GEMM_CLUSTER_MN",
            "C1_MMA_MN",
            "C1_CLUSTER_MN",
        )
    )


def _cached_ones(
    tag: str,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (
        tag,
        shape,
        device.type,
        None if device.index is None else int(device.index),
        str(dtype),
    )
    cached = _TENSOR_CACHE.get(key)
    if cached is None or cached.device != device:
        cached = torch.ones(shape, device=device, dtype=dtype)
        _TENSOR_CACHE[key] = cached
    return cached


def _tensor_cache_key(tensor: torch.Tensor) -> tuple[object, ...]:
    device = tensor.device
    return (
        int(tensor.data_ptr()),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        device.type,
        None if device.index is None else int(device.index),
    )


def _tensor_identity_key(tensor: torch.Tensor) -> tuple[object, ...]:
    return _tensor_cache_key(tensor) + (int(getattr(tensor, "_version", 0)),)


def _is_unit_scale_tensor(tensor: torch.Tensor) -> bool:
    key = _tensor_identity_key(tensor)
    cached = _UNIT_SCALE_CACHE.get(key)
    if cached is not None:
        return cached
    is_unit = bool(torch.all(tensor == 1.0).item())
    _UNIT_SCALE_CACHE[key] = is_unit
    return is_unit


def _cutedsl_scale_shape(
    *,
    experts: int,
    mn: int,
    sf_k: int,
) -> tuple[int, int, int, int, int, int]:
    return (32, 4, (mn + 127) // 128, 4, (sf_k + 3) // 4, experts)


@_cache_first_ok
def _validate_v3_cutedsl_weights(
    weights: object,
    *,
    intermediate_size: int,
    hidden_size: int,
    num_experts: int,
    device: torch.device,
) -> GuardResult:
    required = (
        "c1_weights_gate_up_block_pair",
        "c1_weights_gate_up_block_pair_scale",
        "c2_weights",
        "c2_weights_scale",
    )
    missing = [name for name in required if getattr(weights, name, None) is None]
    if missing:
        return GuardResult(False, f"v3 prepared weights missing fields: {', '.join(missing)}")

    c1 = getattr(weights, "c1_weights_gate_up_block_pair")
    c1_scale = getattr(weights, "c1_weights_gate_up_block_pair_scale")
    c2 = getattr(weights, "c2_weights")
    c2_scale = getattr(weights, "c2_weights_scale")
    tensors = (
        ("c1_weights_gate_up_block_pair", c1),
        ("c1_weights_gate_up_block_pair_scale", c1_scale),
        ("c2_weights", c2),
        ("c2_weights_scale", c2_scale),
    )
    for name, tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            return GuardResult(False, f"v3 prepared {name} must be a tensor")
        if tensor.device != device:
            return GuardResult(False, f"v3 prepared {name} must be on {device}, got {tensor.device}")

    if c1.shape != (num_experts, 2 * intermediate_size, hidden_size // 2):
        return GuardResult(False, f"v3 prepared C1 weight shape is unsupported: {tuple(c1.shape)}")
    if c1.dtype != torch.uint8:
        return GuardResult(False, "v3 prepared C1 weight must be packed NVFP4 uint8")
    c1_scale_shape = _cutedsl_scale_shape(
        experts=num_experts,
        mn=2 * intermediate_size,
        sf_k=hidden_size // 16,
    )
    if c1_scale.shape != c1_scale_shape:
        return GuardResult(False, f"v3 prepared C1 scale shape is unsupported: {tuple(c1_scale.shape)}")
    if c1_scale.dtype != torch.float8_e4m3fn:
        return GuardResult(False, "v3 prepared C1 scale must be FP8 E4M3")

    if c2.shape != (num_experts, hidden_size, intermediate_size // 2):
        return GuardResult(False, f"v3 prepared C2 weight shape is unsupported: {tuple(c2.shape)}")
    if c2.dtype != torch.uint8:
        return GuardResult(False, "v3 prepared C2 weight must be packed NVFP4 uint8")
    c2_scale_shape = _cutedsl_scale_shape(
        experts=num_experts,
        mn=hidden_size,
        sf_k=intermediate_size // 16,
    )
    if c2_scale.shape != c2_scale_shape:
        return GuardResult(False, f"v3 prepared C2 scale shape is unsupported: {tuple(c2_scale.shape)}")
    if c2_scale.dtype != torch.float8_e4m3fn:
        return GuardResult(False, "v3 prepared C2 scale must be FP8 E4M3")
    return GuardResult(True)


def _validate_v4_cutedsl_weights(
    weights: object,
    *,
    intermediate_size: int,
    hidden_size: int,
    num_experts: int,
    device: torch.device,
) -> GuardResult:
    required = (
        "c1_weights_trt_preshuffled",
        "c1_weights_scale_trt_preshuffled",
        "c2_weights_trt_preshuffled",
        "c2_weights_scale_trt_preshuffled",
    )
    missing = [name for name in required if getattr(weights, name, None) is None]
    if missing:
        return GuardResult(
            False,
            f"v4 prepared weights missing fields: {', '.join(missing)}",
        )

    c1 = getattr(weights, "c1_weights_trt_preshuffled")
    c1_scale = getattr(weights, "c1_weights_scale_trt_preshuffled")
    c2 = getattr(weights, "c2_weights_trt_preshuffled")
    c2_scale = getattr(weights, "c2_weights_scale_trt_preshuffled")
    for name, tensor in (
        ("c1_weights_trt_preshuffled", c1),
        ("c1_weights_scale_trt_preshuffled", c1_scale),
        ("c2_weights_trt_preshuffled", c2),
        ("c2_weights_scale_trt_preshuffled", c2_scale),
    ):
        if not isinstance(tensor, torch.Tensor):
            return GuardResult(False, f"v4 prepared {name} must be a tensor")
        if tensor.device != device:
            return GuardResult(
                False,
                f"v4 prepared {name} must be on {device}, got {tensor.device}",
            )

    if c1.shape != (num_experts, 2 * intermediate_size, hidden_size // 2):
        return GuardResult(
            False,
            f"v4 prepared C1 weight shape is unsupported: {tuple(c1.shape)}",
        )
    if c1.dtype != torch.uint8 or not c1.is_contiguous():
        return GuardResult(False, "v4 prepared C1 weight must be contiguous packed NVFP4 uint8")
    if c1_scale.shape != (num_experts, 2 * intermediate_size, hidden_size // 16):
        return GuardResult(
            False,
            f"v4 prepared C1 scale shape is unsupported: {tuple(c1_scale.shape)}",
        )
    if c1_scale.dtype != torch.float8_e4m3fn or not c1_scale.is_contiguous():
        return GuardResult(False, "v4 prepared C1 scale must be contiguous FP8 E4M3")

    if c2.shape != (num_experts, hidden_size, intermediate_size // 2):
        return GuardResult(
            False,
            f"v4 prepared C2 weight shape is unsupported: {tuple(c2.shape)}",
        )
    if c2.dtype != torch.uint8 or not c2.is_contiguous():
        return GuardResult(False, "v4 prepared C2 weight must be contiguous packed NVFP4 uint8")
    if c2_scale.shape != (num_experts, hidden_size, intermediate_size // 16):
        return GuardResult(
            False,
            f"v4 prepared C2 scale shape is unsupported: {tuple(c2_scale.shape)}",
        )
    if c2_scale.dtype != torch.float8_e4m3fn or not c2_scale.is_contiguous():
        return GuardResult(False, "v4 prepared C2 scale must be contiguous FP8 E4M3")
    return GuardResult(True)


def _has_v4_trt_preshuffled_c1(weights: object) -> bool:
    return (
        getattr(weights, "c1_weights_trt_preshuffled", None) is not None
        and getattr(weights, "c1_weights_scale_trt_preshuffled", None) is not None
    )


def _has_v4_trt_preshuffled_c2(weights: object) -> bool:
    return (
        getattr(weights, "c2_weights_trt_preshuffled", None) is not None
        and getattr(weights, "c2_weights_scale_trt_preshuffled", None) is not None
    )


def _validate_v3_family_prepared_weights(
    weights: object,
    *,
    pipeline: str,
    intermediate_size: int,
    hidden_size: int,
    num_experts: int,
    device: torch.device,
) -> GuardResult:
    validate_weights = (
        _validate_v4_cutedsl_weights
        if pipeline == "v4" and _has_v4_trt_preshuffled_c1(weights)
        else _validate_v3_cutedsl_weights
    )
    return validate_weights(
        weights,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        num_experts=num_experts,
        device=device,
    )


def _weight_cache_key(
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    scale_mode: str,
    preswap_c1: bool,
    fused_c1: bool,
    c1_block_pair_order: str,
) -> tuple[object, ...]:
    return (
        str(scale_mode),
        bool(preswap_c1),
        bool(fused_c1),
        str(c1_block_pair_order),
        _tensor_cache_key(gemm1_weights),
        _tensor_cache_key(gemm1_weights_scale),
        _tensor_cache_key(gemm2_weights),
        _tensor_cache_key(gemm2_weights_scale),
    )


def _prepare_cutedsl_weights(
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    scale_to_cutedsl,
    *,
    cache_enabled: bool,
    scale_mode: str,
    preswap_c1: bool,
    fused_c1: bool,
    c1_block_pair_order: str,
) -> CachedCutedslWeights:
    from .moe_primitives import (
        unshuffle_trtllm_gemm1_weight_for_cutedsl,
        unshuffle_trtllm_gemm2_weight_for_cutedsl,
    )

    cache_key = _weight_cache_key(
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        scale_mode,
        preswap_c1,
        fused_c1,
        c1_block_pair_order,
    )
    if cache_enabled:
        cached = _WEIGHT_CACHE.get(cache_key)
        if cached is not None:
            return cached

    c1_weights, c1_weights_scale_linear = unshuffle_trtllm_gemm1_weight_for_cutedsl(
        gemm1_weights.view(torch.uint8),
        gemm1_weights_scale,
    )
    c2_weights, c2_weights_scale_linear = unshuffle_trtllm_gemm2_weight_for_cutedsl(
        gemm2_weights.view(torch.uint8),
        gemm2_weights_scale,
    )
    if preswap_c1:
        half = c1_weights.shape[1] // 2
        c1_weights = torch.cat((c1_weights[:, half:], c1_weights[:, :half]), dim=1).contiguous()
        c1_weights_scale_linear = torch.cat(
            (c1_weights_scale_linear[:, half:], c1_weights_scale_linear[:, :half]),
            dim=1,
        ).contiguous()
    c1_weights_gate_up_block_pair = None
    c1_weights_gate_up_block_pair_scale = None
    if fused_c1:
        c1_weights_gate_up_block_pair = _interleave_gate_up_blocks(
            c1_weights,
            block=64,
            preswapped_c1=preswap_c1,
            output_order=c1_block_pair_order,
        )
        c1_weights_gate_up_block_pair_scale_linear = _interleave_gate_up_blocks(
            c1_weights_scale_linear,
            block=64,
            preswapped_c1=preswap_c1,
            output_order=c1_block_pair_order,
        )
        c1_weights_gate_up_block_pair_scale = scale_to_cutedsl(
            c1_weights_gate_up_block_pair_scale_linear
        )

    prepared = CachedCutedslWeights(
        c1_weights=c1_weights,
        c1_weights_scale_linear=c1_weights_scale_linear,
        c1_weights_scale=scale_to_cutedsl(c1_weights_scale_linear),
        c2_weights=c2_weights,
        c2_weights_scale=scale_to_cutedsl(c2_weights_scale_linear),
        c1_weights_gate_up_block_pair=c1_weights_gate_up_block_pair,
        c1_weights_gate_up_block_pair_scale=c1_weights_gate_up_block_pair_scale,
    )
    if cache_enabled:
        _WEIGHT_CACHE[cache_key] = prepared
    return prepared


def _prepare_v3_cutedsl_weights(
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    scale_to_cutedsl,
    *,
    cache_enabled: bool,
    scale_mode: str,
    preswap_c1: bool,
) -> V3CutedslWeights:
    from .moe_primitives import (
        unshuffle_trtllm_gemm1_weight_for_cutedsl,
        unshuffle_trtllm_gemm2_weight_for_cutedsl,
    )

    cache_key = (
        "v3",
        _weight_cache_key(
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            scale_mode,
            preswap_c1,
            True,
            "up_gate",
        ),
    )
    if cache_enabled:
        cached = _V3_WEIGHT_CACHE.get(cache_key)
        if cached is not None:
            return cached

    c1_weights, c1_weights_scale_linear = unshuffle_trtllm_gemm1_weight_for_cutedsl(
        gemm1_weights.view(torch.uint8),
        gemm1_weights_scale,
    )
    if preswap_c1:
        half = c1_weights.shape[1] // 2
        c1_weights = torch.cat((c1_weights[:, half:], c1_weights[:, :half]), dim=1).contiguous()
        c1_weights_scale_linear = torch.cat(
            (c1_weights_scale_linear[:, half:], c1_weights_scale_linear[:, :half]),
            dim=1,
        ).contiguous()

    c1_weights_gate_up_block_pair = _interleave_gate_up_blocks(
        c1_weights,
        block=64,
        preswapped_c1=preswap_c1,
        output_order="up_gate",
    )
    del c1_weights
    c1_weights_gate_up_block_pair_scale_linear = _interleave_gate_up_blocks(
        c1_weights_scale_linear,
        block=64,
        preswapped_c1=preswap_c1,
        output_order="up_gate",
    )
    del c1_weights_scale_linear
    c1_weights_gate_up_block_pair_scale = scale_to_cutedsl(
        c1_weights_gate_up_block_pair_scale_linear
    )
    del c1_weights_gate_up_block_pair_scale_linear

    c2_weights, c2_weights_scale_linear = unshuffle_trtllm_gemm2_weight_for_cutedsl(
        gemm2_weights.view(torch.uint8),
        gemm2_weights_scale,
    )
    c2_weights_scale = scale_to_cutedsl(c2_weights_scale_linear)
    del c2_weights_scale_linear

    prepared = V3CutedslWeights(
        c1_weights_gate_up_block_pair=c1_weights_gate_up_block_pair,
        c1_weights_gate_up_block_pair_scale=c1_weights_gate_up_block_pair_scale,
        c2_weights=c2_weights,
        c2_weights_scale=c2_weights_scale,
    )
    if cache_enabled:
        _V3_WEIGHT_CACHE[cache_key] = prepared
    return prepared


def _prepare_v4_cutedsl_weights(
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    *,
    cache_enabled: bool,
) -> V4CutedslWeights:
    cache_key = (
        "v4-trt-native-c1-c2",
        _tensor_cache_key(gemm1_weights),
        _tensor_cache_key(gemm1_weights_scale),
        _tensor_cache_key(gemm2_weights),
        _tensor_cache_key(gemm2_weights_scale),
    )
    if cache_enabled:
        cached = _V4_WEIGHT_CACHE.get(cache_key)
        if cached is not None:
            return cached

    prepared = V4CutedslWeights(
        c1_weights_trt_preshuffled=gemm1_weights.view(torch.uint8),
        c1_weights_scale_trt_preshuffled=gemm1_weights_scale,
        c2_weights_trt_preshuffled=gemm2_weights.view(torch.uint8),
        c2_weights_scale_trt_preshuffled=gemm2_weights_scale,
    )
    if cache_enabled:
        _V4_WEIGHT_CACHE[cache_key] = prepared
    return prepared


def prepare_fused_moe_nvfp4_sm103_v3_weights(
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    *,
    cache_enabled: bool = False,
) -> V3CutedslWeights:
    """Prepack public TRTLLM NVFP4 MoE weights for the SM103 v3 pipeline."""

    _sync_legacy_env()
    from .moe_primitives import linear_scale_to_cutedsl_scale

    if _env_flag("CUDA_SCALE"):
        from .moe_cuda_kernels import linear_scale_to_cutedsl_scale_cuda

        scale_to_cutedsl = linear_scale_to_cutedsl_scale_cuda
        scale_mode = "cuda"
    else:
        scale_to_cutedsl = linear_scale_to_cutedsl_scale
        scale_mode = "torch"
    return _prepare_v3_cutedsl_weights(
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        scale_to_cutedsl,
        cache_enabled=cache_enabled,
        scale_mode=scale_mode,
        preswap_c1=_env_flag("PRESWAP_C1"),
    )


def prepare_fused_moe_nvfp4_sm103_v4_weights(
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    *,
    cache_enabled: bool = False,
) -> V4CutedslWeights:
    """Prepare v4 weights while preserving TRT's native GEMM1/GEMM2 preshuffle."""

    _sync_legacy_env()
    return _prepare_v4_cutedsl_weights(
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        cache_enabled=cache_enabled,
    )


def _run_cutedsl_path(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    output1_scale_scalar: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    top_k: int,
    intermediate_size: int,
    routed_scaling_factor: Optional[float],
    output: Optional[torch.Tensor],
    routing_replay_out: Optional[torch.Tensor],
    *,
    pipeline: str,
    v3_prepared_weights: Optional[object] = None,
) -> List[torch.Tensor]:
    _sync_legacy_env()
    from .moe_primitives import (
        compact_nvfp4_routes_by_expert,
        compute_topk_renormalized,
        expert_major_payload_to_cutedsl_operand,
        finalize_weighted_routes_by_expert,
        grouped_fp4_gemm_nt_masked_cutedsl,
        linear_scale_to_cutedsl_scale,
        silu_and_mul_nvfp4_quantize_cutedsl,
    )

    is_v3 = pipeline in V3_FAMILY_PIPELINES
    runtime_num_experts = int(routing_logits.shape[1])
    runtime_hidden_size = int(hidden_states.shape[1]) * 2
    fused_c1 = _env_flag("FUSED_C1")
    preswapped_c1 = _env_flag("PRESWAP_C1")
    if is_v3 and not fused_c1:
        raise ValueError("pipeline='v3' requires fused_gemm1_contiguous")
    if is_v3:
        cutlass_tactic = None
        cutlass_route_tile_m = 128
        if pipeline == "v4":
            from .cutlass_gemm2_finalize import (
                cutlass_gemm2_finalize_required_route_tile_m,
                select_cutlass_gemm2_finalize_tactic,
            )

            cutlass_tactic = select_cutlass_gemm2_finalize_tactic(
                tokens=int(routing_logits.shape[0]),
                n=runtime_hidden_size,
                k=intermediate_size,
                top_k=top_k,
            )
            cutlass_route_tile_m = cutlass_gemm2_finalize_required_route_tile_m(
                cutlass_tactic
            )
            if cutlass_route_tile_m > 256:
                raise ValueError(
                    "pipeline='v4' CUTLASS GEMM2-finalize tactic requires route metadata "
                    f"tile M={cutlass_route_tile_m}, but current v4 route producer "
                    "supports at most C1 tile M=256"
                )
        c1_mma_tiler_mn, c1_cluster_shape_mn = _gemm_tuning(
            "C1",
            default_mma_tiler_mn=_V3_C1_MMA_TILER_MN,
            default_cluster_shape_mn=_V3_C1_CLUSTER_SHAPE_MN,
        )
        c2_mma_tiler_mn, c2_cluster_shape_mn = _gemm_tuning(
            "C2",
            default_mma_tiler_mn=_V3_C2_MMA_TILER_MN,
            default_cluster_shape_mn=_V3_C2_CLUSTER_SHAPE_MN,
        )
        c1_m128_max_m = int(
            _env_get("V3_C1_M128_MAX_M", str(_V3_C1_M128_MAX_M)) or "0"
        )
        if (
            not _has_c1_tuning_override()
            and c1_m128_max_m > 0
            and int(routing_logits.shape[0]) <= c1_m128_max_m
            and cutlass_route_tile_m <= 128
        ):
            c1_mma_tiler_mn = _V3_C1_FALLBACK_MMA_TILER_MN
            c1_cluster_shape_mn = _V3_C1_FALLBACK_CLUSTER_SHAPE_MN
        if cutlass_route_tile_m == 256 and int(c1_mma_tiler_mn[0]) != 256:
            raise ValueError(
                "pipeline='v4' CUTLASS GEMM2-finalize tactic requires C1 tile M=256 route metadata, "
                f"got C1 tile M={c1_mma_tiler_mn[0]}"
            )
    else:
        c1_mma_tiler_mn, c1_cluster_shape_mn = _gemm_tuning("C1")
        c2_mma_tiler_mn, c2_cluster_shape_mn = _gemm_tuning("C2")

    routes = None
    using_v3_fused_route = (
        is_v3
        and _env_flag("V3_FUSED_ROUTE")
        and routing_bias is None
        and routing_logits.dtype == torch.bfloat16
    )
    if using_v3_fused_route:
        from .contiguous_routes import make_fused_topk_contiguous_routes

        topk, routes = make_fused_topk_contiguous_routes(
            routing_logits,
            top_k=top_k,
            num_experts=runtime_num_experts,
            tile_m=int(c1_mma_tiler_mn[0]),
            routed_scaling_factor=routed_scaling_factor,
            debug_sync_checks=_env_flag("CONTIGUOUS_DEBUG_SYNC_CHECKS"),
        )
    else:
        topk = compute_topk_renormalized(
            routing_logits,
            top_k,
            routing_bias=routing_bias,
            routed_scaling_factor=routed_scaling_factor,
        )
    if routing_replay_out is not None:
        routing_replay_out.copy_(topk.ids.to(torch.int16))

    if routes is None:
        if is_v3:
            from .contiguous_routes import make_moe_sort_contiguous_routes

            routes = make_moe_sort_contiguous_routes(
                topk.ids,
                topk.weights,
                num_experts=runtime_num_experts,
                top_k=top_k,
                tile_m=int(c2_mma_tiler_mn[0]),
                debug_sync_checks=_env_flag("CONTIGUOUS_DEBUG_SYNC_CHECKS"),
            )
        elif _env_flag("CUDA_COMPACT"):
            from .moe_cuda_kernels import compact_nvfp4_routes_by_expert_cuda

            fixed_max_m_env = _env_get("CUDA_COMPACT_MAX_M")
            routes = compact_nvfp4_routes_by_expert_cuda(
                hidden_states,
                hidden_states_scale,
                topk.ids,
                num_experts=SUPPORTED_EXPERTS,
                max_m=None if fixed_max_m_env is None else int(fixed_max_m_env),
                physical_scale=_env_flag("CUDA_COMPACT_PHYSICAL_SCALE"),
            )
        else:
            routes = compact_nvfp4_routes_by_expert(
                hidden_states,
                hidden_states_scale,
                topk.ids,
                range(SUPPORTED_EXPERTS),
            )

    if _env_flag("CUDA_SCALE"):
        from .moe_cuda_kernels import linear_scale_to_cutedsl_scale_cuda

        scale_to_cutedsl = linear_scale_to_cutedsl_scale_cuda
        scale_mode = "cuda"
    else:
        scale_to_cutedsl = linear_scale_to_cutedsl_scale
        scale_mode = "torch"

    if is_v3:
        if v3_prepared_weights is None:
            if pipeline == "v4":
                weights = _prepare_v4_cutedsl_weights(
                    gemm1_weights,
                    gemm1_weights_scale,
                    gemm2_weights,
                    gemm2_weights_scale,
                    cache_enabled=_env_flag("CACHE_WEIGHTS"),
                )
            else:
                weights = _prepare_v3_cutedsl_weights(
                    gemm1_weights,
                    gemm1_weights_scale,
                    gemm2_weights,
                    gemm2_weights_scale,
                    scale_to_cutedsl,
                    cache_enabled=_env_flag("CACHE_WEIGHTS"),
                    scale_mode=scale_mode,
                    preswap_c1=preswapped_c1,
                )
        else:
            weight_guard = _validate_v3_family_prepared_weights(
                v3_prepared_weights,
                pipeline=pipeline,
                intermediate_size=intermediate_size,
                hidden_size=runtime_hidden_size,
                num_experts=runtime_num_experts,
                device=hidden_states.device,
            )
            if not weight_guard.ok:
                raise ValueError(
                    f"unsupported {pipeline} prepared weights: {weight_guard.reason}"
                )
            weights = v3_prepared_weights
        if (
            routes.token_id_mapping is None
            or routes.permuted_idx_to_expanded_idx is None
            or routes.tile_idx_to_expert_idx is None
            or routes.tile_idx_to_mn_limit is None
            or routes.num_non_exiting_tiles is None
        ):
            raise ValueError("pipeline='v3' contiguous route metadata is incomplete")
        if routes.c1_tile_idx_to_expert_idx is not None:
            if (
                routes.c1_tile_idx_to_mn_limit is None
                or routes.c1_num_non_exiting_tiles is None
            ):
                raise ValueError("pipeline='v3' C1 route metadata is incomplete")
            c1_tile_idx_to_expert_idx = routes.c1_tile_idx_to_expert_idx
            c1_tile_idx_to_mn_limit = routes.c1_tile_idx_to_mn_limit
            c1_num_non_exiting_tiles = routes.c1_num_non_exiting_tiles
            if int(c2_mma_tiler_mn[0]) == 128:
                c2_tile_idx_to_expert_idx = routes.tile_idx_to_expert_idx
                c2_tile_idx_to_mn_limit = routes.tile_idx_to_mn_limit
                c2_num_non_exiting_tiles = routes.num_non_exiting_tiles
            elif int(c2_mma_tiler_mn[0]) == int(c1_mma_tiler_mn[0]):
                c2_tile_idx_to_expert_idx = c1_tile_idx_to_expert_idx
                c2_tile_idx_to_mn_limit = c1_tile_idx_to_mn_limit
                c2_num_non_exiting_tiles = c1_num_non_exiting_tiles
            else:
                raise ValueError(
                    "pipeline='v3' cannot map C2 tile M "
                    f"{c2_mma_tiler_mn[0]} onto C1 tile M {c1_mma_tiler_mn[0]}"
                )
        else:
            c1_tile_idx_to_expert_idx = routes.tile_idx_to_expert_idx
            c1_tile_idx_to_mn_limit = routes.tile_idx_to_mn_limit
            c1_num_non_exiting_tiles = routes.num_non_exiting_tiles
            c2_tile_idx_to_expert_idx = routes.tile_idx_to_expert_idx
            c2_tile_idx_to_mn_limit = routes.tile_idx_to_mn_limit
            c2_num_non_exiting_tiles = routes.num_non_exiting_tiles
            if int(c1_mma_tiler_mn[0]) != int(c2_mma_tiler_mn[0]):
                if using_v3_fused_route or int(c2_mma_tiler_mn[0]) != 128:
                    raise ValueError(
                        "pipeline='v3' route metadata cannot represent mismatched "
                        f"C1/C2 tile M ({c1_mma_tiler_mn[0]} vs {c2_mma_tiler_mn[0]})"
                    )
                c1_mma_tiler_mn = _V3_C1_FALLBACK_MMA_TILER_MN
                c1_cluster_shape_mn = _V3_C1_FALLBACK_CLUSTER_SHAPE_MN

        c1_token_id_mapping = routes.token_id_mapping
        cutlass_permuted_idx_to_expanded_idx = routes.permuted_idx_to_expanded_idx

        output_dtype = torch.bfloat16 if output is None else output.dtype
        if output_dtype != torch.bfloat16:
            raise ValueError("pipeline='v3'/'v4' only supports bfloat16 output")
        from .blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion import (
            blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_nvfp4,
        )

        output1_alpha_is_one = _is_unit_scale_tensor(output1_scale_gate_scalar)
        output1_global_scale_is_one = _is_unit_scale_tensor(output1_scale_scalar)
        c1_trt_preshuffled = (
            pipeline == "v4" and _has_v4_trt_preshuffled_c1(weights)
        )
        if c1_trt_preshuffled:
            c1_weights = weights.c1_weights_trt_preshuffled
            c1_weights_scale = weights.c1_weights_scale_trt_preshuffled
        else:
            c1_weights = weights.c1_weights_gate_up_block_pair
            c1_weights_scale = weights.c1_weights_gate_up_block_pair_scale
        post_payload, post_scale = blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_nvfp4(
            hidden_states,
            c1_weights,
            hidden_states_scale,
            c1_weights_scale,
            output1_scale_gate_scalar,
            c1_tile_idx_to_expert_idx,
            c1_tile_idx_to_mn_limit,
            c1_token_id_mapping,
            c1_num_non_exiting_tiles,
            out=None,
            out_scale=None,
            global_scale=output1_scale_scalar,
            topk=top_k,
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype="float4_e2m1fn",
            sf_vec_size=16,
            mma_tiler_mn=c1_mma_tiler_mn,
            cluster_shape_mn=c1_cluster_shape_mn,
            vectorized_f32=True,
            raster_along_m=False,
            # Folding router scales before the C1 FP4 requantization widens
            # max_abs against the FlashInfer reference; keep scaling in C2.
            token_final_scales=None,
            fold_final_scales=False,
            alpha_is_one=output1_alpha_is_one,
            global_scale_is_one=output1_global_scale_is_one,
            trt_preshuffled_b=c1_trt_preshuffled,
            sm_count=None,
            enable_pdl=False,
        )
        if (
            post_payload.ndim != 2
            or post_payload.shape[1] != intermediate_size // 2
        ):
            raise ValueError(
                "pipeline='v3' C1 output must be flat C2 A payload "
                f"[permuted_m,{intermediate_size // 2}], got {tuple(post_payload.shape)}"
            )
        if post_payload.dtype != torch.uint8:
            raise ValueError(
                f"pipeline='v3' C1 packed payload must be uint8, got {post_payload.dtype}"
            )

        result = output
        if pipeline == "v4":
            from .cutlass_gemm2_finalize import (
                cutlass_gemm2_finalize_nvfp4,
            )

            c2_trt_preshuffled = _has_v4_trt_preshuffled_c2(weights)
            if c2_trt_preshuffled:
                c2_weights = weights.c2_weights_trt_preshuffled
                c2_weights_scale = weights.c2_weights_scale_trt_preshuffled
            else:
                c2_weights = weights.c2_weights
                c2_weights_scale = weights.c2_weights_scale

            cutlass_flat_m = int(post_payload.shape[0])
            if cutlass_flat_m % 128 != 0:
                raise ValueError(
                    f"pipeline='v4' CUTLASS GEMM2-finalize M must be a multiple of 128, got {cutlass_flat_m}"
                )
            if cutlass_route_tile_m not in (128, 256):
                raise ValueError(
                    "pipeline='v4' currently supports only CUTLASS route tile M=128/256, "
                    f"got {cutlass_route_tile_m}"
                )
            cutlass_tile_idx_to_expert_idx = routes.cutlass_tile_idx_to_expert_idx
            cutlass_tile_idx_to_mn_limit = routes.cutlass_tile_idx_to_mn_limit
            cutlass_num_non_exiting_tiles = routes.cutlass_num_non_exiting_tiles
            if cutlass_route_tile_m == 128 and cutlass_tile_idx_to_expert_idx is None:
                cutlass_tile_idx_to_expert_idx = routes.tile_idx_to_expert_idx
                cutlass_tile_idx_to_mn_limit = routes.tile_idx_to_mn_limit
                cutlass_num_non_exiting_tiles = routes.num_non_exiting_tiles
            if (
                cutlass_tile_idx_to_expert_idx is None
                or cutlass_tile_idx_to_mn_limit is None
                or cutlass_num_non_exiting_tiles is None
            ):
                raise ValueError("pipeline='v4' CUTLASS route metadata is incomplete")
            result = cutlass_gemm2_finalize_nvfp4(
                a=post_payload,
                b=c2_weights,
                a_scale=post_scale,
                b_scale=c2_weights_scale,
                alpha=output2_scale_scalar,
                tile_idx_to_expert_idx=cutlass_tile_idx_to_expert_idx,
                tile_idx_to_mn_limit=cutlass_tile_idx_to_mn_limit,
                cutlass_num_non_exiting_tiles=cutlass_num_non_exiting_tiles,
                permuted_idx_to_expanded_idx=cutlass_permuted_idx_to_expanded_idx,
                token_final_scales=topk.weights,
                top_k=top_k,
                out=result,
                tactic=cutlass_tactic,
                alpha_is_one=_is_unit_scale_tensor(output2_scale_scalar),
                trt_preshuffled_b=c2_trt_preshuffled,
            )
        else:
            from .grouped_gemm_finalize_fusion_nvfp4 import (
                blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4,
            )

            output2_alpha_is_one = _is_unit_scale_tensor(output2_scale_scalar)
            if result is not None:
                result.zero_()
            result = blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4(
                a=post_payload,
                b=weights.c2_weights,
                a_scale=post_scale,
                b_scale=weights.c2_weights_scale,
                alpha=output2_scale_scalar,
                tile_idx_to_expert_idx=c2_tile_idx_to_expert_idx,
                num_non_exiting_tiles=c2_num_non_exiting_tiles,
                tile_idx_to_mn_limit=c2_tile_idx_to_mn_limit,
                permuted_idx_to_expanded_idx=routes.permuted_idx_to_expanded_idx,
                token_final_scales=topk.weights,
                out=result,
                ab_dtype="float4_e2m1fn",
                sf_dtype="float8_e4m3fn",
                out_dtype="bfloat16",
                sf_vec_size=16,
                mma_tiler_mn=c2_mma_tiler_mn,
                cluster_shape_mn=c2_cluster_shape_mn,
                raster_along_m=False,
                sm_count=None,
                enable_pdl=False,
                alpha_is_one=output2_alpha_is_one,
                final_scale_is_folded=False,
            )
        return [result]

    weights = _prepare_cutedsl_weights(
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        scale_to_cutedsl,
        cache_enabled=_env_flag("CACHE_WEIGHTS"),
        scale_mode=scale_mode,
        preswap_c1=preswapped_c1,
        fused_c1=fused_c1,
        c1_block_pair_order="gate_up",
    )

    alpha = torch.ones((1, 1, SUPPORTED_EXPERTS), device=hidden_states.device, dtype=torch.bfloat16)
    route_scale_cutedsl = routes.scale_cutedsl
    if route_scale_cutedsl is None:
        route_scale_cutedsl = scale_to_cutedsl(routes.scale)

    if fused_c1:
        from .grouped_gemm_masked_swiglu_nvfp4 import (
            grouped_fp4_gemm_nt_masked_swiglu_nvfp4_cutedsl,
        )

        c1_weights_block_pair = weights.c1_weights_gate_up_block_pair
        c1_weights_scale_block_pair = weights.c1_weights_gate_up_block_pair_scale
        if c1_weights_block_pair is None or c1_weights_scale_block_pair is None:
            c1_weights_block_pair = _interleave_gate_up_blocks(
                weights.c1_weights,
                block=64,
                preswapped_c1=preswapped_c1,
            )
            c1_weights_scale_block_pair_linear = _interleave_gate_up_blocks(
                weights.c1_weights_scale_linear,
                block=64,
                preswapped_c1=preswapped_c1,
            )
            c1_weights_scale_block_pair = scale_to_cutedsl(c1_weights_scale_block_pair_linear)
        post_payload, post_scale = grouped_fp4_gemm_nt_masked_swiglu_nvfp4_cutedsl(
            expert_major_payload_to_cutedsl_operand(routes.payload),
            expert_major_payload_to_cutedsl_operand(c1_weights_block_pair),
            route_scale_cutedsl,
            c1_weights_scale_block_pair,
            routes.masked_m,
            n=c1_weights_block_pair.shape[1],
            k=SUPPORTED_HIDDEN,
            global_scale=output1_scale_scalar,
            mma_tiler_mn=c1_mma_tiler_mn,
            cluster_shape_mn=c1_cluster_shape_mn,
        )
    else:
        c1 = grouped_fp4_gemm_nt_masked_cutedsl(
            expert_major_payload_to_cutedsl_operand(routes.payload),
            expert_major_payload_to_cutedsl_operand(weights.c1_weights),
            route_scale_cutedsl,
            weights.c1_weights_scale,
            routes.masked_m,
            n=weights.c1_weights.shape[1],
            k=SUPPORTED_HIDDEN,
            alpha=alpha,
            alpha_dtype="bfloat16",
            mma_tiler_mn=c1_mma_tiler_mn,
            cluster_shape_mn=c1_cluster_shape_mn,
        )

        if _env_flag("PRESWAP_C1"):
            c1_epilogue_input = c1
        else:
            half = c1.shape[-1] // 2
            c1_epilogue_input = torch.cat((c1[..., half:], c1[..., :half]), dim=-1).contiguous()
        post_payload, post_scale = silu_and_mul_nvfp4_quantize_cutedsl(
            c1_epilogue_input,
            routes.masked_m,
            output1_scale_scalar,
        )

    output_dtype = torch.bfloat16 if output is None else output.dtype
    if pipeline == "v2":
        if output_dtype != torch.bfloat16:
            raise ValueError("pipeline='v2' only supports bfloat16 output")
        from .grouped_gemm_finalize_fusion_nvfp4 import (
            blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4,
        )
        from .moe_cuda_kernels import repack_c2_finalize_inputs_cuda

        (
            c2_a,
            c2_a_scale,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            num_non_exiting_tiles,
            permuted_idx_to_expanded_idx,
        ) = repack_c2_finalize_inputs_cuda(
            post_payload,
            post_scale,
            routes.masked_m,
            routes.route_tokens,
            topk.ids,
            tile_m=_V2_C2_MMA_TILER_MN[0],
        )
        result = output
        if result is None:
            result = torch.zeros(
                (hidden_states.shape[0], SUPPORTED_HIDDEN),
                device=hidden_states.device,
                dtype=output_dtype,
            )
        else:
            result.zero_()
        alpha_c2 = torch.ones((SUPPORTED_EXPERTS,), device=hidden_states.device, dtype=torch.float32)
        blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4(
            a=c2_a,
            b=weights.c2_weights,
            a_scale=c2_a_scale,
            b_scale=weights.c2_weights_scale,
            alpha=alpha_c2,
            tile_idx_to_expert_idx=tile_idx_to_expert_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            token_final_scales=topk.weights,
            out=result,
            mma_tiler_mn=_V2_C2_MMA_TILER_MN,
            cluster_shape_mn=_V2_C2_CLUSTER_SHAPE_MN,
            sm_count=_V2_C2_SM_COUNT,
            enable_pdl=True,
            alpha_is_one=True,
        )
        return [result]

    c2 = grouped_fp4_gemm_nt_masked_cutedsl(
        post_payload,
        expert_major_payload_to_cutedsl_operand(weights.c2_weights),
        post_scale,
        weights.c2_weights_scale,
        routes.masked_m,
        n=SUPPORTED_HIDDEN,
        k=intermediate_size,
        alpha=alpha,
        alpha_dtype="bfloat16",
        mma_tiler_mn=c2_mma_tiler_mn,
        cluster_shape_mn=c2_cluster_shape_mn,
    )
    if _env_flag("CUDA_FINALIZE"):
        from .moe_cuda_kernels import finalize_weighted_routes_by_expert_cuda

        result = finalize_weighted_routes_by_expert_cuda(
            c2,
            routes.route_tokens,
            routes.expert_ids,
            topk.ids,
            topk.weights,
            (hidden_states.shape[0], SUPPORTED_HIDDEN),
            output_dtype,
            route_rows=routes.route_rows,
        )
    else:
        result = finalize_weighted_routes_by_expert(
            c2,
            routes.route_tokens,
            routes.expert_ids,
            topk.ids,
            topk.weights,
            (hidden_states.shape[0], SUPPORTED_HIDDEN),
            output_dtype,
        )
    if output is not None:
        output.copy_(result)
        return [output]
    return [result]


def fused_moe_nvfp4_sm103(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_bias: Optional[torch.Tensor],
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm2_bias: Optional[torch.Tensor],
    output1_scale_scalar: Optional[torch.Tensor],
    output1_scale_gate_scalar: Optional[torch.Tensor],
    output2_scale_scalar: Optional[torch.Tensor],
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    activation_type: int = SUPPORTED_ACTIVATION,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
    norm_topk_prob: bool = True,
    routing_replay_out: Optional[torch.Tensor] = None,
    *,
    enable_cutedsl: Optional[bool] = None,
    fallback: bool = False,
    pipeline: str = DEFAULT_PIPELINE,
    v3_prepared_weights: Optional[object] = None,
) -> List[torch.Tensor]:
    """SM103 CuTeDSL NVFP4 MoE entry point.

    The positional signature mirrors `flashinfer.trtllm_fp4_block_scale_moe`.
    On SM103 the custom CuTeDSL path is enabled by default. On other
    architectures, the default behavior is fail-closed; pass `fallback=True`
    to call the FlashInfer baseline instead.
    """

    _sync_legacy_env()
    pipeline = _check_pipeline(pipeline)
    if v3_prepared_weights is not None and pipeline not in V3_FAMILY_PIPELINES:
        raise ValueError("v3_prepared_weights requires pipeline='v3' or 'v4'")
    if not _custom_enabled(enable_cutedsl):
        if v3_prepared_weights is not None:
            raise ValueError("v3_prepared_weights requires the SM103 custom path")
        return _flashinfer_call(
            routing_logits,
            routing_bias,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm1_bias,
            gemm1_alpha,
            gemm1_beta,
            gemm1_clamp_limit,
            gemm2_weights,
            gemm2_weights_scale,
            gemm2_bias,
            output1_scale_scalar,
            output1_scale_gate_scalar,
            output2_scale_scalar,
            num_experts,
            top_k,
            n_group,
            topk_group,
            intermediate_size,
            local_expert_offset,
            local_num_experts,
            routed_scaling_factor,
            routing_method_type,
            do_finalize,
            enable_pdl,
            activation_type,
            output,
            tune_max_num_tokens,
            norm_topk_prob,
            routing_replay_out,
        )

    guard = guard_supported_cutedsl_moe_call(
        routing_logits,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        num_experts,
        top_k,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routing_method_type,
        do_finalize,
        activation_type,
        routing_bias=routing_bias,
        output1_scale_scalar=output1_scale_scalar,
        output1_scale_gate_scalar=output1_scale_gate_scalar,
        output2_scale_scalar=output2_scale_scalar,
        n_group=n_group,
        topk_group=topk_group,
        enable_pdl=enable_pdl,
        output=output,
        norm_topk_prob=norm_topk_prob,
        routing_replay_out=routing_replay_out,
        pipeline=pipeline,
        v3_prepared_weights=v3_prepared_weights,
    )
    if not guard.ok:
        if not fallback or v3_prepared_weights is not None:
            raise ValueError(f"unsupported SM103 CuTeDSL MoE call: {guard.reason}")
        return _flashinfer_call(
            routing_logits,
            routing_bias,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm1_bias,
            gemm1_alpha,
            gemm1_beta,
            gemm1_clamp_limit,
            gemm2_weights,
            gemm2_weights_scale,
            gemm2_bias,
            output1_scale_scalar,
            output1_scale_gate_scalar,
            output2_scale_scalar,
            num_experts,
            top_k,
            n_group,
            topk_group,
            intermediate_size,
            local_expert_offset,
            local_num_experts,
            routed_scaling_factor,
            routing_method_type,
            do_finalize,
            enable_pdl,
            activation_type,
            output,
            tune_max_num_tokens,
            norm_topk_prob,
            routing_replay_out,
        )

    return _run_cutedsl_path(
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        top_k,
        intermediate_size,
        routed_scaling_factor,
        output,
        routing_replay_out,
        pipeline=pipeline,
        v3_prepared_weights=v3_prepared_weights,
    )


def fused_moe_nvfp4_sm103_v3_prepacked(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    v3_prepared_weights: object,
    output1_scale_scalar: Optional[torch.Tensor],
    output1_scale_gate_scalar: Optional[torch.Tensor],
    output2_scale_scalar: Optional[torch.Tensor],
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    activation_type: int = SUPPORTED_ACTIVATION,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
    norm_topk_prob: bool = True,
    routing_replay_out: Optional[torch.Tensor] = None,
    *,
    enable_cutedsl: Optional[bool] = None,
    fallback: bool = False,
) -> List[torch.Tensor]:
    """Run SM103 v3 with weights returned by prepare_fused_moe_nvfp4_sm103_v3_weights."""

    return fused_moe_nvfp4_sm103(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=getattr(v3_prepared_weights, "c1_weights_gate_up_block_pair"),
        gemm1_weights_scale=getattr(v3_prepared_weights, "c1_weights_gate_up_block_pair_scale"),
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=getattr(v3_prepared_weights, "c2_weights"),
        gemm2_weights_scale=getattr(v3_prepared_weights, "c2_weights_scale"),
        gemm2_bias=None,
        output1_scale_scalar=output1_scale_scalar,
        output1_scale_gate_scalar=output1_scale_gate_scalar,
        output2_scale_scalar=output2_scale_scalar,
        num_experts=num_experts,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        intermediate_size=intermediate_size,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        routed_scaling_factor=routed_scaling_factor,
        routing_method_type=routing_method_type,
        do_finalize=do_finalize,
        enable_pdl=enable_pdl,
        activation_type=activation_type,
        output=output,
        tune_max_num_tokens=tune_max_num_tokens,
        norm_topk_prob=norm_topk_prob,
        routing_replay_out=routing_replay_out,
        enable_cutedsl=enable_cutedsl,
        fallback=fallback,
        pipeline="v3",
        v3_prepared_weights=v3_prepared_weights,
    )


def fused_moe_nvfp4_sm103_v4_prepacked(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    v3_prepared_weights: object,
    output1_scale_scalar: Optional[torch.Tensor],
    output1_scale_gate_scalar: Optional[torch.Tensor],
    output2_scale_scalar: Optional[torch.Tensor],
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    activation_type: int = SUPPORTED_ACTIVATION,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
    norm_topk_prob: bool = True,
    routing_replay_out: Optional[torch.Tensor] = None,
    *,
    enable_cutedsl: Optional[bool] = None,
    fallback: bool = False,
) -> List[torch.Tensor]:
    """Run v4 with TRT-native or legacy v3-prepacked C1/C2 weights."""

    if _has_v4_trt_preshuffled_c1(v3_prepared_weights):
        gemm1_weights = getattr(v3_prepared_weights, "c1_weights_trt_preshuffled")
        gemm1_weights_scale = getattr(
            v3_prepared_weights,
            "c1_weights_scale_trt_preshuffled",
        )
    else:
        gemm1_weights = getattr(v3_prepared_weights, "c1_weights_gate_up_block_pair")
        gemm1_weights_scale = getattr(
            v3_prepared_weights,
            "c1_weights_gate_up_block_pair_scale",
        )

    if _has_v4_trt_preshuffled_c2(v3_prepared_weights):
        gemm2_weights = getattr(v3_prepared_weights, "c2_weights_trt_preshuffled")
        gemm2_weights_scale = getattr(
            v3_prepared_weights,
            "c2_weights_scale_trt_preshuffled",
        )
    else:
        gemm2_weights = getattr(v3_prepared_weights, "c2_weights")
        gemm2_weights_scale = getattr(v3_prepared_weights, "c2_weights_scale")

    return fused_moe_nvfp4_sm103(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        gemm2_bias=None,
        output1_scale_scalar=output1_scale_scalar,
        output1_scale_gate_scalar=output1_scale_gate_scalar,
        output2_scale_scalar=output2_scale_scalar,
        num_experts=num_experts,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        intermediate_size=intermediate_size,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        routed_scaling_factor=routed_scaling_factor,
        routing_method_type=routing_method_type,
        do_finalize=do_finalize,
        enable_pdl=enable_pdl,
        activation_type=activation_type,
        output=output,
        tune_max_num_tokens=tune_max_num_tokens,
        norm_topk_prob=norm_topk_prob,
        routing_replay_out=routing_replay_out,
        enable_cutedsl=enable_cutedsl,
        fallback=fallback,
        pipeline="v4",
        v3_prepared_weights=v3_prepared_weights,
    )


trtllm_fp4_block_scale_moe_cutedsl = fused_moe_nvfp4_sm103
