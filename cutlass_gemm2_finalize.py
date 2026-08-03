#!/usr/bin/env python3
"""CUTLASS GEMM2/fused-finalize backend for SM103 v4."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.3")

import torch


THIS_DIR = Path(__file__).resolve().parent
CUTLASS_DIR = (
    THIS_DIR
    if (THIS_DIR / "src" / "fp4_gemm_cutlass_sm103_extract.cu").exists()
    else THIS_DIR / "cutlass"
)
SRC = CUTLASS_DIR / "src" / "fp4_gemm_cutlass_sm103_extract.cu"
INCLUDE_DIR = CUTLASS_DIR / "include"
_ENV_PREFIX = "ATREX_NVFP4_FUSED_MOE_SM103_CUTLASS_GEMM2_FINALIZE"
_DEFAULT_COMPILED_TOP_K = 10
_SUPPORTED_COMPILED_TOP_K = frozenset({8, 10})
_EXPECTED_TACTIC_COUNT = 63
_EXT_FAST_CACHE: dict[tuple[object, ...], Any] = {}
_EXT_CACHE: dict[tuple[object, ...], Any] = {}
_PHYSICAL_SCALE_CACHE: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor]] = {}
_LAST_RUN_METADATA: dict[str, int | str | bool] = {}

_TACTIC_TILES = (
    ("128x64x128B", 128, 64, 128),
    ("128x256x128B", 128, 256, 128),
    ("128x128x256B", 128, 128, 256),
    ("128x256x256B", 128, 256, 256),
    ("128x128x768B", 128, 128, 768),
    ("128x192x768B", 128, 192, 768),
    ("128x256x768B", 128, 256, 768),
)
_TACTIC_CLUSTERS = (
    ("1x1x1", 1, 1, 1),
    ("1x2x1", 1, 2, 1),
    ("2x1x1", 2, 1, 1),
    ("2x2x1", 2, 2, 1),
    ("1x4x1", 1, 4, 1),
    ("4x2x1", 4, 2, 1),
    ("2x4x1", 2, 4, 1),
    ("4x4x1", 4, 4, 1),
    ("4x1x1", 4, 1, 1),
)
_BEST_TACTIC_ORIGINAL_INDEXES = (22, 20, 29, 4, 18)


def _build_tactic_infos() -> tuple[dict[str, int | str | bool], ...]:
    original: list[dict[str, int | str | bool]] = []
    for tile_idx, (tile, cta_m, cta_n, cta_k) in enumerate(_TACTIC_TILES):
        for cluster_idx, (cluster, cluster_m, cluster_n, cluster_k) in enumerate(_TACTIC_CLUSTERS):
            original_index = tile_idx * len(_TACTIC_CLUSTERS) + cluster_idx
            original.append(
                {
                    "original_index": original_index,
                    "tile": tile,
                    "cta_m": cta_m,
                    "cta_n": cta_n,
                    "cta_k": cta_k,
                    "cluster": cluster,
                    "cluster_m": cluster_m,
                    "cluster_n": cluster_n,
                    "cluster_k": cluster_k,
                    "supports_fused_finalize": cta_k != 768,
                }
            )
    reordered = [original[index] for index in _BEST_TACTIC_ORIGINAL_INDEXES]
    reordered.extend(
        info
        for index, info in enumerate(original)
        if index not in _BEST_TACTIC_ORIGINAL_INDEXES
    )
    for tactic, info in enumerate(reordered):
        info["tactic"] = tactic
    return tuple(reordered)


_TACTIC_INFOS = _build_tactic_infos()
if len(_TACTIC_INFOS) != _EXPECTED_TACTIC_COUNT:
    raise RuntimeError(
        f"expected {_EXPECTED_TACTIC_COUNT} CUTLASS GEMM2-finalize tactics, "
        f"got {len(_TACTIC_INFOS)}"
    )
_SMALL_M_DEFAULT_TACTIC = next(
    int(info["tactic"])
    for info in _TACTIC_INFOS
    if info["tile"] == "128x256x128B" and info["cluster"] == "1x2x1"
)
_LARGE_M_DEFAULT_TACTIC = next(
    int(info["tactic"])
    for info in _TACTIC_INFOS
    if info["tile"] == "128x256x128B" and info["cluster"] == "2x2x1"
)
_CONSERVATIVE_DEFAULT_TACTIC = _LARGE_M_DEFAULT_TACTIC
_DEFAULT_SELECTOR_SHAPE = {
    "n": 4096,
    "k": 1024,
    "top_k": _DEFAULT_COMPILED_TOP_K,
}
_DEFAULT_TOKEN_TACTICS = (
    (9999, _SMALL_M_DEFAULT_TACTIC),
    (2**31 - 1, _LARGE_M_DEFAULT_TACTIC),
)


def _truthy(value: str) -> bool:
    return value in ("1", "true", "TRUE", "on", "ON", "yes", "YES")


def _falsey(value: str) -> bool:
    return value in ("0", "false", "FALSE", "off", "OFF", "no", "NO")


def _build_tag() -> str:
    return os.environ.get(
        f"{_ENV_PREFIX}_BUILD_TAG",
        "task24_trt_w2_direct_finalize_v3",
    )


def supported_cutlass_gemm2_finalize_tactics() -> tuple[int, ...]:
    return tuple(range(len(_TACTIC_INFOS)))


def cutlass_gemm2_finalize_tactic_info(tactic: int) -> dict[str, int | str | bool]:
    selected = _normalize_cutlass_gemm2_finalize_tactic(tactic)
    return dict(_TACTIC_INFOS[selected])


def last_cutlass_gemm2_finalize_metadata() -> dict[str, int | str | bool]:
    return dict(_LAST_RUN_METADATA)


def _normalize_cutlass_gemm2_finalize_tactic(tactic: int) -> int:
    if 0 <= int(tactic) < len(_TACTIC_INFOS):
        return int(tactic)
    raise ValueError(
        "CUTLASS GEMM2-finalize tactic must be in "
        f"[0,{len(_TACTIC_INFOS)}), got {tactic}"
    )


def select_cutlass_gemm2_finalize_tactic(
    *,
    flat_m: int | None = None,
    tokens: int | None = None,
    n: int,
    k: int,
    top_k: int,
    env_override: int | None = None,
) -> int:
    raw_value = (
        env_override
        if env_override is not None
        else os.environ.get(f"{_ENV_PREFIX}_TACTIC")
    )
    if raw_value is not None:
        raw = int(raw_value)
        if raw != -1:
            return _normalize_cutlass_gemm2_finalize_tactic(raw)

    if (
        int(n) == int(_DEFAULT_SELECTOR_SHAPE["n"])
        and int(k) == int(_DEFAULT_SELECTOR_SHAPE["k"])
        and int(top_k) == int(_DEFAULT_SELECTOR_SHAPE["top_k"])
        and tokens is not None
    ):
        runtime_tokens = int(tokens)
        for max_tokens, tactic in _DEFAULT_TOKEN_TACTICS:
            if runtime_tokens <= int(max_tokens):
                return _normalize_cutlass_gemm2_finalize_tactic(tactic)

    del flat_m
    return _CONSERVATIVE_DEFAULT_TACTIC


def _cutlass_gemm2_finalize_tactic_cluster_m(tactic: int | None = None) -> int:
    selected = select_cutlass_gemm2_finalize_tactic(
        n=4096,
        k=1024,
        top_k=_DEFAULT_COMPILED_TOP_K,
        env_override=tactic,
    )
    return int(_TACTIC_INFOS[selected]["cluster_m"])


def cutlass_gemm2_finalize_required_route_tile_m(tactic: int | None = None) -> int:
    # Without runtime tokens, keep the conservative large-M route requirement.
    return 128 * _cutlass_gemm2_finalize_tactic_cluster_m(tactic)


def cutlass_gemm2_finalize_requires_m256_route(tactic: int | None = None) -> bool:
    return cutlass_gemm2_finalize_required_route_tile_m(tactic) >= 256


def _tactic_supports_fused_finalize(tactic: int) -> bool:
    return bool(_TACTIC_INFOS[_normalize_cutlass_gemm2_finalize_tactic(tactic)]["supports_fused_finalize"])


def _verbose_build() -> bool:
    return _truthy(
        os.environ.get(
            f"{_ENV_PREFIX}_VERBOSE",
            "0",
        )
    )


def _validate_route_metadata_enabled() -> bool:
    return _truthy(
        os.environ.get(
            f"{_ENV_PREFIX}_VALIDATE_ROUTE_METADATA",
            "0",
        )
    )


def _extension_env_key(compiled_top_k: int, trt_preshuffled_b: bool) -> tuple[object, ...]:
    return (
        _build_tag(),
        ("compiled_top_k", int(compiled_top_k)),
        ("trt_preshuffled_b", bool(trt_preshuffled_b)),
        ("tile_scheduler_kind", _tile_scheduler_kind()),
        ("tactic_count", len(_TACTIC_INFOS)),
    )


def _tile_scheduler_kind() -> int:
    raw = os.environ.get(
        f"{_ENV_PREFIX}_TILE_SCHEDULER",
        "static_sm100",
    ).strip().lower()
    aliases = {
        "0": 0,
        "persistent": 0,
        "persistent_sm100": 0,
        "default": 3,
        "3": 3,
        "static": 3,
        "static_persistent": 3,
        "static_sm100": 3,
    }
    try:
        return aliases[raw]
    except KeyError as exc:
        valid = ", ".join(sorted(k for k in aliases if not k.isdigit()))
        raise ValueError(f"unknown {_ENV_PREFIX}_TILE_SCHEDULER={raw!r}; valid: {valid}") from exc


def _max_physical_scale_cache_entries() -> int:
    return int(
        os.environ.get(
            f"{_ENV_PREFIX}_PHYSICAL_SCALE_CACHE_ENTRIES",
            "4",
        )
    )


def _tensor_identity_key(tensor: torch.Tensor) -> tuple[object, ...]:
    device = tensor.device
    try:
        version: object = int(getattr(tensor, "_version", 0))
    except RuntimeError:
        # Inference tensors created inside ``torch.inference_mode()`` do not
        # track a version counter and raise RuntimeError on ``._version`` /
        # ``.version_counter`` access. Fall back to a version-free identity —
        # safe for cache lookup because such tensors are immutable within
        # their inference-mode scope.
        version = None
    return (
        int(tensor.data_ptr()),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        device.type,
        None if device.index is None else int(device.index),
        version,
    )


def _extra_cuda_cflags(compiled_top_k: int, trt_preshuffled_b: bool) -> list[str]:
    from flashinfer.jit.gemm.core import current_compilation_context

    flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10, 11, 12]
    )
    flags += [
        "-DENABLE_BF16",
        "-DENABLE_FP4",
        "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
        f"-DGEMM2_FINALIZE_COMPILED_TOP_K={int(compiled_top_k)}",
        f"-DGEMM2_FINALIZE_TRT_PRESHUFFLED_B={1 if trt_preshuffled_b else 0}",
        f"-DGEMM2_FINALIZE_TILE_SCHEDULER_KIND={int(_tile_scheduler_kind())}",
        "-DNDEBUG",
    ]
    return flags


def _load_cutlass_gemm2_finalize_extension(compiled_top_k: int, trt_preshuffled_b: bool) -> Any:
    fast_key = _extension_env_key(compiled_top_k, trt_preshuffled_b)
    cached = _EXT_FAST_CACHE.get(fast_key)
    if cached is not None:
        return cached

    from flashinfer.jit.gemm.core import (
        gen_gemm_sm103_module_cutlass_fp4,
        gen_jit_spec,
        jit_env,
    )

    base_spec = gen_gemm_sm103_module_cutlass_fp4()
    generated_sources = [
        source
        for source in base_spec.sources
        if Path(source).name.startswith("fp4_gemm_cutlass___nv_bfloat16_")
    ]
    if len(generated_sources) != len(_TACTIC_TILES):
        names = ", ".join(Path(source).name for source in generated_sources)
        raise RuntimeError(
            "FlashInfer did not generate the expected 7 CUTLASS bf16 SM103 FP4 sources; "
            f"got {len(generated_sources)}: {names}"
        )

    include_paths = [
        INCLUDE_DIR / "flashinfer_modified",
        INCLUDE_DIR / "cutlass_modified",
        jit_env.FLASHINFER_DATA / "cutlass" / "include",
        jit_env.FLASHINFER_DATA / "cutlass" / "tools" / "util" / "include",
    ]
    extra_cuda_cflags = _extra_cuda_cflags(compiled_top_k, trt_preshuffled_b)
    config_hash = hashlib.sha1(
        "\n".join(
            extra_cuda_cflags
            + [str(path) for path in include_paths]
            + [Path(source).name for source in generated_sources]
        ).encode()
    ).hexdigest()[:8]
    name = f"atrex_sm103_cutlass_gemm2_finalize_{_build_tag()}_{config_hash}"
    cache_key = (name, tuple(extra_cuda_cflags), tuple(str(path) for path in include_paths))
    cached = _EXT_CACHE.get(cache_key)
    if cached is not None:
        _EXT_FAST_CACHE[fast_key] = cached
        return cached

    old_verbose = os.environ.get("FLASHINFER_JIT_VERBOSE")
    if _verbose_build():
        os.environ["FLASHINFER_JIT_VERBOSE"] = "1"
    try:
        ext = gen_jit_spec(
            name,
            [SRC, *generated_sources],
            extra_cuda_cflags=extra_cuda_cflags,
            extra_cflags=["-DFAST_BUILD", "-DNDEBUG"],
            extra_include_paths=include_paths,
        ).build_and_load()
    finally:
        if _verbose_build():
            if old_verbose is None:
                os.environ.pop("FLASHINFER_JIT_VERBOSE", None)
            else:
                os.environ["FLASHINFER_JIT_VERBOSE"] = old_verbose
    _EXT_CACHE[cache_key] = ext
    _EXT_FAST_CACHE[fast_key] = ext
    return ext


def _workspace(ext: Any, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m = int(a.shape[0])
    n = int(b.shape[1])
    k = int(b.shape[2]) * 2
    batch_count = 1
    required = int(ext.fp4_gemm_workspace_size(m, n, k, batch_count))
    return torch.empty((required,), device=a.device, dtype=torch.uint8)


def _require_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> None:
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if dtype is not None and tensor.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")


def _scale_bits(
    name: str,
    tensor: torch.Tensor,
    *,
    device: torch.device,
    physicalize_6d: bool,
) -> torch.Tensor:
    _require_tensor(name, tensor, device=device)
    if tensor.dtype == torch.uint8:
        bits = tensor
    elif tensor.dtype == torch.float8_e4m3fn:
        bits = tensor.view(torch.uint8)
    else:
        raise ValueError(f"{name} must be uint8/float8_e4m3fn scale storage, got {tensor.dtype}")
    if (
        physicalize_6d
        and bits.ndim == 6
        and bits.shape[0] == 32
        and bits.shape[1] == 4
        and bits.shape[3] == 4
    ):
        key = ("physical_6d", _tensor_identity_key(bits))
        cached = _PHYSICAL_SCALE_CACHE.get(key)
        if cached is not None:
            source_bits, physical_bits = cached
            if source_bits.device == device and physical_bits.device == device:
                return physical_bits
        max_entries = _max_physical_scale_cache_entries()
        if max_entries <= 0:
            return bits.permute(5, 2, 4, 0, 1, 3).contiguous()
        if len(_PHYSICAL_SCALE_CACHE) >= max_entries:
            _PHYSICAL_SCALE_CACHE.clear()
        physical_bits = bits.permute(5, 2, 4, 0, 1, 3).contiguous()
        _PHYSICAL_SCALE_CACHE[key] = (bits, physical_bits)
        return physical_bits
    if bits.is_contiguous():
        return bits
    return bits.contiguous()


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _validate_route_metadata(
    *,
    flat_m: int,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    token_final_scales_flat: torch.Tensor,
) -> None:
    if flat_m <= 0:
        return
    active_tiles = _ceil_div(flat_m, 128)
    if int(tile_idx_to_mn_limit.numel()) < active_tiles:
        raise ValueError(
            "CUTLASS GEMM2-finalize route metadata validation requires tile_idx_to_mn_limit "
            f"to cover {active_tiles} tiles, got {tile_idx_to_mn_limit.numel()}"
        )
    if int(permuted_idx_to_expanded_idx.numel()) < flat_m:
        raise ValueError(
            "CUTLASS GEMM2-finalize route metadata validation requires permuted_idx_to_expanded_idx "
            f"to cover Mflat={flat_m}, got {permuted_idx_to_expanded_idx.numel()}"
        )

    device = tile_idx_to_mn_limit.device
    limits = tile_idx_to_mn_limit[:active_tiles].to(device=device, dtype=torch.long)
    min_limit = int(limits.min().item())
    max_limit = int(limits.max().item())
    if min_limit < 0:
        raise ValueError(f"CUTLASS GEMM2-finalize route metadata has negative mn_limit: {min_limit}")
    if max_limit > flat_m:
        raise ValueError(
            f"CUTLASS GEMM2-finalize route metadata mn_limit exceeds Mflat: max={max_limit}, Mflat={flat_m}"
        )

    rows = torch.arange(flat_m, device=device, dtype=torch.long)
    tile_rows = rows // 128
    valid = rows < limits.index_select(0, tile_rows)
    if not bool(valid.any().item()):
        return
    expanded = permuted_idx_to_expanded_idx[:flat_m].to(device=device, dtype=torch.long)
    expanded_valid = expanded[valid]
    min_expanded = int(expanded_valid.min().item())
    max_expanded = int(expanded_valid.max().item())
    if min_expanded < 0:
        raise ValueError(
            f"CUTLASS GEMM2-finalize route metadata has row-limit row with negative expanded_idx: {min_expanded}"
        )
    if max_expanded >= int(token_final_scales_flat.numel()):
        raise ValueError(
            "CUTLASS GEMM2-finalize route metadata expanded_idx exceeds token_final_scales: "
            f"max={max_expanded}, limit={token_final_scales_flat.numel()}"
        )


def cutlass_gemm2_finalize_nvfp4(
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    alpha: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    cutlass_num_non_exiting_tiles: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    token_final_scales: torch.Tensor,
    top_k: int,
    tactic: int,
    out: torch.Tensor | None = None,
    alpha_is_one: bool = False,
    trt_preshuffled_b: bool = False,
) -> torch.Tensor:
    """Run flat-global CUTLASS GEMM2 with Mflat-gated finalize fusion."""

    if a.ndim != 2 or b.ndim != 3:
        raise ValueError(f"CUTLASS GEMM2-finalize expects a [Mflat,K/2] and b [E,N,K/2], got {a.shape}, {b.shape}")
    if a.dtype != torch.uint8 or b.dtype != torch.uint8:
        raise ValueError(f"CUTLASS GEMM2-finalize expects uint8 packed FP4 operands, got {a.dtype}, {b.dtype}")
    if a.shape[1] != b.shape[2]:
        raise ValueError(f"CUTLASS GEMM2-finalize K mismatch: a.shape={tuple(a.shape)} b.shape={tuple(b.shape)}")
    device = a.device
    _require_tensor("b", b, device=device, dtype=torch.uint8)
    a_scale = _scale_bits("a_scale", a_scale, device=device, physicalize_6d=False)
    b_scale = _scale_bits("b_scale", b_scale, device=device, physicalize_6d=True)
    _require_tensor("alpha", alpha, device=device, dtype=torch.float32)
    _require_tensor("tile_idx_to_expert_idx", tile_idx_to_expert_idx, device=device, dtype=torch.int32)
    _require_tensor("tile_idx_to_mn_limit", tile_idx_to_mn_limit, device=device, dtype=torch.int32)
    _require_tensor(
        "cutlass_num_non_exiting_tiles",
        cutlass_num_non_exiting_tiles,
        device=device,
        dtype=torch.int32,
    )
    if cutlass_num_non_exiting_tiles.numel() < 1:
        raise ValueError("cutlass_num_non_exiting_tiles must contain at least one int32 value")
    _require_tensor(
        "permuted_idx_to_expanded_idx",
        permuted_idx_to_expanded_idx,
        device=device,
        dtype=torch.int32,
    )
    _require_tensor("token_final_scales", token_final_scales, device=device, dtype=torch.float32)
    if alpha.ndim != 1 or int(alpha.numel()) != int(b.shape[0]):
        raise ValueError(
            "CUTLASS GEMM2-finalize output2_scale_scalar must contain one value per expert: "
            f"expected {b.shape[0]}, got shape={tuple(alpha.shape)}"
        )

    token_final_scales_flat = token_final_scales.reshape(-1).contiguous()
    if token_final_scales_flat.numel() % int(top_k) != 0:
        raise ValueError(
            f"token_final_scales size {token_final_scales_flat.numel()} must be divisible by top_k={top_k}"
        )
    compiled_top_k = int(top_k)
    if compiled_top_k not in _SUPPORTED_COMPILED_TOP_K:
        raise ValueError(
            "CUTLASS GEMM2-finalize supports top_k in "
            f"{sorted(_SUPPORTED_COMPILED_TOP_K)}, got {compiled_top_k}"
        )
    tokens = int(token_final_scales_flat.numel() // int(top_k))
    n = int(b.shape[1])
    if out is None:
        out = torch.empty((tokens, n), device=device, dtype=torch.bfloat16)
    else:
        _require_tensor("out", out, device=device, dtype=torch.bfloat16)
        if out.shape != (tokens, n):
            raise ValueError(f"out must be [{tokens},{n}], got {tuple(out.shape)}")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous for CUTLASS GEMM2-finalize")

    flat_m = int(a.shape[0])
    if flat_m % 128 != 0:
        raise ValueError(
            f"CUTLASS GEMM2-finalize M must be a multiple of 128, got {flat_m}"
        )
    tile_idx_to_expert_idx = tile_idx_to_expert_idx.contiguous()
    tile_idx_to_mn_limit = tile_idx_to_mn_limit.contiguous()
    cutlass_num_non_exiting_tiles = cutlass_num_non_exiting_tiles.contiguous()
    permuted_idx_to_expanded_idx = permuted_idx_to_expanded_idx.contiguous()
    if _validate_route_metadata_enabled():
        _validate_route_metadata(
            flat_m=flat_m,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            token_final_scales_flat=token_final_scales_flat,
        )
    runtime_k = int(a.shape[1]) * 2
    if tactic is None:
        raise ValueError(
            "CUTLASS GEMM2-finalize requires an explicit tactic matching route metadata"
        )
    selected_tactic = _normalize_cutlass_gemm2_finalize_tactic(tactic)
    tactic_info = _TACTIC_INFOS[selected_tactic]
    _LAST_RUN_METADATA.clear()
    _LAST_RUN_METADATA.update(
        {
            "flat_m": flat_m,
            "tokens": tokens,
            "n": n,
            "k": runtime_k,
            "top_k": int(top_k),
            "selected_tactic": selected_tactic,
            "tile": str(tactic_info["tile"]),
            "cluster": str(tactic_info["cluster"]),
            "supports_fused_finalize": bool(tactic_info["supports_fused_finalize"]),
            "finalize_accumulator": "bf16_atomic",
            "trt_preshuffled_b": bool(trt_preshuffled_b),
        }
    )

    if not _tactic_supports_fused_finalize(selected_tactic):
        raise ValueError(
            "CUTLASS GEMM2-finalize tactic "
            f"{selected_tactic} ({tactic_info['tile']}, cluster {tactic_info['cluster']}) "
            "is compiled but does not support fused finalize at runtime"
        )

    ext = _load_cutlass_gemm2_finalize_extension(compiled_top_k, bool(trt_preshuffled_b))
    if trt_preshuffled_b and n % 32 != 0:
        raise ValueError(f"TRT-preshuffled GEMM2 output N must be divisible by 32, got {n}")
    out.zero_()
    ext.fp4_gemm_flat_global_tilemap_finalize(
        a,
        b,
        a_scale,
        b_scale,
        alpha,
        out,
        _workspace(ext, a, b),
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        cutlass_num_non_exiting_tiles,
        permuted_idx_to_expanded_idx,
        token_final_scales_flat,
        int(top_k),
        selected_tactic,
        int(bool(trt_preshuffled_b)),
    )
    return out
