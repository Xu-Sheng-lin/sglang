"""
Triton kernel for scatter-writing KV cache in x-interleaved (K) and transposed (V) layout.

This matches rtp-llm's getKLocalIdx / getVLocalIdx indexing from kv_cache_utils.h:
  K layout per (block, head): [head_dim//X, block_size, X]  (X=8 for bf16)
  V layout per (block, head): [head_dim, block_size]

Used by paged_attention_rocm for decode and mha_batch_prefill_func for prefill.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_kv_cache_kernel(
    K_SRC,      # [N, kv_heads, head_dim] — source K
    V_SRC,      # [N, kv_heads, head_dim] — source V
    K_CACHE,    # [num_blocks, kv_heads, head_dim//X, block_size, X] — flat
    V_CACHE,    # [num_blocks, kv_heads, head_dim, block_size] — flat
    SLOT_IDS,   # [N] — slot index per token
    kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    X: tl.constexpr,            # vectorize_size = 16 // element_size (8 for bf16)
    BLOCK_D: tl.constexpr,      # >= head_dim, power of 2
):
    """Scatter-write K/V from [N, kv_heads, head_dim] to x-interleaved/transposed cache.

    Grid: (N, kv_heads)
    Each program writes one (token, head) pair — head_dim elements for both K and V.
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    slot_id = tl.load(SLOT_IDS + token_idx)
    block_id = slot_id // block_size
    pos = slot_id % block_size

    # Source: K_SRC[token_idx, head_idx, :] and V_SRC[token_idx, head_idx, :]
    src_base = token_idx * kv_heads * head_dim + head_idx * head_dim
    dims = tl.arange(0, BLOCK_D)
    mask = dims < head_dim

    # Load source K and V
    k_vals = tl.load(K_SRC + src_base + dims, mask=mask)
    v_vals = tl.load(V_SRC + src_base + dims, mask=mask)

    # K cache strides: [num_blocks, kv_heads, head_dim//X, block_size, X]
    #   block_stride = kv_heads * head_dim  (since head_dim//X * block_size * X = head_dim * block_size... no)
    #   Actually: per-block-head size = (head_dim // X) * block_size * X = head_dim * block_size
    k_per_block_head = head_dim * block_size  # = (head_dim // X) * block_size * X
    k_base = block_id * (kv_heads * k_per_block_head) + head_idx * k_per_block_head

    # K offset: dim_group * block_size * X + pos * X + dim_in_group
    # where dim_group = dims // X, dim_in_group = dims % X
    k_offsets = (dims // X) * (block_size * X) + pos * X + (dims % X)
    tl.store(K_CACHE + k_base + k_offsets, k_vals, mask=mask)

    # V cache strides: [num_blocks, kv_heads, head_dim, block_size]
    #   per-block-head size = head_dim * block_size
    v_per_block_head = head_dim * block_size
    v_base = block_id * (kv_heads * v_per_block_head) + head_idx * v_per_block_head

    # V offset: dim * block_size + pos
    v_offsets = dims * block_size + pos
    tl.store(V_CACHE + v_base + v_offsets, v_vals, mask=mask)


@triton.jit
def _gather_kv_cache_kernel(
    K_DST,      # [N, kv_heads, head_dim] — output K
    V_DST,      # [N, kv_heads, head_dim] — output V
    K_CACHE,    # [num_blocks, kv_heads, head_dim//X, block_size, X] — flat
    V_CACHE,    # [num_blocks, kv_heads, head_dim, block_size] — flat
    SLOT_IDS,   # [N] — slot index per token
    kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    X: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Gather K/V from x-interleaved/transposed cache to [N, kv_heads, head_dim]."""
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    slot_id = tl.load(SLOT_IDS + token_idx)
    block_id = slot_id // block_size
    pos = slot_id % block_size

    dst_base = token_idx * kv_heads * head_dim + head_idx * head_dim
    dims = tl.arange(0, BLOCK_D)
    mask = dims < head_dim

    # K cache
    k_per_block_head = head_dim * block_size
    k_base = block_id * (kv_heads * k_per_block_head) + head_idx * k_per_block_head
    k_offsets = (dims // X) * (block_size * X) + pos * X + (dims % X)
    k_vals = tl.load(K_CACHE + k_base + k_offsets, mask=mask)
    tl.store(K_DST + dst_base + dims, k_vals, mask=mask)

    # V cache
    v_per_block_head = head_dim * block_size
    v_base = block_id * (kv_heads * v_per_block_head) + head_idx * v_per_block_head
    v_offsets = dims * block_size + pos
    v_vals = tl.load(V_CACHE + v_base + v_offsets, mask=mask)
    tl.store(V_DST + dst_base + dims, v_vals, mask=mask)


def gather_kv_cache(
    k: torch.Tensor,        # [N, kv_heads, head_dim] — output
    v: torch.Tensor,        # [N, kv_heads, head_dim] — output
    k_cache: torch.Tensor,  # [num_blocks, kv_heads, head_dim//X, block_size, X]
    v_cache: torch.Tensor,  # [num_blocks, kv_heads, head_dim, block_size]
    slot_ids: torch.Tensor, # [N]
):
    """Gather K/V from x-interleaved K cache and transposed V cache."""
    N = k.shape[0]
    if N == 0:
        return
    kv_heads = k.shape[1]
    head_dim = k.shape[2]
    block_size = k_cache.shape[3]
    X = 16 // k.element_size()

    BLOCK_D = triton.next_power_of_2(head_dim)
    slot_ids = slot_ids.contiguous()

    _gather_kv_cache_kernel[(N, kv_heads)](
        k, v, k_cache, v_cache, slot_ids,
        kv_heads=kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        X=X,
        BLOCK_D=BLOCK_D,
    )


def scatter_kv_cache(
    k: torch.Tensor,        # [N, kv_heads, head_dim]
    v: torch.Tensor,        # [N, kv_heads, head_dim]
    k_cache: torch.Tensor,  # [num_blocks, kv_heads, head_dim//X, block_size, X]
    v_cache: torch.Tensor,  # [num_blocks, kv_heads, head_dim, block_size]
    slot_ids: torch.Tensor, # [N]
):
    """Scatter-write K/V to x-interleaved K cache and transposed V cache.

    Args:
        k: Key tensor [N, kv_heads, head_dim]
        v: Value tensor [N, kv_heads, head_dim]
        k_cache: Key cache [num_blocks, kv_heads, head_dim//X, block_size, X]
        v_cache: Value cache [num_blocks, kv_heads, head_dim, block_size]
        slot_ids: Slot indices [N], where block_id = slot_id // block_size, pos = slot_id % block_size
    """
    N = k.shape[0]
    if N == 0:
        return
    kv_heads = k.shape[1]
    head_dim = k.shape[2]
    block_size = k_cache.shape[3]
    X = 16 // k.element_size()  # 8 for bf16, 4 for fp32

    BLOCK_D = triton.next_power_of_2(head_dim)

    k = k.contiguous()
    v = v.contiguous()
    slot_ids = slot_ids.contiguous()

    _scatter_kv_cache_kernel[(N, kv_heads)](
        k, v, k_cache, v_cache, slot_ids,
        kv_heads=kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        X=X,
        BLOCK_D=BLOCK_D,
    )

