"""
Gluon-based paged attention decode kernels.
Derived from atrex/src/triton/pa_decode_gluon.py
"""
import triton
import triton.language as tl

from triton.experimental import gluon
import triton.experimental.gluon.language as gl


@gluon.jit
def _pa_decode_dot_kernel(
    exp_sums_ptr,       # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    max_logits_ptr,     # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    logits_ptr,         # [num_seqs, num_kv_heads, max_parts, q_grp_sz, head_sz]
    q_ptr,              # [num_seqs, num_kv_heads * query_grp_sz, head_sz]
    k_cache_ptr,        # [num_blks, num_kv_heads, head_sz/x, kv_blk_sz, x]
    v_cache_ptr,        # [num_blks, num_kv_heads, head_sz, kv_blk_sz]
    blk_tables_ptrs,    # [num_seqs, max_num_blks_per_seq]
    seq_lens_ptr,       # [num_seqs]
    scale,
    alibi_slopes,
    stride_max_logits_s,
    stride_max_logits_nh,
    stride_max_logits_p,
    stride_logits_s,
    stride_logits_nh,
    stride_logits_p,
    stride_logits_g,
    stride_q_s,
    stride_q_nh,
    stride_k_b,
    stride_k_nh,
    stride_k_hz,
    stride_k_bz,
    stride_v_b,
    stride_v_nh,
    stride_v_hz,
    stride_bt_s,
    USE_ALIBI_SLOPES: gl.constexpr,
    HEAD_SZ: gl.constexpr,
    QUERY_GRP_SZ: gl.constexpr,
    QUERY_GRP_SZ_POW2: gl.constexpr,
    KV_BLK_SZ: gl.constexpr,
    KV_BLK_SZ_POW2: gl.constexpr,
    SEQ_PARTITION_SZ: gl.constexpr,
):
    seq_idx = gl.program_id(0)
    kv_head_idx = gl.program_id(1)
    seq_part_idx = gl.program_id(2)

    log2e: gl.constexpr = 1.4426950408889634
    CONTIGUOUS_KV_ELEMS_16B_LOAD: gl.constexpr = 8
    K_HEAD_SZ_SPLIT: gl.constexpr = HEAD_SZ // CONTIGUOUS_KV_ELEMS_16B_LOAD

    seq_len = gl.load(seq_lens_ptr + seq_idx)

    seq_start_idx = seq_part_idx * SEQ_PARTITION_SZ
    if seq_start_idx >= seq_len:
        return

    seq_end_idx = gl.minimum(seq_start_idx + SEQ_PARTITION_SZ, seq_len)
    MAX_NUM_KV_BLKS: gl.constexpr = (SEQ_PARTITION_SZ + KV_BLK_SZ - 1) // KV_BLK_SZ
    num_kv_blks = gl.cdiv(seq_end_idx - seq_start_idx, KV_BLK_SZ)

    if HEAD_SZ == 64:
        blocked_q: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 4],
            threads_per_warp=[4, 16],
            warps_per_cta=[4, 1],
            order=[1, 0],
        )
    elif HEAD_SZ >= 128:
        blocked_q: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 8],
            threads_per_warp=[4, 16],
            warps_per_cta=[4, 1],
            order=[1, 0],
        )

    shared_a_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, order=[1, 0])
    blocked_k: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 2, 1, 8],
        threads_per_warp=[1, 4, 16, 1],
        warps_per_cta=[4, 1, 1, 1],
        order=[3, 2, 1, 0],
    )
    qk_mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16], transposed=True, warps_per_cta=[1, 4]
    )
    qk_lhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=qk_mfma_layout, k_width=16
    )
    qk_rhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=qk_mfma_layout, k_width=16
    )

    pv_mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16], transposed=False, warps_per_cta=[1, 4]
    )
    pv_lhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=pv_mfma_layout, k_width=16
    )
    pv_rhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=pv_mfma_layout, k_width=16
    )

    query_grp_sz_layout: gl.constexpr = gl.SliceLayout(1, blocked_q)
    head_sz_layout: gl.constexpr = gl.SliceLayout(0, blocked_q)

    blk_id_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, blocked_k)))
    head_sz_div_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, blocked_k)))
    blk_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, blocked_k)))
    contiguous_kv_elems_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, blocked_k)))

    q_grp_offs = gl.arange(0, QUERY_GRP_SZ_POW2, layout=query_grp_sz_layout)
    head_sz_offs = gl.arange(0, HEAD_SZ, layout=head_sz_layout)
    head_sz_div_offs = gl.arange(0, K_HEAD_SZ_SPLIT, layout=head_sz_div_layout)
    blk_offs = gl.arange(0, KV_BLK_SZ_POW2, layout=blk_layout)
    contiguous_kv_elems_offs = gl.arange(0, CONTIGUOUS_KV_ELEMS_16B_LOAD, layout=contiguous_kv_elems_layout)

    kv_blk_start = seq_part_idx * MAX_NUM_KV_BLKS
    qk_row_offs = gl.arange(0, QUERY_GRP_SZ_POW2, layout=gl.SliceLayout(1, qk_mfma_layout))
    qk_col_offs = kv_blk_start * KV_BLK_SZ_POW2 + gl.arange(0, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2, layout=gl.SliceLayout(0, qk_mfma_layout))

    if not USE_ALIBI_SLOPES:
        alibi_slope = gl.zeros([QUERY_GRP_SZ_POW2], dtype=gl.float32)
    else:
        alibi_slope = gl.amd.cdna3.buffer_load(ptr=alibi_slopes + kv_head_idx * QUERY_GRP_SZ, offsets=qk_row_offs, mask=qk_row_offs < QUERY_GRP_SZ)

    blk_ids = gl.arange(0, MAX_NUM_KV_BLKS, layout=blk_id_layout)
    masked_blk_ids = gl.where(blk_ids < num_kv_blks, blk_ids, 0)
    blk_tables_start_ptr = blk_tables_ptrs + seq_idx * stride_bt_s
    kv_blk_nums = gl.amd.cdna3.buffer_load(ptr=blk_tables_start_ptr + kv_blk_start, offsets=masked_blk_ids)

    q_offs = (
        seq_idx * stride_q_s
        + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_q_nh
        + head_sz_offs[None, :]
    )
    q_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_sz_offs[None, :] < HEAD_SZ)
    q = gl.amd.cdna3.buffer_load(ptr=q_ptr, offsets=q_offs, mask=q_mask)
    q = (q * scale).to(q.dtype)
    q_shared = gl.allocate_shared_memory(q.dtype, q.shape, shared_a_layout, q)

    k_blk_offs = (
        kv_blk_nums[:, None, None, None].to(tl.int64) * stride_k_b
        + kv_head_idx * stride_k_nh
        + head_sz_div_offs[None, :, None, None] * stride_k_hz
        + blk_offs[None, None, :, None] * CONTIGUOUS_KV_ELEMS_16B_LOAD
        + contiguous_kv_elems_offs[None, None, None, :]
    )
    k = gl.load(k_cache_ptr + k_blk_offs)
    kt_temp = tl.permute(k, [1, 3, 0, 2])
    kt = tl.reshape(kt_temp, [HEAD_SZ, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2])
    accumulator = gl.zeros((QUERY_GRP_SZ_POW2, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2), dtype=gl.float32, layout=qk_mfma_layout)
    qc = q_shared.load(qk_lhs_layout)
    kc = gl.convert_layout(kt, layout=qk_rhs_layout)
    qk = gl.amd.cdna3.mfma(qc, kc, accumulator)

    if USE_ALIBI_SLOPES:
        qk += (alibi_slope[:, None] * (qk_col_offs - seq_len + 1)[None, :]).to(gl.float32)
    qk = gl.where(
        (qk_row_offs[:, None] < QUERY_GRP_SZ) & (qk_col_offs[None, :] < seq_len),
        qk,
        float("-inf"),
    )

    max_logit_new = gl.max(qk, axis=1)
    p = tl.math.exp2((qk - max_logit_new[:, None]) * log2e)
    exp_sum = gl.sum(p, axis=1)
    p = p.to(q.dtype)

    m_l_base_offs = gl.arange(0, QUERY_GRP_SZ_POW2, layout=gl.SliceLayout(1, qk_mfma_layout))
    m_l_offs = (
        seq_idx * stride_max_logits_s
        + kv_head_idx * stride_max_logits_nh
        + seq_part_idx * stride_max_logits_p
        + m_l_base_offs
    )
    m_l_grp_mask = m_l_base_offs < QUERY_GRP_SZ
    gl.amd.cdna3.buffer_store(stored_value=max_logit_new, ptr=max_logits_ptr, offsets=m_l_offs, mask=m_l_grp_mask)
    gl.amd.cdna3.buffer_store(stored_value=exp_sum, ptr=exp_sums_ptr, offsets=m_l_offs, mask=m_l_grp_mask)

    blocked_v_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1, 16],
        threads_per_warp=[4, 16, 1],
        warps_per_cta=[1, 4, 1],
        order=[2, 1, 0],
    )
    v_dim0_offs = gl.arange(0, MAX_NUM_KV_BLKS, layout=gl.SliceLayout(1, gl.SliceLayout(2, blocked_v_layout)))
    v_dim1_offs = gl.arange(0, HEAD_SZ, layout=gl.SliceLayout(0, gl.SliceLayout(2, blocked_v_layout)))
    v_dim2_offs = gl.arange(0, KV_BLK_SZ_POW2, layout=gl.SliceLayout(0, gl.SliceLayout(1, blocked_v_layout)))

    kv_blk_nums2 = gl.convert_layout(kv_blk_nums, layout=gl.SliceLayout(1, gl.SliceLayout(2, blocked_v_layout)))
    v_blk_offs = (
        kv_blk_nums2[:, None, None].to(tl.int64) * stride_v_b
        + kv_head_idx * stride_v_nh
        + v_dim1_offs[None, :, None] * stride_v_hz
        + v_dim2_offs[None, None, :]
    )

    v_0 = gl.load(v_cache_ptr + v_blk_offs)
    v = v_0.to(q.dtype)
    v = gl.permute(v, [0, 2, 1])
    v = gl.reshape(v, [MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2, HEAD_SZ])

    accumulator2 = gl.zeros((QUERY_GRP_SZ_POW2, HEAD_SZ), dtype=gl.float32, layout=pv_mfma_layout)
    pc = gl.convert_layout(p, layout=pv_lhs_layout)
    vc = gl.convert_layout(v, layout=pv_rhs_layout)
    acc = gl.amd.cdna3.mfma(pc, vc, accumulator2)
    exp_sum = gl.convert_layout(exp_sum[:, None], layout=pv_mfma_layout)
    exp_sum = tl.broadcast_to(exp_sum, QUERY_GRP_SZ_POW2, HEAD_SZ)
    acc = acc / exp_sum
    acc = acc.to(q.dtype)

    o_grp_offs = gl.arange(0, QUERY_GRP_SZ_POW2, layout=gl.SliceLayout(1, pv_mfma_layout))
    o_head_sz_offs = gl.arange(0, HEAD_SZ, layout=gl.SliceLayout(0, pv_mfma_layout))
    o_mask = (o_grp_offs[:, None] < QUERY_GRP_SZ) & (o_head_sz_offs[None, :] < HEAD_SZ)
    logits_offs = seq_idx * stride_logits_s
    logits_offs += kv_head_idx * stride_logits_nh
    logits_offs += (
        seq_part_idx * stride_logits_p
        + o_grp_offs[:, None] * stride_logits_g
        + o_head_sz_offs[None, :]
    )
    gl.amd.cdna3.buffer_store(stored_value=acc, ptr=logits_ptr, offsets=logits_offs, mask=o_mask)


@gluon.jit
def _pa_decode_reduce_kernel_64k(
    out_ptr,
    exp_sums_ptr,
    max_logits_ptr,
    logits_ptrs,
    seq_lens_ptr,
    stride_o_s,
    stride_o_h,
    stride_exp_sums_s,
    stride_exp_sums_h,
    stride_exp_sums_p,
    stride_logits_s,
    stride_logits_h,
    stride_logits_p,
    stride_logits_g,
    HEAD_SZ: gl.constexpr,
    QUERY_GRP_SZ: gl.constexpr,
    SEQ_PARTITION_SZ: gl.constexpr,
    MAX_NUM_SEQ_PARTITIONS_POW2: gl.constexpr,
):
    blocked_full: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 16],
        threads_per_warp=[16, 4],
        warps_per_cta=[1, 4],
        order=[1, 0],
    )
    slice_full_dim1: gl.constexpr = gl.SliceLayout(1, blocked_full)
    slice_full_dim0: gl.constexpr = gl.SliceLayout(0, blocked_full)

    blocked_out: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, 16],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    slice_out_dim0: gl.constexpr = gl.SliceLayout(0, blocked_out)

    seq_idx = gl.program_id(0)
    q_head_idx = gl.program_id(1)

    kv_head_idx = q_head_idx // QUERY_GRP_SZ
    grp_idx = q_head_idx % QUERY_GRP_SZ

    seq_len = gl.load(seq_lens_ptr + seq_idx)
    log2_sz: gl.constexpr = 9
    num_partitions = (seq_len + SEQ_PARTITION_SZ - 1) >> log2_sz

    ml_base_scalar = (
        seq_idx * stride_exp_sums_s
        + kv_head_idx * stride_exp_sums_h
        + grp_idx
    )
    logits_base_scalar = (
        seq_idx * stride_logits_s
        + kv_head_idx * stride_logits_h
        + grp_idx * stride_logits_g
    )

    part_offs = gl.arange(0, MAX_NUM_SEQ_PARTITIONS_POW2, layout=slice_full_dim1)
    part_mask = part_offs < num_partitions
    ml_offs = ml_base_scalar + part_offs * stride_exp_sums_p

    all_max_logits = gl.amd.cdna3.buffer_load(
        ptr=max_logits_ptr, offsets=ml_offs, mask=part_mask
    )
    global_max = gl.max(all_max_logits, axis=0)

    all_exp_sums = gl.amd.cdna3.buffer_load(
        ptr=exp_sums_ptr, offsets=ml_offs, mask=part_mask
    )
    rescaled = all_exp_sums * gl.exp(all_max_logits - global_max)
    global_exp_sum = gl.sum(rescaled, axis=0)

    p = rescaled / global_exp_sum

    head_offs = gl.arange(0, HEAD_SZ, layout=slice_full_dim0)

    tile_offs = (
        logits_base_scalar
        + gl.expand_dims(part_offs, axis=1) * stride_logits_p
        + gl.expand_dims(head_offs, axis=0)
    )
    tile_mask = gl.expand_dims(part_mask, axis=1)
    tile = gl.load(logits_ptrs + tile_offs, mask=tile_mask)

    p_2d = gl.expand_dims(p, axis=1)
    weighted = tile.to(gl.float32) * p_2d

    result = gl.sum(weighted, axis=0)

    result_2d = gl.reshape(result, (1, HEAD_SZ))
    result_out = gl.convert_layout(result_2d, blocked_out)
    out = result_out.to(out_ptr.dtype.element_ty)

    head_offs_out = gl.arange(0, HEAD_SZ, layout=slice_out_dim0)
    out_offs = (
        seq_idx * stride_o_s
        + q_head_idx * stride_o_h
        + gl.expand_dims(head_offs_out, axis=0)
    )
    gl.store(
        out_ptr + out_offs,
        out,
        mask=(gl.expand_dims(head_offs_out, axis=0) < HEAD_SZ),
    )


@gluon.jit
def _pa_decode_reduce_kernel_256k(
    out_ptr,
    exp_sums_ptr,
    max_logits_ptr,
    logits_ptrs,
    seq_lens_ptr,
    stride_o_s,
    stride_o_h,
    stride_exp_sums_s,
    stride_exp_sums_h,
    stride_exp_sums_p,
    stride_logits_s,
    stride_logits_h,
    stride_logits_p,
    stride_logits_g,
    HEAD_SZ: gl.constexpr,
    QUERY_GRP_SZ: gl.constexpr,
    SEQ_PARTITION_SZ: gl.constexpr,
    MAX_NUM_SEQ_PARTITIONS_POW2: gl.constexpr,
):
    blocked_meta: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[2, 4],
        threads_per_warp=[64, 1],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    slice_meta_part: gl.constexpr = gl.SliceLayout(1, blocked_meta)

    blocked_tile: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[32, 16],
        threads_per_warp=[16, 4],
        warps_per_cta=[1, 4],
        order=[1, 0],
    )
    slice_tile_part: gl.constexpr = gl.SliceLayout(1, blocked_tile)
    slice_tile_head: gl.constexpr = gl.SliceLayout(0, blocked_tile)

    seq_idx = gl.program_id(0)
    q_head_idx = gl.program_id(1)

    kv_head_idx = q_head_idx // QUERY_GRP_SZ
    grp_idx = q_head_idx % QUERY_GRP_SZ

    seq_len = gl.load(seq_lens_ptr + seq_idx)
    log2_sz: gl.constexpr = 9
    num_partitions = (seq_len + SEQ_PARTITION_SZ - 1) >> log2_sz

    ml_base = seq_idx * stride_exp_sums_s + kv_head_idx * stride_exp_sums_h + grp_idx
    logits_base = (
        seq_idx * stride_logits_s
        + kv_head_idx * stride_logits_h
        + grp_idx * stride_logits_g
    )

    meta_part_offs = gl.arange(0, MAX_NUM_SEQ_PARTITIONS_POW2, layout=slice_meta_part)
    meta_mask = meta_part_offs < num_partitions
    ml_offs = ml_base + meta_part_offs * stride_exp_sums_p

    max_logits = gl.amd.cdna3.buffer_load(ptr=max_logits_ptr, offsets=ml_offs, mask=meta_mask)
    exp_sums = gl.amd.cdna3.buffer_load(ptr=exp_sums_ptr, offsets=ml_offs, mask=meta_mask)

    ml = gl.max(max_logits, axis=0)
    LOG2E: gl.constexpr = 1.4426950408889634
    exp_sums = exp_sums * tl.math.exp2((max_logits - ml) * LOG2E)
    exp_sum = gl.sum(exp_sums, axis=0)
    p_meta = exp_sums / exp_sum

    tile_part_offs = gl.arange(0, MAX_NUM_SEQ_PARTITIONS_POW2, layout=slice_tile_part)
    head_offs = gl.arange(0, HEAD_SZ, layout=slice_tile_head)
    tile_mask = gl.expand_dims(tile_part_offs < num_partitions, axis=1)

    tile_offs = (
        logits_base
        + gl.expand_dims(tile_part_offs, axis=1) * stride_logits_p
        + gl.expand_dims(head_offs, axis=0)
    )
    tile = gl.load(logits_ptrs + tile_offs, mask=tile_mask)

    p_tile = gl.convert_layout(p_meta, slice_tile_part)

    weighted = tile.to(gl.float32) * gl.expand_dims(p_tile, axis=1)
    result = gl.sum(weighted, axis=0)

    out = result.to(out_ptr.dtype.element_ty)
    o_offs = seq_idx * stride_o_s + q_head_idx * stride_o_h + head_offs
    gl.store(out_ptr + o_offs, out, mask=head_offs < HEAD_SZ)
