"""
Gluon-based paged attention decode API.
Derived from atrex/python/atrex/api/paged_attention_decode.py
"""
import torch
import triton
from sglang.srt.layers.attention.triton_ops.pa_decode_gluon import (
    _pa_decode_dot_kernel,
    _pa_decode_reduce_kernel_64k,
    _pa_decode_reduce_kernel_256k,
)

_SEQ_PARTITION_SIZE = 512  # HIP


def paged_attention_decode(
    output: torch.Tensor,
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,   # [num_blocks, num_kv_heads, head_size/x, block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, block_size]
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    attn_scale: float,
    max_seq_len: int,
    alibi_slopes: torch.Tensor = None,
) -> None:

    num_kv_heads = key_cache.shape[1]

    max_num_partitions = int((max_seq_len + _SEQ_PARTITION_SIZE - 1) // _SEQ_PARTITION_SIZE)
    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    kv_blk_sz = value_cache.shape[3]
    head_sz = query.shape[2]
    query_grp_sz = num_q_heads // num_kv_heads
    query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)

    kv_blk_sz_pow2 = triton.next_power_of_2(kv_blk_sz)
    head_sz_pow2 = triton.next_power_of_2(head_sz)
    if head_sz_pow2 != head_sz:
        raise TypeError("only support headsize in power of 2!")
    use_alibi_slopes = (alibi_slopes is not None)
    # MHA
    if query_grp_sz == 1:
        raise TypeError("MHA with transposed kvcache not supported!")
    # GQA
    else:
        grid = (num_seqs, num_kv_heads, max_num_partitions)
        if query_grp_sz <= 16:
            query_grp_sz_pow2 = 16
        else:
            query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)
        # in that case temp_output = output, so we directly write to output
        if (max_num_partitions == 1):
            _pa_decode_dot_kernel[grid](
                exp_sums,
                max_logits,
                output,
                query,
                key_cache,
                value_cache,
                block_tables,
                seq_lens,
                attn_scale,
                alibi_slopes,
                exp_sums.stride(0),
                exp_sums.stride(1),
                exp_sums.stride(2),
                tmp_output.stride(0),
                tmp_output.stride(1),
                tmp_output.stride(2),
                tmp_output.stride(3),
                query.stride(0),
                query.stride(1),
                key_cache.stride(0),
                key_cache.stride(1),
                key_cache.stride(2),
                key_cache.stride(3),
                value_cache.stride(0),
                value_cache.stride(1),
                value_cache.stride(2),
                block_tables.stride(0),
                USE_ALIBI_SLOPES=use_alibi_slopes,
                HEAD_SZ=head_sz,
                QUERY_GRP_SZ=query_grp_sz,
                QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
                KV_BLK_SZ=kv_blk_sz,
                KV_BLK_SZ_POW2=kv_blk_sz_pow2,
                SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
            )
            return
        else:
            _pa_decode_dot_kernel[grid](
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                block_tables,
                seq_lens,
                attn_scale,
                alibi_slopes,
                exp_sums.stride(0),
                exp_sums.stride(1),
                exp_sums.stride(2),
                tmp_output.stride(0),
                tmp_output.stride(1),
                tmp_output.stride(2),
                tmp_output.stride(3),
                query.stride(0),
                query.stride(1),
                key_cache.stride(0),
                key_cache.stride(1),
                key_cache.stride(2),
                key_cache.stride(3),
                value_cache.stride(0),
                value_cache.stride(1),
                value_cache.stride(2),
                block_tables.stride(0),
                USE_ALIBI_SLOPES=use_alibi_slopes,
                HEAD_SZ=head_sz,
                QUERY_GRP_SZ=query_grp_sz,
                QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
                KV_BLK_SZ=kv_blk_sz,
                KV_BLK_SZ_POW2=kv_blk_sz_pow2,
                SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
            )
            num_q_heads = num_kv_heads * query_grp_sz
            grid = (num_seqs, num_q_heads, 1)
            if(max_seq_len == 256 * 1024):
                _pa_decode_reduce_kernel_256k[grid](
                            output,
                            exp_sums,
                            max_logits,
                            tmp_output,
                            seq_lens,
                            output.stride(0),
                            output.stride(1),
                            exp_sums.stride(0),
                            exp_sums.stride(1),
                            exp_sums.stride(2),
                            tmp_output.stride(0),
                            tmp_output.stride(1),
                            tmp_output.stride(2),
                            tmp_output.stride(3),
                            HEAD_SZ=head_sz,
                            QUERY_GRP_SZ=query_grp_sz,
                            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
                            MAX_NUM_SEQ_PARTITIONS_POW2=int(triton.next_power_of_2(max_num_partitions)),
                        )
            else:
                _pa_decode_reduce_kernel_64k[grid](
                            output,
                            exp_sums,
                            max_logits,
                            tmp_output,
                            seq_lens,
                            output.stride(0),
                            output.stride(1),
                            exp_sums.stride(0),
                            exp_sums.stride(1),
                            exp_sums.stride(2),
                            tmp_output.stride(0),
                            tmp_output.stride(1),
                            tmp_output.stride(2),
                            tmp_output.stride(3),
                            HEAD_SZ=head_sz,
                            QUERY_GRP_SZ=query_grp_sz,
                            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
                            MAX_NUM_SEQ_PARTITIONS_POW2=int(triton.next_power_of_2(max_num_partitions)),
                        )
