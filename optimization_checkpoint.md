md 文件：# 实验 Checkpoint: SGLang Qwen3.5-9B Aiter Attention 精度回归定位

## 环境
- 时间: 2026-04-16 06:20:58 UTC
- GPU: 待补充（用户当前描述为 ROCm/AMD 环境，TP=2）
- 模型: Qwen3.5-9B, TP=2
- 框架: SGLang（Python 模块路径: `/opt/sglang/python/sglang`）
- 模型路径: `/data/models/Qwen3.5-9B/`
- 关键环境变量: `SGLANG_USE_AITER_NEW_CA=false`, `SGLANG_USE_AITER=1`, `HSA_NO_SCRATCH_RECLAIM=1`, `SGLANG_DISABLE_CUDNN_CHECK=1`

## 用户提供的标准命令
### 服务启动命令（基线正确）
```bash
SGLANG_USE_AITER_NEW_CA=false SGLANG_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1 SGLANG_DISABLE_CUDNN_CHECK=1 python3.10 -m sglang.launch_server --port 8009 --model-path /data/models/Qwen3.5-9B/ --tp-size 2 --mem-fraction-static 0.4 --max-running-requests 128 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --attention-backend triton --disable-radix-cache --disable-custom-all-reduce --page-size 1
```

### 服务启动命令（坏 case 1）
```bash
SGLANG_USE_AITER_NEW_CA=false SGLANG_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1 SGLANG_DISABLE_CUDNN_CHECK=1 python3.10 -m sglang.launch_server --port 8009 --model-path /data/models/Qwen3.5-9B/ --tp-size 2 --mem-fraction-static 0.4 --max-running-requests 128 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --attention-backend aiter --disable-radix-cache --disable-custom-all-reduce
```

### 服务启动命令（坏 case 2）
```bash
SGLANG_USE_AITER_NEW_CA=false SGLANG_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1 SGLANG_DISABLE_CUDNN_CHECK=1 python3.10 -m sglang.launch_server --port 8009 --model-path /data/models/Qwen3.5-9B/ --tp-size 2 --mem-fraction-static 0.4 --max-running-requests 128 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --attention-backend aiter --page-size 1 --disable-radix-cache --disable-custom-all-reduce
```

### 精度用例
```bash
cd ~ && curl -X POST http://localhost:8009/v1/chat/completions -H "Content-Type: application/json" -d @req.json
```

## 基线数据
| 指标 | 值 | 测量条件 |
|------|---|---------|
| 输出正确性 | 正常 | `--attention-backend triton` |
| 输出正确性 | 异常，出现复读机 | `--attention-backend aiter` |
| 输出正确性 | 异常，出现复读机 | `--attention-backend aiter --page-size 1` |

## 问题定义
- 用户已确认：`triton` 基线无精度问题。
- 用户已确认：`aiter` 路径存在精度问题，表现为模型重复输出同一段话。
- 用户已确认：`aiter + page-size 1` 走另一条 aiter 的 triton paged attention 路径，仍然出现同类复读机问题。
- 当前初步判断：问题更像是 `aiter backend` 共享的 decode/KV cache 语义错误，而不是单一 HIP kernel 独有问题。

## 代码路径确认
- Attention backend 实现文件: `/opt/sglang/python/sglang/srt/layers/attention/aiter_backend.py`
- Triton backend 对照文件: `/opt/sglang/python/sglang/srt/layers/attention/triton_backend.py`
- ROCm block KV cache: `/opt/sglang/python/sglang/srt/mem_cache/memory_pool.py`
- KV pool 初始化分流: `/opt/sglang/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
- 本地参考文档:
  - `/root/inference-engine-wiki/common/optimization_playbook.md`
  - `/root/inference-engine-wiki/sglang/troubleshooting/attention_backend.md`

## 实验记录
### 实验 0: 当前用户重启实例复现确认（原始坏服务）
- 状态: 已完成
- 服务:
  - 进程 cmdline 确认当前用户手动重启实例为 `--attention-backend aiter --port 8009 --disable-radix-cache --disable-custom-all-reduce`
  - 未显式关闭 cudagraph，实际为 `disable_cuda_graph=False`
  - `page_size=16`
- 结果:
  - 对当前 `localhost:8009` 连续发起 5 次同一 `req.json`
  - 5/5 都返回同一个坏结果：
    - `finish_reason="length"`
    - `content=null`
    - `reasoning_content` 长度固定为 `5055`
    - `reasoning_content` 哈希固定为 `eb443166`
  - 坏输出表现为 reasoning 模板段落循环复读，和用户给出的现象一致
- 结论:
  - 当前实例上的问题不是“低概率偶现”，而是进入某种稳定坏状态后可以 100% 复现
  - 问题发生在生成轨迹本身，不是 parser/后处理把正常答案弄坏

### 实验 1: 代码路径静态对比
- 状态: 进行中
- 改动: 无
- 结果:
  - `page_size > 1` 时，`aiter` 使用 `MHARocmBlockKVPool`，decode 走 `paged_attention_rocm`（HIP）或 `paged_attention_decode`（当 `SGLANG_USE_GLUON_PA=1` 时）。
  - `page_size = 1` 时，不走 `MHARocmBlockKVPool`，decode 回落到 `paged_attention_ragged` 或其他 `aiter` 非 block-KV 路径。
  - 两个坏 case 最终都落在 `aiter_backend.forward_decode()` 的不同分支，但共享同一个 backend 的元数据构造和部分 decode 语义。
  - 发现首个高可疑差异：`aiter_backend.forward_decode()` 的 block-KV decode 分支未使用 `layer.scaling` / `layer.logit_cap`，而是使用 backend 级固定 `self.scale = 1 / sqrt(head_dim)` 与 `self.k_scale = self.v_scale = 1`。
- 结论:
  - 该差异不足以单独定罪，但它是当前最明确的 `triton` vs `aiter` 行为分叉点之一。
  - 下一步应继续确认 Qwen3.5-9B 实际 layer 配置，并检查 `aiter` 分支是否还存在 KV 索引、cache 写入布局、GQA head 数传参等语义偏差。

### 实验 2: cudagraph 语义核对
- 状态: 已完成
- 改动:
  - `/opt/sglang/python/sglang/srt/layers/attention/aiter_backend.py`
- 结果:
  - 确认 `CudaGraphRunner` 在 capture 时会创建固定 shape 的输入 buffer，`seq_lens` 的 fill value 来自 `attn_backend.get_cuda_graph_seq_len_fill_value()`。
  - `AiterAttnBackend.get_cuda_graph_seq_len_fill_value()` 返回 `1`，因此 graph capture 阶段不能安全地把 decode partition 数直接绑定到 runtime `seq_len`。
  - `cuda graph` replay 前只会更新 tensor buffer；capture 时决定的 kernel launch/grid 和 Python 标量参数不会在 replay 时重新求值。
  - 因此，runtime partition 修复若直接作用于 graph capture/replay 路径，会有把 graph 错误固化成短序列版本的风险。
  - 已将修复收紧为：仅 non-graph decode 路径按 runtime `max seq len` 计算有效 partition 数；graph 路径仍保持 capture-time 最大值语义。
  - **同事已验证：关闭 `--disable-cuda-graph` 后 aiter 仍然复读，确认问题与 cudagraph 无关。**
- 结论:
  - cudagraph 已排除，问题在 `aiter` decode kernel 本身或 KV cache 语义层面。

### 实验 3: decode 前清零 PA workspace（graph 保持开启）
- 状态: 已完成
- 改动:
  - `/opt/sglang/python/sglang/srt/layers/attention/aiter_backend.py`
  - 新增 debug 开关 `SGLANG_AITER_ZERO_PA_WORKSPACE=1`
  - 行为：
    - block-KV decode (`paged_attention_rocm` / gluon paged attention) 前，对 `tmp_output` / `exp_sums` / `max_logits` 执行 `.zero_()`
    - ragged decode (`paged_attention_ragged`) 前，对 `workspace_buffer` 执行 `.zero_()`
- 对照服务:
  - 端口 `8011`
  - 启动参数与坏服务等价，保持 `attention_backend=aiter`、`page_size=16`、`disable_cuda_graph=False`
  - 唯一新增环境变量：`SGLANG_AITER_ZERO_PA_WORKSPACE=1`
- 结果:
  - `8011` 仍然随机出现复读现象
- 结论:
  不是 PA workspace导致的

### 实验 1 (续): 深度代码路径对比 — Python 层参数全量核对
- 状态: 已完成
- 方法:
  - 逐行对比 `aiter_backend.py` (2277行) 和 `triton_backend.py` (1333行) 的 forward_decode 路径
  - 核对 `RadixAttention` 层属性（scaling, logit_cap, k_scale, v_scale）
  - 核对 Qwen3.5-9B 模型配置（qwen3_5.py）
  - 核对 KV cache pool（MHARocmBlockKVPool, HybridMHATokenToKVPool）的 set/get_kv_buffer 实现
  - 核对 `_build_block_tables` block table 构建逻辑
  - 核对 `scatter_kv_cache` triton kernel 的 K/V 写入布局
- 结果:
  - Qwen3.5-9B: `layer.scaling` = `head_dim**-0.5` = `self.scale`，**数值相同**
  - Qwen3.5-9B: `layer.logit_cap` = 0.0（默认值，不启用 softcapping），**两个 backend 等价**
  - `k_scale/v_scale`: bf16 cache 下 `set_kv_buffer` 中 dtype 检查直接跳过，**不影响**
  - `kv_indptr/kv_indices` 构建逻辑完全相同
  - `HybridMHATokenToKVPool` 的 layer_id 重映射在 set/get 两侧一致
  - `_build_block_tables` 和 `scatter_kv_cache` 逻辑正确
- 结论:
  - **Python 层所有传参在 Qwen3.5-9B 配置下完全等价，不存在数值差异**
  - 问题不在 Python 层的参数传递，而在 aiter HIP kernel 本身或 extend/prefill 阶段
- 关键观察:
  - 两个坏 case（page_size=1 和 page_size=16）的 **decode kernel 完全不同**（`paged_attention_ragged` vs `paged_attention_rocm`）
  - 但两个坏 case 的 **extend kernel 是同一个** `mha_batch_prefill_func`
  - triton 基线的 extend 用的是 `extend_attention_fwd`（自研 triton kernel）
  - **高度怀疑问题在 extend (prefill) 阶段，而非 decode 阶段**

### 实验 4: 对比 triton vs aiter 的 extend hash (冷启动 A/B)
- 状态: **已完成**
- 方法:
  - 在 `aiter_backend.py` 和 `triton_backend.py` 的 `forward_extend` 返回前加 debug hook
  - 通过 `SGLANG_DEBUG_ATTN_HASH=1` 开关控制，用 `get_is_capture_mode()` 避免 graph capture 阶段触发
  - 自动化脚本 `/root/exp4_cold_start.sh`: 先跑 triton golden baseline，再循环 N 次 aiter 冷启动
  - 配置: `page_size=1`, `temperature=0`, `max_tokens=512`, 其余与用户标准命令一致
- 结果 (5/5 次全部 bad):
  - **Extend (prefill) 阶段 aiter 的输出已经与 triton 不同**
  - TP shard A (vals 起始 `-0.203125`): triton 和 aiter 前 8 值有小差异 (~0.001 级)
    - Triton: `-0.204102, 0.058105, -0.201172, -0.225586, -0.359375, -0.068359, -0.292969, 0.012878`
    - Aiter:  `-0.203125, 0.057861, -0.200195, -0.224609, -0.357422, -0.067871, -0.291016, 0.012817`
  - TP shard B (vals 起始 `0.029663`): **triton 和 aiter 前 8 值完全一致**
  - Aiter 每次冷启动的完整 tensor hash 都不同 (non-deterministic)，但前 8 个值相同
  - 5 次全部 `finish_reason=length` (复读)
- 结论:
  - **确认问题在 extend (prefill) 阶段，`mha_batch_prefill_func` 是根因所在**
  - 问题只影响一个 TP shard，另一个 shard 完全正确 — 可能与 head index 分片有关
  - kernel 内部存在非确定性行为 (hash 不稳定)
  - 下一步: 实验 5 混合 backend 进一步确认

### 实验 5: 替换 extend kernel — flash_attn_varlen_func 替代 mha_batch_prefill_func
- 状态: **已完成**
- 改动:
  - `/opt/sglang/python/sglang/srt/layers/attention/aiter_backend.py`
  - 新增环境变量 `SGLANG_AITER_USE_FA_EXTEND=1`
  - 行为: 在 ragged extend 路径 (非 MLA, page_size=1) 中，用 `flash_attn_varlen_func` (gather KV 后调用) 替代 `mha_batch_prefill_func`
  - `flash_attn_varlen_func` 和 `mha_batch_prefill_func` 都来自 `aiter` 包，前者是标准 flash attention varlen 实现
- 对照配置:
  - `--attention-backend aiter --page-size 1 --disable-cuda-graph`
  - 其余参数与标准命令一致
  - `temperature=0`, `max_tokens=2980`
- 结果:
  - **使用 `flash_attn_varlen_func`: 3/3 冷启动 + 每次 5 次请求 = 全部 `finish_reason=stop`，输出正确**
  - **使用 `mha_batch_prefill_func` (原始): 5/5 冷启动全部复读**
  - `flash_attn_varlen_func` 的 extend hash 在同一 server 实例内完全确定性 (deterministic)
  - `mha_batch_prefill_func` 的 extend hash 每次冷启动都不同 (non-deterministic)，虽然前 8 值一致
  - 两者的前 8 个输出值相同，差异在 tensor 其余位置 — 说明 `mha_batch_prefill_func` 在某些位置产生了垃圾值
- 结论:
  - **根因确认: `mha_batch_prefill_func` (aiter 的 batch prefill HIP kernel) 存在 bug**
  - bug 表现为 output tensor 中部分位置非确定性且错误，导致 prefill 结果污染后续 decode
  - `flash_attn_varlen_func` 可作为临时替代方案 (workaround)
  - 长期修复: 需要 aiter 团队排查 `mha_batch_prefill_func` 的实现问题

## 修复方案

### 当前状态
`flash_attn_varlen_func` 已经默认接入 `aiter_backend.py` 的两条 extend 路径（**无 env var 开关**）：
- 行 1681-1725: `use_rocm_block_kv` 分支 (page_size > 1，gather_kv_cache → flash_attn_varlen_func)
- 行 1726-1755: ragged 分支 (page_size = 1，直接 indexing K/V → flash_attn_varlen_func)

`SGLANG_AITER_USE_FA_EXTEND` env var 已移除，新代码完全替换掉 `mha_batch_prefill_func` 的使用。

### 长期修复
- 向 aiter 维护方报告 `mha_batch_prefill_func` 在特定配置下 (GQA, bf16, 长序列) 产生非确定性错误
- 需要排查该 kernel 的 shared memory 使用、reduction 逻辑、或 thread scheduling 相关问题

---

## 后续问题: 启用 radix cache + page_size=16 后第二个 req 命中 prefix 仍然复读
- 时间: 2026-04-17
- 触发条件: 去掉 `--disable-radix-cache`，保持 `--page-size 16`，连发 2 条相同 req.json，第 2 条命中 prefix 后输出复读
- 影响范围: 只在 prefix-cache 命中场景出现；首条 prefill (无 prefix) 用 `flash_attn_varlen_func` 已验证正常 (实验 5)；page_size=1 + disable-radix-cache 也正常

### 已审计的代码路径（静态分析未发现 bug）
| 检查点 | 文件:行 | 结论 |
|--------|---------|------|
| extend 主分支 | `/opt/sglang/python/sglang/srt/layers/attention/aiter_backend.py:1681-1725` | gather_kv_cache → flash_attn_varlen_func 调用顺序正确，set_kv_buffer 在前 (line 1333-1339) |
| metadata 构建 | `aiter_backend.py:1969-2033` (`update_single_wrapper`) | `kv_indices` 覆盖 prefix+new (用 `paged_kernel_lens=seq_lens`)，`qo_indptr` 仅 new tokens (用 `extend_lens=seq_lens-prefix_lens`)，`max_q_len`/`max_kv_len` 区分正确 |
| scatter/gather kernel | `/opt/sglang/python/sglang/srt/layers/attention/triton_ops/rocm_kv_cache.py` | K (x-interleaved [num_blocks, kv_heads, head_dim//X, page_size, X]) 和 V (transposed [num_blocks, kv_heads, head_dim, page_size]) 读写 offset 完全对称 |
| HybridLinearKVPool layer_id | `/opt/sglang/python/sglang/srt/mem_cache/memory_pool.py:1420-1437, 1454-1481` | set/get 都通过 `_transfer_full_attention_id` 一致映射到 full_kv_pool (MHARocmBlockKVPool) |
| causal mask | `/opt/aiter/aiter/ops/mha.py:2543-2592` | `flash_attn_varlen_func` 明确支持 q_len < kv_len 的 bottom-right causal (docstring 第 2582-2592 行举例确认) |
| 分配器 last_loc | `/opt/sglang/python/sglang/srt/mem_cache/allocator.py:407-410` | radix prefix 总是 page-aligned (`_page_align_keys`)，`(last_loc + 1) % page_size == prefix_lens % page_size` 保证新分配从下一 page 边界开始 |
| 自身 use_rocm_block_kv 探测 | `aiter_backend.py:217-223` | 正确识别 HybridLinearKVPool 的 `full_kv_pool` 并解包到 MHARocmBlockKVPool |
| int32 offset 限制 | 内存池估算 | 当前 mem_fraction=0.4 下 max_total_tokens 远低于 4,194,288 安全上限，**不是 int32 溢出** |

### 仍需 runtime 验证的可疑点
1. **`flash_attn_varlen_func` 在 q_len ≪ kv_len 数值正确性**: 静态看 docstring OK，但需要 runtime 对比同 K/V 在 mha_batch_prefill_func vs flash_attn_varlen_func 的输出，特别是 q_len=4, kv_len=96+ 的场景
2. **scatter→gather 同 stream 顺序**: 两次连续 triton kernel launch 在 default stream 上应序列化，但值得 runtime 加 `torch.cuda.synchronize` 确认是否有 race
3. **v_gathered shape**: aiter_backend.py:1707 用 `layer.head_dim` 而非 `v_head_dim` 分配 v_gathered (Qwen3.5 二者等于 256，但形状校验可加 assert)

### 计划好的下一步排查方向（待优先级排序）
- **方向 A (推荐)**: 用 `--page-size 16 --max-total-tokens 4194288` + 启用 radix cache 起服务，在 use_rocm_block_kv extend 分支加临时 debug print（gather 前后 K/V hash、qo_indptr/kv_indptr、prefix_len），连发 2 个相同 req 对比第 2 个 (带 prefix) 的中间值
- **方向 B**: 写 `flash_attn_varlen_func` 单元测试，直接喂相同 K/V 给 mha_batch_prefill_func vs flash_attn_varlen_func，对比 q_len ≪ kv_len 场景输出
- **方向 C**: 临时回退到 mha_batch_prefill_func + 启用 radix cache，验证假设 "只是 flash_attn_varlen_func 与 prefix-cache 不兼容"

---

## 后续问题: page_size=1 + radix-cache 启用后 prefix-cache 完全 miss
- 时间: 2026-04-17
- 触发条件: `--page-size 1 --attention-backend aiter`（去掉 `--disable-radix-cache`），`--mamba-scheduler-strategy` 默认 `no_buffer`，连发 2 条相同 req.json
- 现象: 两条请求日志均 `#cached-token: 0`，第 2 条没有命中 prefix
- 怀疑：是不是近 3 条 commit (`b18a0b5`/`f50b10769`/`0cfb18a80`) 引入

### 实验 6: 静态对比 + runtime debug 定位 (排除 commit 嫌疑)
- 状态: **已完成**
- 静态对比 `b748da8af` (parent) 和 HEAD 的 `mamba_radix_cache.py`：
  - `_match_prefix_helper` 和 `_split_node` **完全一致**
  - `b18a0b5` 新增的 `if cache_len > page_aligned_len` / 截断逻辑均在 `if self.page_size != 1:` 分支内，对 page_size=1 是 no-op
  - `model_runner_kv_cache_mixin.py` / `memory_pool.py` 新增的 `MHARocmBlockKVPool` 也只走 `page_size > 1` 分支
- runtime 加 debug 打印（gated by `SGLANG_DBG_MAMBA_RADIX=1`）：在 `match_prefix` / `_match_prefix_helper` / `cache_finished_req` 入口出口打印 key_len / value / best_value_len / mamba_value 状态
- 关键 log（第 2 条 req 的 match）：
  ```
  iter1: root → node7(1, mamba=None), prefix_len=1, ELSE
  iter2: node7 → node8(15733, mamba=set), prefix_len=15732 < 15733 → SPLIT!
         新建 node10(15732, mamba=None), node8 变成 node10 的 child(1 token)
  FINAL: best_value_len=0
  ```
- 结论: **不是 3 条 commit 的锅**，是 MambaRadixCache v1 模式 + `max_prefix_len = input_len - 1` 的 off-by-one 互动导致的**预先存在的设计限制**

### 根因分析
- `_split_node` 注释明确写 "mamba cache can not be split"，强行把 split 出来的 parent node 设为 `mamba=None` tombstone
- `_match_prefix_helper` 末尾 `if node.mamba_value is not None` 检查只看当前 node，split 后 node = new_node (None) → `best_value_len` 保持 0
- v1 模式下，每条 req 的 mamba state 只在 `len(fill_ids)` 一个位置 cache（cache_unfinished_req）+ 全长位置 cache（cache_finished_req）
- 第 2 条相同请求的 match key = `input_len - 1`，比 cache 位置短 1 → 必触发 split → tombstone → `#cached-token: 0`
- **对 prefix sharing 同样灾难**：第一次 split 后 shared prefix 节点变成永久 tombstone，后续所有相同前缀请求都 miss

### 为什么 page_size=16 之前能命中 (但有精度问题)
- `_key_match_paged` 只返回 `page_size` 倍数的 prefix_len
- `cache_unfinished_req` 在 `page_size != 1` 分支已经把 cache 截断到 page 边界（`b18a0b5` 加的）
- 例如 input=15734，cached at 15728 (`15734 // 16 * 16`)
- match key=15733 (max_prefix_len) 时：`prefix_len = key_match_paged(node15728.key, 15733) = 15728`（多余的 5 token 不够一页，直接停在 15728）
- `15728 == len(child.key)` → 不 split → ELSE 分支 → loop 退出
- 退出后 `if node15728.mamba_value is not None` → True → `best_value_len = 1` → **HIT 15728 token**
- 但是 cached mamba_value 实际是 `state_at_15734`（位置不一致），命中后会引入 mamba state 错位

### 实验 7: 尝试 truncate-by-1 fix 验证 mamba 错位影响
- 状态: **已完成（fix 已回退）**
- 改动：`mamba_radix_cache.py` 的 `cache_unfinished_req` 在 page_size=1 分支截掉最后 1 token（gated by `SGLANG_MAMBA_RADIX_RESERVE_LAST=1`）
- 测试结果：
  | 指标 | R1 (无 cache) | R2 (with fix, 15733 cache hit) |
  |------|---|---|
  | `#cached-token` | 0 | 15733 ✓ |
  | `finish_reason` | stop | **length** ✗ |
  | `completion_tokens` | 375 | **2980 (max)** |
  | 内容 | 正常 | **复读机** |
  | 时间 | 3.7s | 21s（卡 max） |
- 结论：mamba state 错位 1 token 不被模型容忍，直接导致复读机
- **重要推论**: 之前 page_size=16 + radix-cache 看到的复读机不只是 `mha_batch_prefill_func` 的锅，**mamba state 错位（1~15 token）也是直接原因**。即使把 `flash_attn_varlen_func` 接到 use_rocm_block_kv extend 分支上，page_size=16 + radix-cache 仍然会复读

### 实验 8: 启用 mamba `extra_buffer` 模式
- 状态: **已完成 ✓**
- 背景：v1 mode (`no_buffer`) 只在请求结束时 cache 单一位置的 mamba state；`extra_buffer` 模式用 ping-pong buffer 在每 `mamba_track_interval` (默认 256) 边界保存中间 mamba state，配合 FLA chunk 的 `h` 输出
- 改动：只改 `server_args.py` 一处 assertion（原本硬限 CUDA-only）：
  ```python
  -                is_cuda()
  +                is_cuda() or is_hip()
  ```
- ROCm 上 FLA backend 实际可用：`hybrid_linear_attn_backend.py` 的 import 是 `if not is_cpu() and not is_npu()`，HIP 走这条路径；`chunk_gated_delta_rule` 在 ROCm 上有 `_is_hip` 优化路径
- 启动命令额外加 `--mamba-scheduler-strategy extra_buffer`
- 测试结果：
  ```
  R1 (cold): #new-token: 15734, #cached-token: 0
  R2 (warm): #new-token: 54,    #cached-token: 15680
  ```
  - 15680 = `15734 // 64 * 64`，FLA_CHUNK_SIZE=64 对齐（page_size=1 时 mamba_cache_chunk_size = max(64, 1) = 64）
  - **命中 99.66% prefix**
  | 指标 | R1 | R2 |
  |---|---|---|
  | finish_reason | stop ✓ | stop ✓ |
  | completion_tokens | 421 | 407 |
  | 复读机 | 无 | **无** ✓ |
  | 输出内容 | 合理客服回复 | 合理客服回复（不同表述） |
  | 时间 | 3.83s | 2.88s |
- R1/R2 输出非 bit-exact（top_k=1 应 deterministic）：mamba state 在 chunk 边界 (15680) 的快照 vs 全量 prefill 到 15680 的累加结果有微小 bf16 误差，是 SSM 模型 + prefix cache 的固有现象

### 修复方案

**最终推荐配置 (page_size=1 + ROCm + radix-cache + 命中且无复读)：**
```bash
AITER_QUICK_REDUCE_CAST_BF16_TO_FP16=0 AITER_QUICK_REDUCE_QUANTIZATION=FP \
SGLANG_ROCM_USE_AITER_LINEAR_SHUFFLE=1 SGLANG_ROCM_USE_AITER_LINEAR_FP8HIPB=1 \
SGLANG_USE_AITER_NEW_CA=false SGLANG_USE_CUDA_IPC_TRANSPORT=1 \
SGLANG_VLM_CACHE_SIZE_MB=8192 SGLANG_USE_AITER=1 \
HSA_NO_SCRATCH_RECLAIM=1 SGLANG_DISABLE_CUDNN_CHECK=1 \
python3.10 -m sglang.launch_server --port 8009 \
  --model-path /root/Qwen3.5-9B/ --tp-size 2 \
  --mem-fraction-static 0.4 --max-running-requests 1024 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --attention-backend aiter --page-size 1 \
  --context-length 65000 --disable-custom-all-reduce \
  --mamba-scheduler-strategy extra_buffer
```

注：`flash_attn_varlen_func` 已默认接入两条 extend 路径，不再需要 `SGLANG_AITER_USE_FA_EXTEND` env var。page_size=16 + extra_buffer + radix-cache 理论上也可用（mha_batch_prefill bug + mamba 错位都已解决），但本次没有 runtime 验证。

### 代码改动文件
- `/opt/sglang/python/sglang/srt/server_args.py`
  - 行 1730: `is_cuda()` → `is_cuda() or is_hip()` (放开 ROCm 上 extra_buffer 的硬阻挡)

### 设计层面遗留问题
- v1 (`no_buffer`) 模式 + radix-cache 在数学上不可能正确：mamba state 位置敏感，cache 位置 ≤ `len(fill_ids)` 必然有错位
- v1 模式 + radix-cache 的常见踩坑场景：
  - 重复发送相同请求（必 miss，命中也错）
  - Prefix sharing 高并发（第一次 split 后 shared prefix 永久 tombstone）
  - Batch eval / 重复 prompt 测试（接近完全失效）
- v1 模式还能用的场景：多轮对话累加（match key 远大于任何 cache 位置，自然走 ELSE 分支不 split）
- 长期：v1 (`no_buffer`) 模式应该被弃用 / warning 提醒用户切到 `extra_buffer`

