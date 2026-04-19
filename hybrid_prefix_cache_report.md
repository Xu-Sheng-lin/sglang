# Hybrid 模型前缀缓存：RadixAttention、PagedAttention、Mamba State 三方交互原理与实验报告

**场景**：Qwen3.5-9B (hybrid SSM + self-attention) on SGLang + ROCm (MI300X)
**日期**：2026-04
**范围**：前缀缓存机制的理论推导、踩坑案例、修复方案、实验验证

---

## 摘要

SGLang 的 RadixAttention 前缀缓存和 vLLM 的 PagedAttention 是**两个正交的概念**：前者是前缀匹配策略，后者是 KV cache 物理布局。SGLang 内部同时用两者。

对于 Qwen3.5 这类混合 SSM + self-attention 模型（8 层 full-attn + 24 层 GDN/Mamba），前缀缓存的可行性**由 SSM 层的状态管理决定**，与 KV cache 布局无关。换成 PagedAttention 不会放松任何限制。

Mamba/SSM state 具有**位置敏感性**：状态 $h_t$ 依赖于完整历史 $x_1 \dots x_t$，不能在任意位置"切片"或"反推"。因此前缀缓存只能在**预先保存了 state 快照的位置**命中。

SGLang 提供两种 SSM 状态管理策略：
- **`no_buffer`**（默认）：只在 prefill 结束和 decode 结束两个位置保存 $h$
- **`extra_buffer`**：每 `FLA_CHUNK_SIZE=64` token 通过 ping-pong buffer 保存一次 $h$

本报告通过实验和源码分析证明：
1. **重复发送相同请求时**，`no_buffer` 必然 miss 或命中后复读
2. **多轮对话场景**，`no_buffer` 能正常命中（match key 天然长于 cached 锚点）
3. **`extra_buffer` 是混合模型高命中率 + 正确输出的唯一可行方案**

---

## 1. 术语澄清：RadixAttention ≠ PagedAttention

两者经常一起出现，但关注点完全不同：

| 维度 | PagedAttention | RadixAttention |
|---|---|---|
| **关注点** | KV cache 怎么**物理存储** | 前缀 KV 怎么被**找到并复用** |
| **核心数据结构** | 分页内存 + block table | 压缩 trie（radix tree） |
| **粒度** | 固定大小 page（如 16 tokens） | 变长 token 序列 |
| **解决问题** | 内存碎片化、CoW 共享 | 前缀重复计算 |
| **出处** | vLLM, Kwon et al., SOSP 2023 | SGLang, Zheng et al., NeurIPS 2024 |

**SGLang 同时使用两者**：radix tree 作为前缀匹配索引，paged KV cache 作为物理存储。`page_size` 参数控制分页粒度，`page_size=1` 退化为非分页（每 token 一 slot）。

---

## 2. PagedAttention 原理

### 2.1 问题背景

传统做法：每个 request 预分配连续的 KV cache（$\text{max\_seq\_len} \times L \times H \times d$）。问题：
- 实际长度 < max_seq_len 导致内存浪费
- 共同前缀的多个 request 各存一份
- GPU 内存利用率仅 20–40%

### 2.2 核心设计

将 KV cache 切成固定大小的 **page**（典型 page_size=16 tokens）。每个 request 维护一个 **block table**：

```
Request A 的 block_table:  [3, 7, 12, 5]      ← 逻辑位置 → 物理 page 号
Request B 的 block_table:  [3, 7, 12, 8]      ← 前 3 page 与 A 共享
```

物理 KV cache：
```
page 0   page 1   page 2   page 3   page 4   page 5 ...
[16 tok][16 tok][16 tok][16 tok][16 tok][16 tok]
```

### 2.3 Attention Kernel

PagedAttention kernel 对每个 query 位置：
1. 通过 block_table 查到该位置所在的物理 page 号
2. 从物理 page 读 K/V
3. 计算 attention score 与加权求和

KV 在**物理上不连续**但逻辑上线性。

### 2.4 优势

- 零内存碎片化
- 共享前缀零成本（多个 block_table 指向相同 pages；CoW 按写入 fork）
- GPU 内存利用率 ~90%+

---

## 3. RadixAttention 原理

### 3.1 核心思想

**Radix tree** 是压缩版 trie：单子节点链合并。用它索引"哪些 token 序列有对应的 KV cache"。

```
                  root
                 /    \
           [system:] [Human:]
            /    \        \
      [你是...]  [分析...]  [介绍...]
          |          |           |
     kv_block#12 kv_block#34 kv_block#56
```

### 3.2 前缀匹配流程

新 request 到来时：
1. 把 prompt token_ids 作为 key 在 radix tree 里**最长前缀匹配**
2. 匹配上的部分对应的 KV cache **直接复用**（不 recompute）
3. 未匹配的后缀走正常 prefill
4. Prefill 结束后把完整序列 insert 回 tree（新 suffix 接到匹配点）

### 3.3 Split 语义

若 request 匹配到某个节点的前半段，需要 **split** 该节点：父节点持有公共前缀，两个子节点分别持有两个分支的尾部。

```
Before split:
  parent → node(key="AB C D E F", value=[1,2,3,4,5,6])

New request with prefix "AB C D X":
  parent → new_parent(key="AB C D", value=[1,2,3,4])
             ├→ old_node(key="E F", value=[5,6])
             └→ new_leaf(key="X", value=[...])
```

### 3.4 KV Cache 复用的数学基础

对**纯 self-attention 层**，token $i$ 的 $K_i, V_i$ 只依赖 token_id[i] 和当前层权重：
$$K_i = W_K \cdot E[token\_id_i] + \text{pos\_enc}_i$$

两个 request 前 $N$ 个 token 相同 ⟹ 前 $N$ 个 $K_i, V_i$ **严格相等**。

所以 radix tree 上 "相同 token 序列 → 可复用 KV cache" 的语义成立。

### 3.5 与 PagedAttention 的配合

SGLang 的 radix tree 节点存的是 **page 索引列表**。命中后：
1. Radix tree 返回匹配到的 pages
2. 设置 request 的 block_table 指向这些 pages
3. Attention kernel 走 PagedAttention 访问

---

## 4. Qwen3.5 混合模型架构

### 4.1 层结构

```
32 层，pattern: (3 linear + 1 full_attention) × 8
```
展开：
```
layer 0-2:  linear (GDN)
layer 3:    full_attention
layer 4-6:  linear
layer 7:    full_attention
...
layer 28-30: linear
layer 31:    full_attention
```
共 **8 层 full-attn + 24 层 linear (Gated Delta Net)**。

### 4.2 两种层的状态

| 层类型 | 状态形式 | 位置特性 |
|---|---|---|
| **full_attention** | KV cache `[seq_len, heads, head_dim]` | **per-token**，可按位置索引 |
| **linear (GDN)** | SSM state $h \in \mathbb{R}^{H \times K \times V}$ + conv state | **整体**，每次前向推进 1 步 |

### 4.3 GDN 状态演化

简化的递推关系：
$$h_t = g_t \odot (I - \beta_t k_t k_t^T) h_{t-1} + \beta_t v_t k_t^T$$
$$o_t = q_t h_t$$

**关键性质**：
- $h_t$ 依赖完整历史 $x_1 \dots x_t$
- SSM state 不能按 token 切片
- 递推通常不可逆（无法从 $h_t$ 反推 $h_{t-1}$）

---

## 5. 混合模型前缀缓存的核心挑战

### 5.1 Full-attention 层：没问题

两个相同前缀的 request 在 full_attn 层的 KV cache 严格相同，可按 radix tree 粒度任意切片复用。

### 5.2 Linear (GDN/Mamba) 层：位置敏感

Request 1 prefill 到 15734 tokens，GDN 层 state 到达 $h_{15734}$。

Request 2 想复用前 15680 tokens 的缓存：
- Full-attn 层：取 $KV[0:15680]$，正确
- GDN 层：需要 $h_{15680}$ 快照 → **只有 $h_{15734}$，不能用**

原因：
1. 不能从 $h_{15734}$ "倒推"到 $h_{15680}$（递推不可逆）
2. 不能取 $h_{15734}$ 当作 $h_{15680}$（位置错位，数学上就错了）

### 5.3 唯一可行方案

要在位置 $P$ 复用 prefix，必须**预先在位置 $P$ 保存了 $h_P$ 的快照**。这就是为什么 mamba state 管理策略至关重要。

---

## 6. `no_buffer` vs `extra_buffer`

### 6.1 `no_buffer` 策略

只在两个时机保存 $h$：

| 调用点 | 时机 | 保存位置 |
|---|---|---|
| `cache_unfinished_req` | prefill 刚结束 | `len(fill_ids)` |
| `cache_finished_req` | decode 结束 | `len(fill_ids + output_ids)` |

**Prefill 过程中**产生的中间 $h_i$（$i=1, \dots, len(fill_ids)-1$）**全部丢弃**。

### 6.2 `extra_buffer` 策略

利用 FLA chunk kernel 的"分块计算 + 中间状态可导出"特性：
- `chunk_gated_delta_rule(...)` 返回 `core_attn_out, last_recurrent_state, h`
- 其中 `h.shape = [1, num_chunks, H, K, V]`，包含每个 `FLA_CHUNK_SIZE=64` token 边界的 state（**临时**，仅在 kernel 调用期间存在）

**重要：FLA 计算粒度是 64，但持久化粒度不是。** 每次 prefill 调用结束后，`_track_mamba_state_extend` 只挑出 `h` 中**最后一个 chunk-aligned 位置**的 state 复制到 ping-pong slot，其他 244 个 chunk 边界的 state 随 `h` 释放。

`cache_unfinished_req` 插入时：
- `cache_len = req.mamba_last_track_seqlen`（=本次 prefill 末尾 chunk-aligned 位置，如 15680）
- `mamba_value = req.mamba_ping_pong_track_buffer[...]`（对应 15680 的 state）

**key 长度和 mamba_value 位置严格对齐。**

### 6.3 代价

`extra_buffer` 多占显存：每 req 多 2 个 mamba state slot。Qwen3.5-9B TP=2 下单 slot 约 24GB（ssm_state）+ 0.5GB（conv_state），ping-pong 翻倍。实测 `max_mamba_cache_size=1013` 减少约 15–30%。

---

## 7. 深度分析：`no_buffer` 下前缀缓存的命中规律

### 7.1 Match 逻辑回顾

`_match_prefix_helper` 的核心伪代码：

```python
def match(key):
    node = root
    best_value_len = 0
    while len(key) > 0 and key[0] in node.children:
        child = node.children[key[0]]
        if node.mamba_value is not None:      # ← 关键检查点 1
            best_value_len = len(value)
        prefix_len = longest_common_prefix(child.key, key)
        if prefix_len < len(child.key):
            split_parent = split(child, prefix_len)
            split_parent.mamba_value = None   # ← 关键：split tombstone
            value.append(split_parent.value)
            node = split_parent
            break
        else:
            value.append(child.value)
            node = child
            key = key[prefix_len:]
    if node.mamba_value is not None:          # ← 关键检查点 2
        best_value_len = len(value)
    return value, best_value_len
```

### 7.2 命中/未命中的判定条件

**关键：`best_value_len > 0` 的充要条件是 loop 过程中或结束时 `node.mamba_value is not None`。**

Split 出来的 parent 节点永远是 `mamba_value=None`，因此：
- **触发 split** ⟹ loop break 时 `node` 是 split parent（None）⟹ 检查点 2 失败
- **不触发 split**（走 ELSE 分支到底 / 自然退出 loop） ⟹ `node` 是真实锚点（可能 mamba=set）⟹ 检查点可能成功

### 7.3 三种场景的对比

设：cached 节点链的末端位置为 $E$，当前 request 的 match key 长度为 $K$（典型情况 $K = \text{input\_len} - 1$）。

| 情况 | 条件 | Split? | 命中? |
|---|---|---|---|
| A: 短请求 | $K < E$ | **是**（在末端节点中间停下） | ❌ tombstone |
| B: 刚好 | $K = E$ | 否（loop 末尾 `child.key_len == key_remain`） | ✓ |
| C: 长请求 | $K > E$ | 否（走完所有 cached nodes 后没匹配的 child，loop 自然退出） | ✓ |

### 7.4 场景 A：相同请求重复为什么踩坑

Request 1: input 长度 15734，output 374 → 结束后 tree 有：
```
root → node_A(key_len=15734, mamba@15734)
         └→ node_B(key_len=374, mamba@16108)
```

Request 2（相同 input）的 match key 长度 = $15734 - 1 = 15733$：
- iter 1: 进 node_A，prefix_len = 15733 **<** node_A.key_len = 15734 → SPLIT
- split 出 `new_parent(key_len=15733, mamba=None)` 和 `node_A_tail(key_len=1, mamba@15734)`
- loop break，node = new_parent（mamba=None）
- best_value_len = 0 ⟹ **miss**

**物理意义**：从 req 1 的 prefill 角度，$h_{15733}$ 的值**确实存在过**，但在 cache_unfinished_req 调用时只保存了 $h_{15734}$。Tree 里找不到位置 15733 的锚点。split 只能切 key 和 pointer，切不出一个原本就没保存的 $h_{15733}$ 张量。

### 7.5 场景 C：多轮对话为什么能命中

Turn 1: input = system + user1 (15000 tok)，output = assistant1 (200 tok) → tree 有：
```
root → node_A(key_len=15000, mamba@15000)        ← cache_unfinished_req 存的
         └→ node_B(key_len=200, mamba@15200)      ← cache_finished_req 存的
```

Turn 2: input = [system + user1 + assistant1 + user2] = 15700 tok，match key 长度 = 15699。

逐步执行：

```
iter 1:
  node = root (mamba=None)
  child = node_A (key_len=15000, mamba=set)
  检查 root.mamba → None，不更新 best
  prefix_len = 15000 (完全匹配), ELSE 分支
  value += node_A.value
  node ← node_A
  key_remain = 15699 - 15000 = 699

iter 2:
  node = node_A (mamba=set)     ← 注意：node_A 有 mamba！
  child = node_B (key_len=200, mamba=set)
  检查 node_A.mamba → SET ✓ → best_value_len=1, best_last_node=node_A
  prefix_len = 200 (完全匹配), ELSE 分支
  value += node_B.value
  node ← node_B
  key_remain = 699 - 200 = 499

iter 3 尝试:
  找 node_B 的 child 匹配 user2 的第一个 token
  → 不存在（user2 是新内容）
  → while loop 条件失败，自然退出

loop 外检查:
  node = node_B, mamba=set ✓ → best_value_len=2, best_last_node=node_B
```

**返回 best_value_len=2，命中 15200 token。**

### 7.6 为什么多轮对话天然满足条件 C

对多轮对话：
- 上一轮 cache 链末端位置 $E = \text{上一轮 input} + \text{上一轮 output}$
- 本轮 match key 长度 $K = \text{本轮 input} - 1 = E + \text{本轮新增 user reply} - 1$
- 只要本轮 user reply ≥ 2 token，就有 $K \geq E + 1 > E$，**自动满足场景 C**

对相同请求重复：
- 上一轮 cache 链末端位置 $E = \text{input\_len}$（cache_unfinished_req）或 $\text{input\_len} + \text{output\_len}$（cache_finished_req）
- 本轮 match key 长度 $K = \text{input\_len} - 1$
- $K < E$ 恒成立，**必然踩场景 A**

### 7.7 抽象规律

> **`no_buffer` 模式下，前缀缓存命中当且仅当 match key 能"穿过"所有 cached node 或者刚好对齐到锚点，否则 split tombstone 使 best_value_len=0。**

---

## 8. `extra_buffer` 如何解决

`extra_buffer` 的本质：**让 prefill / decode 末尾的锚点位置对齐到 chunk 边界，避免 split 时被强制 tombstone**。

- `no_buffer`：在请求 prefill 末尾（`len(fill_ids)`）和 decode 末尾（`len(fill_ids)+len(output_ids)`）埋锚点，**位置 = 当前精确长度**（如 15734、16108）
- `extra_buffer`：在请求 prefill 末尾**对齐到 chunk 边界**（如 15680）和 decode 末尾**对齐到 track interval**（如 15872、16128）

⚠️ **重要纠正：`extra_buffer` 不是"每 64 token 都埋一个锚点"**。FLA chunk kernel 计算粒度是 64 token，但**持久化粒度是按 prefill / decode 调用的边界**。详见 §8.5。

### 为什么对齐到 chunk 边界就能命中

Request 2 的 match key=15733 场景：
- Tree 里只有一个 prefill-end 锚点：node_A(15680, mamba@15680)
- match descent 进入 node_A，prefix_len=15680（完全匹配 node_A.key 长度）→ **ELSE 分支，不 split**
- 退出 loop 后检查 `node_A.mamba=set` → `best_value_len=1`
- 命中 15680 token + prefill 剩余 54 token

**命中率**：15680 / 15734 ≈ **99.66%**。

**输出正确性**：mamba_value = $h_{15680}$，key 长度也是 15680，位置严格对齐。Prefill 54 token 从 $h_{15680}$ 推到 $h_{15734}$，数学上无错位。

### 对比 `no_buffer` 的失败

`no_buffer` 下 cache 在 15734（精确 prefill 长度），match key=15733 进入 node 时 `prefix_len=15733 < 15734` → **必 split**，新 parent 强制 mamba=None tombstone → miss。`extra_buffer` 把 cache 位置从 15734 拉回到 15680，腾出 54 token 余量，避开 split。

---

## 8.5 `extra_buffer` 触发时机详解

`extra_buffer` 的 state 快照分两个阶段触发：

### 8.5.1 Extend (Prefill) 阶段：每次 prefill 调用 persist 1 个锚点

- **触发条件**（per prefill call，per req）：`req.extend_input_len >= mamba_cache_chunk_size`
  - `mamba_cache_chunk_size = max(FLA_CHUNK_SIZE=64, page_size)`，page_size=1/16 时都是 **64**
- **保存位置**（一次 prefill 调用持久化一个）：
  - `len(prefix_indices) + (extend_input_len // chunk_size) * chunk_size`
  - 15734 输入 + 0 prefix → **15680**
- **数据来源**：FLA 的 `chunk_gated_delta_rule(...)` 返回 `(core_attn_out, last_recurrent_state, h)`
  - `h` shape `[1, num_chunks, H, K, V]`，包含**所有 chunk 边界**的 state（如 245 个）
  - h tensor **临时**存在，per-layer 峰值 ~254 MB
- **落地 kernel**：`hybrid_linear_attn_backend.py:637 _track_mamba_state_extend` 只挑 `h[最后一个 chunk]` 复制到 ping-pong slot（24 MB），**其他 chunk 边界的 state 随 h 释放**
- **ping-pong 翻转**：`mamba_next_track_idx = 1 - mamba_next_track_idx`

⚠️ **关键事实：单次 prefill 调用只持久化 1 个锚点（最后一个 chunk-aligned 位置），不是 N 个**。

state 读取的 3 种情况（决定从 `h` 还是 `last_recurrent_state` 取）：

| 情况 | state 来源 |
|---|---|
| chunk-aligned **且**是序列末尾 | `last_recurrent_state` |
| chunk-aligned **但**不是末尾 | `h`（通过 `_force_track_h(i) = i+1` trick） |
| 非 chunk-aligned | `h` |

### 8.5.2 Decode 阶段：每 `mamba_track_interval` token persist 1 个锚点

- **触发条件**：`seq_len % mamba_track_interval == 0`（默认 `mamba_track_interval = 256`）
- **触发点**：`schedule_batch.py:2014 prepare_for_decode` 每次 decode forward 前检查
- **落地 kernel**：`hybrid_linear_attn_backend.py:607 _track_mamba_state_decode` → `track_mamba_states_if_needed`（Triton）：
  ```python
  conv_states[mamba_track_indices[i]] = conv_states[cache_indices[i]]
  ssm_states[mamba_track_indices[i]] = ssm_states[cache_indices[i]]
  ```

### 8.5.3 一个 Request 实际产生的锚点

以单 chunk prefill 15734 input + 374 output（典型场景，`chunked_prefill_size=16384 ≥ 15734`）为例：

```
阶段                  持久化的锚点位置        数量
─────────────────────────────────────────────────
Prefill (1 chunk)     15680                  1 个
Decode (374 token)    15872                  1 个 (16128 超出 15734+374=16108，不触发)
─────────────────────────────────────────────────
合计                                          2 个锚点
```

**整个 request 生命周期只产生 2 个锚点**，不是 245 个。FLA 的 chunk 计算粒度（64）只决定 `h` tensor 中能取到哪些位置，**持久化粒度由 prefill 调用边界决定**。

#### Ping-pong slot 时间轴

```
T0 (prefill 末):       slot A = h[15680]  ← cache_unfinished_req fork → tree 节点 mamba@15680
T1 (decode 到 15872):  slot B = h[15872]  ← (decode 中途，ping-pong 翻转)
T2 (decode 到 16108):  cache_finished_req 用 slot B (15872) fork → tree 节点 mamba@15872
                       (此时 working state @16108 没有被持久化)
```

ping-pong 只保留最近 2 个，但 `cache_unfinished_req` / `cache_finished_req` 时会 `fork_from` 到独立 slot 给 radix tree（不被后续 track 覆盖）。

### 8.5.4 锚点稀疏对命中的影响

由于一次 prefill 调用只产生 1 个锚点，**比 cached 锚点短的 follow-up 请求会 split miss**：

| 场景 | Tree 锚点 | 新 req key 长度 | descent 行为 | 结果 |
|---|---|---|---|---|
| Req 重复同长度 (15734) | node@15680 | 15733 | 进 node 完全匹配 → ELSE → 退出 loop | **HIT 15680** ✓ |
| Req 长度更长 (16500) | node@15680 + node@15872 | 16499 | 走完所有 cached node → 自然退出 | **HIT 15872** ✓ |
| Req **更短** (8000，前缀同) | node@15680 | 7999 | 进 node, prefix_len=7999 < 15680 → **SPLIT** | **MISS** ✗ |

#### 怎么获得更密集锚点

1. **减小 `--chunked-prefill-size`**：把 prefill 拆成 N 个 chunk，每个 chunk 末尾各产生 1 个锚点。代价：N 次 kernel launch + 中间 chunk fork 开销
2. **减小 `--mamba-track-interval`**：仅影响 decode 期间锚点密度（默认 256 → 64 也行），不影响 prefill
3. **scheduler 主动多 chunk**：SGLang 已有 `req.mamba_branching_seqlen` 机制，能识别"分叉点"主动埋锚（需 prompt 模板感知）

### 8.5.5 多 chunk prefill 的锚点分布示例

**例**：64K 输入 + `chunked_prefill_size=16384` → 拆成 4 chunk。

```
Chunk 拆分：
  Chunk 1: token 0      → 16384  (16384 tok)
  Chunk 2: token 16384  → 32768  (16384 tok)
  Chunk 3: token 32768  → 49152  (16384 tok)
  Chunk 4: token 49152  → 64000  (14848 tok)

每个 chunk 结束后触发 cache_unfinished_req，按公式：
  anchor 位置 = len(prefix_indices) + (extend_input_len // 64) * 64

Chunk 1: 0     + (16384 // 64)*64 = 16384
Chunk 2: 16384 + (16384 // 64)*64 = 32768
Chunk 3: 32768 + (16384 // 64)*64 = 49152
Chunk 4: 49152 + (14848 // 64)*64 = 64000
```

最终 radix tree 持有 **4 个 mamba 锚点**：

```
root → N1 (key_len=16384, mamba@16384)        ← Chunk 1
         └→ N2 (key_len=16384, mamba@32768)    ← Chunk 2
              └→ N3 (key_len=16384, mamba@49152) ← Chunk 3
                   └→ N4 (key_len=14848, mamba@64000) ← Chunk 4
```

加上 decode 期间每 256 token 触发的锚点（如生成 1000 token 会多 3 个）。

#### 命中场景对比

| Follow-up 长度 | 单 chunk prefill (15734) | 4-chunk prefill (64000) |
|---|---|---|
| 同长度重复 | HIT 15680 ✓ | HIT 64000 ✓ |
| 16400 (≈ 16384) | MISS ✗ | **HIT 16384** ✓ |
| 33000 | MISS ✗ | **HIT 32768** ✓ |
| 50000 | MISS ✗ | **HIT 49152** ✓ |
| 8000 (更短) | MISS ✗ | MISS ✗（仍小于第一个锚点 16384） |

**4-chunk 自然提供 ~16K 粒度的锚点网格**。chunked_prefill_size 越小命中粒度越细。

#### 显存代价

每个 fork 的锚点 = 1 个独立 mamba slot = **24 MB**。

单个 64K request 多 chunk prefill 后总占用：

| 占用 | slot 数 | 显存 |
|---|---|---|
| Working state | 1 | 24 MB |
| Ping-pong buffer | 2 | 48 MB |
| Radix tree fork (4 chunk) | 4 | 96 MB |
| **合计** | **7** | **168 MB** |

⚠️ 注意 sglang 的 ratio 公式（`MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP=2`）按 **5 slot/req** 算预算，假设单 chunk 1 fork。**多 chunk 场景实际占用超出 ratio 预算**，可能挤压并发上限或导致 mamba pool 提前满。

### 8.5.6 对比纯 self-attention：为什么混合 SSM 受这个限制

**短 follow-up miss 是混合 SSM 模型独有问题**，纯 self-attention 模型完全没有。核心区别在 split 语义。

#### MambaRadixCache 的 `_split_node`
```python
new_node.mamba_value = None  # 强制 tombstone！
```
mamba state **不能切片**（state 在位置 N 的值≠任意中间位置的 state），所以 split parent 必须置 None。`_match_prefix_helper` 末尾的 `if node.mamba_value is not None` 检查失败 → `best_value_len=0` → miss。

#### 普通 RadixCache 的 `_split_node`（`radix_cache.py:674`）
```python
new_node.value = child.value[:split_len].clone()  # 切片 KV indices，照常返回
```
KV cache 是 **per-token 独立**存储的，切前 N 个 token 的索引和切前 N 个 KV 一一对应。普通 RadixCache 的 `_match_prefix_helper`（`radix_cache.py:648`）也**没有 mamba_value 这个 check**，直接返回切出来的 value：
```python
if prefix_len < len(child.key):
    new_node = self._split_node(child.key, child, prefix_len)
    value.append(new_node.value)  # 切片后的 KV indices，HIT！
    node = new_node
    break
```

#### 同样场景 - 不同结局

设 Req 1 = 15734 tokens，Req 2 = 8000 tokens（共享前 8000 token 前缀）。

**纯 self-attention（LLaMA / Qwen2 / Mistral 等）**：
```
Tree: root → node_A(15734 tokens, KV pages [p1..p15734])

Req 2 match key = 7999:
  iter 1: 进入 node_A
    prefix_len = 7999 < 15734 → SPLIT
    new_parent.value = node_A.value[:7999]  ← 前 7999 个 KV page 索引！
  返回 value = [前 7999 KV indices]，HIT 7999 ✓
```

**混合 SSM（Qwen3.5）**：
```
Tree: root → node_A(15680 tokens, KV pages + mamba@15680)

Req 2 match key = 7999:
  iter 1: 进入 node_A
    prefix_len = 7999 < 15680 → SPLIT
    new_parent.mamba_value = None ← 强制 tombstone
    new_parent.value = node_A.value[:7999]（KV 切片其实存在！）
  best_value_len 检查 mamba_value=None 失败 → 0
  返回 value = []，MISS ✗
```

**讽刺**：KV cache 部分实际上已经切好了（`new_parent.value` 是有效的 7999 个 KV indices），**只因为 mamba state 这一部分无法 split，整个 cache 都被放弃**。

#### 为什么混合模型必须这么严格

混合模型的 prefill / decode kernel 是**所有层一起算的**（24 GDN + 8 attention 在同一 forward pass）：
- 如果只用 KV cache 的 7999 个 token 而 mamba state 没有 7999 位置的快照
- → mamba 层就要从位置 0 重算（即重新 prefill 7999 个 token）
- → 既然要重新 forward 7999 个 token，KV cache 的复用就没意义了（forward 时 KV 也会被覆盖式重算）

所以 sglang 选择**全有全无**：要么 mamba state 严格对齐到 cached 位置（命中），要么完全跳过 cache（miss）。这是混合模型的**架构耦合代价**，不是实现 bug。

#### 模型架构 vs 前缀缓存能力

| 模型类型 | "短 follow-up" 命中 | 任意位置 split 安全 |
|---|---|---|
| **纯 attention** (LLaMA, Qwen2, Mistral) | ✓ 任意粒度 | ✓ |
| **MLA** (DeepSeek) | ✓ 任意粒度 | ✓ |
| **混合 SSM** (Qwen3.5, Jamba, Falcon-H1) | ✗ 仅锚点位置可用 | ✗（mamba 必 tombstone） |
| **纯 SSM** (Mamba) | ✗ 仅锚点位置可用 | ✗ |

---

## 8.6 `extra_buffer` 对性能的影响

### 8.6.1 显存占用（绝对值）

Qwen3.5-9B TP=2 bf16（实测数字）：

| 项目 | 数值 | 计算 |
|---|---|---|
| 单 slot ssm_state | ~**24 MB** | 24 层 linear × H × K × V × 2 bytes / 1013 slot |
| 单 slot conv_state | ~**0.55 MB** | 24 层 × conv_kernel_dim × (k_dim+v_dim) × 2 bytes / 1013 slot |
| **单 slot 合计** | **~24.5 MB** | |
| Mamba pool 总大小 | **24.3 GB/GPU** | ssm 23.77 + conv 0.56 |

**两种模式 mamba pool 物理大小完全相同**（由 `mamba_full_memory_ratio=0.9` 和总空闲显存决定）。

### 8.6.2 并发度退化（`max_running_requests` 公式）

`model_runner_kv_cache_mixin.py:366` 的计算：
```python
MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO = 3      # 基础比例
MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP = 2          # extra_buffer 额外占用

if enable_mamba_extra_buffer():
    additional_ratio = 2
if disable_radix_cache:
    ratio = 1
else:
    ratio = 3 + additional_ratio

max_num_reqs ≤ max_mamba_cache_size // ratio
```

实测匹配：

| 模式 | ratio | `max_mamba_cache_size / ratio` | 实测 `max_running_requests` |
|---|---|---|---|
| no_buffer + radix | 3 | 1013 / 3 = 337 | **337** ✓ |
| extra_buffer + radix | **5** | 1013 / 5 = 202 | **202** ✓ |
| 任意 + `--disable-radix-cache` | 1 | 1013 | 1013 |

### 8.6.3 为什么 ratio 是 3 和 5

每个 running req 在其生命周期内需要占用这些 mamba slot：

**`no_buffer` (ratio = 3)**：
```
1 slot: req 自己的 working state (req.mamba_pool_idx)
1 slot: cache_unfinished_req 时 fork 到 radix tree 的 mamba_value
1 slot: cache_finished_req 时 fork 到 radix tree 的 mamba_value
──────
3 slot/req
```

**`extra_buffer` (ratio = 5)**：
```
1 slot: req 自己的 working state
2 slot: ping-pong track buffer (mamba_ping_pong_track_buffer)
1 slot: cache_unfinished_req 的 fork
1 slot: cache_finished_req 的 fork
──────
5 slot/req
```

多出来的 2 slot 就是 **ping-pong buffer**，每 req 多占 ~49 MB 的逻辑预算。

### 8.6.4 Prefill 时延影响（~5%）

额外工作：
- FLA chunk 的 `h` tensor 要保留而不是丢弃（额外显存带宽）
- `_track_mamba_state_extend`：按 indices 把 `h` 元素 copy 到 ping-pong slot
- `_init_track_ssm_indices`：CPU 计算 src/dst indices

实测 15734-token prefill 耗时：

| 模式 | 总 latency | 估算 prefill-only |
|---|---|---|
| no_buffer | 3.47s（+ 375 token decode） | ~0.97s |
| extra_buffer | 3.83s（+ 421 token decode） | ~1.02s |

**prefill 本身慢 ~5%**，对长序列 prefill 不显著。

### 8.6.5 Decode 时延影响（< 1%）

每 256 token 一次 `track_mamba_states_if_needed` Triton kernel，亚毫秒级。实测 decode 吞吐 ~150 tok/s，两种模式差异在噪声内。

### 8.6.6 命中后的净收益

关键：**cache 命中时 prefill 从 15734 token 变成 54 token**。

| 场景 | Prefill tokens | Prefill 时长（估算） |
|---|---|---|
| no_buffer cache miss | 15734 | ~970 ms |
| extra_buffer cache hit | 54 | ~3.4 ms |

TTFT 省掉 **99.6%**，远超 5% prefill 慢和 40% concurrency 下降的代价（对有前缀复用的场景）。

### 8.6.7 不同场景的净收益判断

| 场景 | cache 命中率 | 推荐 |
|---|---|---|
| 随机 zero-shot prompts，无前缀重用 | ~0% | `no_buffer + --disable-radix-cache` |
| 多轮对话（累加 context） | 80-95% | `no_buffer` 够用（multi-turn 天然满足 "match key > cached end" 条件） |
| 重复相同请求（eval/测试/debug） | 99%+ | **必须 `extra_buffer`**（no_buffer 必 miss） |
| 共享 system prompt + 不同 user query | 60-90% | **必须 `extra_buffer`**（no_buffer 第一次 split 后 shared prefix 永久 tombstone） |

**经验阈值**：cache 命中率 > 67% 时，`extra_buffer` 的收益超过 40% concurrency 代价。

---

## 8.7 替代优化方案

对"`extra_buffer` 太吃并发度"有几个方向，按可行性排序：

### 方案 A（最简单）：单 buffer 替代 ping-pong

**改动**：`memory_pool.py:171` 把 `mamba_ping_pong_track_buffer_size` 从 2 改成 1，同步 `MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP` 从 2 改成 1。

**收益**：ratio 从 5 → 4，concurrency 从 202 → **253** (+25%)。

**代价/风险**：ping-pong 的作用是"下一次 track 完成前保留上一次结果"。去掉后需验证 `cache_unfinished_req` 的 `fork_from` 在 track kernel 完成之后再执行（看代码 sync 调用，应该安全但需要 runtime 测试）。

### 方案 B（中等）：Mamba state fp8 量化

**改动**：仿照 SGLang 已有的 `kv_cache_dtype=fp8`，扩展到 mamba state（`MambaPool` 存 fp8，用时 dequant）。约 ~100 行 kernel。

**收益**：单 slot 24 MB → 12 MB，pool 从 1013 slot → 2026 slot，concurrency **翻倍**（ratio=5 下 202 → 405）。也可以选择扩大 track 密度。

**代价/风险**：
- 精度：SSM state 动态范围比 KV 广，fp8 损失需实验验证（预估 < 1%）
- 实现复杂度：需新 kernel 和反量化逻辑

### 方案 C（长期最优）：0-token Prefill

**改动**：
1. Scheduler 改 `max_prefix_len = input_len`（去掉 `-1`）
2. Prefill kernel 支持 `extend_input_len = 0`，从 cached state 直接出 next-token logits（本质上是把 decode 第 1 步包装成 prefill）

**收益**：
- 相同请求 100% 命中，无需 prefill
- **`no_buffer` 模式也能用**（不依赖中间 state）
- 非 mamba 模型也受益（省 1 token 的 prefill）
- 前缀共享场景依赖 split 修复，仍需 `extra_buffer`

**代价/风险**：
- 涉及 scheduler 多处改动
- Prefill kernel 的 0-token 路径需要新实现

### 方案 D（复杂）：CPU / HiCache Offload

**改动**：仿照 SGLang 已有的 attention KV HiCache，扩展到 mamba state。GPU 只保留热点 mamba slot，冷 slot offload 到 CPU 内存 / SSD。

**收益**：
- CPU 64 GB 可装 ~2600 slot（10x GPU 容量）
- 异步 prefetch 支持长尾命中

**代价/风险**：
- 单 slot 24 MB，PCIe 4.0×16 (32 GB/s) 每次加载 ~1 ms，可接受
- 实现复杂度高：radix tree 要感知 GPU/CPU 位置、prefetcher、eviction policy
- 对**热点集中**场景收益大；对随机访问收益不明显

### 方案 E（一般）：调整 `mamba_track_interval`

**改动**：启动参数 `--mamba-track-interval 128` 或 `64`。

**收益**：decode 期间 track 频率变化，锚点密度影响命中率。**但 ratio=5 不变，concurrency 不变**。

**适用**：只调整命中概率分布，是精调参数而非根本优化。

### 方案 F（极端）：放弃前缀缓存

`--disable-radix-cache`，ratio=1，concurrency 拉满到 1013。

**适用**：完全随机 prompt + 高并发批量推理（命中率几乎 0 的场景）。

### 8.7.1 方案对比表

| 方案 | 改动量 | concurrency 收益 | 其他收益 | 风险 |
|---|---|---|---|---|
| A: 单 buffer | 1 行常量 | +25% (202→253) | - | 低（需验证时序） |
| B: fp8 量化 | ~100 行 kernel | +100% (202→405) | - | 中（精度风险） |
| C: 0-token prefill | scheduler + kernel 改造 | - | 解决 no_buffer 重复请求 miss；非 mamba 模型也受益 | 中（多处改动） |
| D: HiCache for mamba | 大 feature | 间接（通过外存扩 pool） | 长尾命中场景 | 高 |
| E: 调 track_interval | 参数 | 0 | 命中率微调 | 低 |
| F: 关 radix-cache | 1 flag | 最大（1013 并发） | - | 0（完全不缓存） |

### 8.7.2 推荐组合

- **立即收益**：方案 A（单 buffer），一行常量改动。
- **中期重构**：方案 B（fp8 量化 mamba）或方案 C（0-token prefill）。
- **长期 feature**：方案 D（HiCache for mamba）。

短期内 `extra_buffer` + 方案 A 是最佳折中方案。

---

## 9. 为什么 PagedAttention 换不了这个问题

结论：**混合模型前缀缓存的约束在 SSM 层，与 KV cache 布局无关。**

### 9.1 纯 self-attention 模型

| KV 布局 | 前缀缓存可行性 |
|---|---|
| Contiguous | ✓ 任意粒度 |
| Paged | ✓ 任意粒度 |

两者对纯 attention 都无限制。

### 9.2 Mamba / SSM 层

| 布局设想 | 可行性 |
|---|---|
| "State page" 分页？ | ✗ SSM state 是 per-sequence 单张量，不能分 token |
| 完整 state per-sequence | ✓（本来就是这样） |

**PagedAttention 是为 per-token 的 KV 结构设计的，SSM state 不是 per-token 结构，分页没意义**。

### 9.3 vLLM 的处理

vLLM 对 hybrid 模型（Jamba、Mamba2、Qwen3.5 等）的前缀缓存同样有此约束，典型做法：
- 早期：**不支持前缀缓存**
- 中期：**要求请求在 chunk 边界对齐**
- 现在：同样引入**intermediate state tracking**（与 sglang 的 `extra_buffer` 等价）

这不是 SGLang 的实现 bug，是 **hybrid SSM 模型 + 前缀缓存的固有代价**。

---

## 10. 实验验证

### 10.1 测试环境

- 模型：Qwen3.5-9B（8 full-attn + 24 GDN，head_dim=256，kv_heads=4）
- 硬件：AMD MI300X TP=2
- 框架：SGLang 0.1.dev9895（`HEAD` = `0cfb18a80`）
- 测试负载：15734-token 客服对话 prompt（top_k=1 deterministic）

### 10.2 结果矩阵

| 配置 | 第 2 个请求 `#cached-token` | `finish_reason` | 输出 |
|---|---|---|---|
| `page=1, no_buffer, radix-cache on` | 0 | stop | 正确 |
| `page=1, no_buffer, radix-cache off` | N/A | stop | 正确（但无缓存） |
| `page=1, no_buffer + truncate-by-1 fix` | 15733 | **length (2980)** | **复读机** |
| `page=16, no_buffer, radix-cache on` | ~15728 | (类比未直接重测) | **复读机**（位置错位） |
| **`page=1, extra_buffer, radix-cache on`** | **15680** | **stop** | **正确** ✓ |

### 10.3 性能

相同 req 连发 2 次：
- R1 冷启动：3.83s
- R2 命中 15680 token：2.88s（prefill 只做 54 token）

### 10.4 Multi-turn 验证

用 `page=1, no_buffer`：
- Turn 1 (15000 tok → 200 tok output): `#cached-token: 0`
- Turn 2 (15700 tok, 前 15200 与 Turn 1 相同): `#cached-token: 15200` ✓

印证了场景 C 在 no_buffer 下也能命中。

### 10.5 代码修复

为在 ROCm 上启用 `extra_buffer`，需放开 `server_args.py` 的硬阻挡：

```python
# /opt/sglang/python/sglang/srt/server_args.py:1730
         if self.enable_mamba_extra_buffer():  # extra_buffer
             assert (
-                is_cuda()
+                is_cuda() or is_hip()
             ), "Mamba extra_buffer is only supported on CUDA/HIP devices with FLA backend"
```

原断言过度保守。实际上 `hybrid_linear_attn_backend.py` 的 FLA import 走 `if not is_cpu() and not is_npu()`，HIP 路径正常可用。

---

## 11. 结论与建议

### 11.1 关键 takeaway

1. **RadixAttention 和 PagedAttention 是正交概念**，前者负责"找什么前缀"，后者负责"怎么存 KV"。
2. **混合 SSM 模型的前缀缓存可行性由 SSM 层决定**，和 KV cache 布局无关。
3. **`no_buffer` 模式前缀缓存命中条件**：match key 长度 ≥ 所有 cached node 链末端位置。这让它在多轮对话场景可用、在重复请求场景不可用。
4. **`extra_buffer` 模式**通过把 prefill / decode 末尾的锚点位置对齐到 chunk 边界（FLA_CHUNK_SIZE=64 / `mamba_track_interval`=256），避开 split tombstone 实现命中。一个 request 生命周期通常只产生 2-3 个锚点（prefill 末 1 个 + decode 末若干），不是密集到每 64 token 一个。代价是 40% concurrency。

### 11.2 `extra_buffer` 性能代价速查表

| 维度 | 影响 |
|---|---|
| Mamba pool 绝对显存 | **不变**（24.3 GB/GPU，两种模式相同） |
| `max_running_requests` | **-40%**（337 → 202，由 ratio 3→5 导致） |
| Prefill latency | **+5%**（FLA h tensor 写回 + track kernel） |
| Decode latency | **< 1%**（每 256 token 一次亚毫秒级 Triton copy） |
| 命中后 TTFT | **-99.6%**（15734 token prefill → 54 token prefill） |

### 11.3 生产环境推荐配置（按场景）

| 场景 | 推荐配置 |
|---|---|
| 多轮对话 / 前缀复用率高 | `extra_buffer`（可选方案 A 单 buffer 优化） |
| 重复相同请求（eval/测试） | **必须 `extra_buffer`**（no_buffer 必 miss） |
| Prefix-sharing 高并发 | **必须 `extra_buffer`**（no_buffer 永久 tombstone） |
| 随机 zero-shot prompts（无前缀复用） | `no_buffer + --disable-radix-cache` |

**经验阈值**：cache 命中率 > 67% 时 `extra_buffer` 净收益为正。

通用启动：
```bash
--attention-backend aiter
--mamba-scheduler-strategy extra_buffer
--page-size 1（或 16，extra_buffer 下两者等价）
# 不要 --disable-radix-cache
```
外加 `server_args.py` 的 `is_cuda()` → `is_cuda() or is_hip()` patch（ROCm）。

### 11.4 优化方案推荐路径

| 时间窗口 | 方案 | 预期收益 |
|---|---|---|
| **立即可做** | 方案 A：单 buffer 替代 ping-pong（1 行常量） | concurrency +25%（202 → 253） |
| **下一迭代** | 方案 B：fp8 量化 mamba state（~100 行 kernel） | concurrency +100%（202 → 405） |
| **长期重构** | 方案 C：0-token prefill（scheduler + kernel） | 重复请求 100% 命中，非 mamba 模型也受益 |
| **大 feature** | 方案 D：CPU/HiCache Offload for mamba | 间接扩 pool，利好长尾命中 |

### 11.5 长期建议（上游）

- 默认 `--mamba-scheduler-strategy auto` 的行为应该根据是否启用 radix-cache 做判断：
  - radix-cache on ⟹ `extra_buffer`
  - radix-cache off ⟹ `no_buffer`
- 或者在 `no_buffer + radix-cache on` 时打印 warning，说明重复请求场景会 miss / 出错。
- `mamba_ping_pong_track_buffer_size = 1`（方案 A）验证后可以作为新默认。
- Mamba state 量化（方案 B）和 0-token prefill（方案 C）值得作为独立 PR 推给上游。

---

## 附录 A：关键源码位置

| 模块 | 路径 |
|---|---|
| Aiter attention backend | `/opt/sglang/python/sglang/srt/layers/attention/aiter_backend.py` |
| Hybrid linear attn backend | `/opt/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` |
| FLA chunk kernel | `/opt/sglang/python/sglang/srt/layers/attention/fla/chunk.py` |
| MambaRadixCache | `/opt/sglang/python/sglang/srt/mem_cache/mamba_radix_cache.py` |
| Hybrid KV pool | `/opt/sglang/python/sglang/srt/mem_cache/memory_pool.py` (HybridReqToTokenPool, MambaPool) |
| Server args / 校验 | `/opt/sglang/python/sglang/srt/server_args.py` |

## 附录 B：相关 Issue / 实验记录

- 详细定位过程：`/opt/sglang/optimization_checkpoint.md`
- 实验 6/7/8 覆盖了本报告所有实测数据

## 附录 C：参考文献

1. Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023 — PagedAttention 原始论文
2. Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*, NeurIPS 2024 — RadixAttention 原始论文
3. Yang et al., *Gated Linear Attention Transformers with Hardware-Efficient Training*, ICLR 2024 — GDN / GLA
4. [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) — FLA 开源实现
