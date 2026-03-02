# LLM 推理加速全流程（请求 → TTFT → 吞吐）

目标：把一次在线推理从“请求到达”拆成 CPU 预处理、GPU prefill、GPU decode、输出返回四段，并把每段的性能瓶颈说清楚：哪里影响 TTFT（首 token 延迟），哪里决定吞吐（tokens/s），以及各段常见的工程级优化（如前缀/Prompt cache、连续批处理、PagedAttention/RadixAttention、FlashAttention/FlashInfer、投机解码、量化、多 GPU 并行与调度）。每一步都同时回答“做了什么/为什么这么做”，并配一句“就像……”的类比，建立可以落地的性能心智模型。

---

## 1. 请求到达与预处理（CPU 侧主导：决定 TTFT 的起跑线）

1. Prompt 组装：把 system prompt + history + user message 按 chat template 拼接，插入角色/边界控制 token（为了让模型明确轮次与边界，避免“谁在说话”含混；就像把会议纪要按发言人和段落打上标签，后面检索才不会串台）
2. 编码与控长：用 tokenizer 把文本编码成 `input_ids`，并在超出上下文窗口时截断/滑窗/保留关键片段（为了严格复现训练期编码规则并避免 OOM；就像先按出版社规范做目录索引，再把行李控制在登机尺寸内）
3. 前缀复用（可选但常见）：对固定且高频的前缀复用缓存的 `input_ids`，甚至直接复用对应的 prefill 结果（prefix KV/prompt cache）（为了把重复编码与前向变成一次性成本，显著压低 TTFT；就像常用公文先盖章存档，下次直接复印不用从头写）
4. Batch 对齐：对批处理请求做 padding 或变长打包，并生成 `attention_mask`/`cu_seqlens` 等（为了让 GPU 并行吃满，同时避免 padding 参与计算；就像把不同长度的快递要么塞同规格箱子并填充泡沫，要么按真实体积拼箱运输）

---

## 2. 进入 GPU：Prefill（一次性“读完提示词”：通常 compute-bound，也可能被长上下文带宽拖慢）

Prefill 的产物是：最后一个位置的隐藏状态（用于采样第一个新 token）+ prompt 全部历史的 KV cache（用于后续 decode 复用）。

1. Embedding + 位置处理：`input_ids → embedding lookup`，并叠加/注入位置编码（常见 RoPE）（为了把离散 id 映射到连续向量空间并编码顺序信息；就像把“编号清单”换成带坐标的地图，同时标出先后顺序）
2. Transformer 前向（整段提示词）：对整段 prompt 做多层 self-attention/MLP，计算 Q/K/V、mask、softmax、加权聚合（为了把“整段上下文”压缩成可用于预测下一个 token 的表示；就像通读整篇材料再写摘要，而不是只看最后一句）
3. KV cache 初始化：把 prompt 的每个位置在每一层产生的 K/V 写入 cache（为了让后续每生成一个 token 时不必重算历史，只需复用历史 K/V；就像先把所有资料做成索引卡片，后面只查卡片不重读全文）
4. 产出第一个 token：取最后位置 hidden state 过 `lm_head → logits → 解码策略` 得到 token_1，TTFT 主要由“CPU 预处理 + prefill + 采样”构成（为了尽快让用户看到第一段输出；就像先把第一勺汤端上桌，后面再慢慢上菜）

---

## 3. Decode 循环（逐 token 生成：memory-bound 的“读 KV cache”）

Decode 的基本特征：每步只新增 1 个 token 的计算，但要对齐并读取越来越长的历史 KV，因此常见瓶颈是显存带宽与 cache 管理，而不是纯算力。

1. 单步输入：把“最新生成的 1 个 token”作为当前步输入，并设定当前位置（RoPE 的 position、或等价位置信号）（为了让模型知道“这是第几步接着写”；就像在接龙时先确认现在轮到第几句）
2. 单步前向：新 token 产生 Q，注意力用新 Q 匹配历史 KV，并计算当前步输出（为了把计算从“整段重算”降为“只算一步”，但代价是每步都要读很长的 KV；就像只写一页续集，但要不断翻前面的设定）
3. KV 更新：把当前步的 K/V 写入并追加到 cache（为了让下一步能复用到最新历史；就像把新写的一页装订进书脊，后面翻书能看到）
4. 解码与停止：`lm_head → logits → 解码策略` 选出下一 token，遇到 stop/EOS 或到达 `max_tokens` 结束（为了在质量、确定性与预算之间取舍；就像写作时一边选句子一边看篇幅上限）

---

## 4. 输出后处理与返回（用户感知：流式是默认）

1. Detokenization：把 `output_ids` 还原为文本，并处理特殊 token/空白规则（为了把模型内部的离散符号变成用户可读字符串；就像把零件编号重新拼回自然语言）
2. 增量输出：把新生成片段拼接/去重后返回给客户端（为了避免重复片段并让 UI 平滑展示；就像直播打字时只推送新敲出来的字）
3. Streaming 返回：边 decode 边推送增量文本（为了降低“感知延迟”，即使总生成较长也能尽快看到进展；就像边写边直播打字，而不是写完一整篇才发）
4. 收尾与清理：结束后释放 KV cache/上下文状态并记录关键指标（TTFT、tokens/s 等）（为了避免资源泄漏并让优化有数据闭环；就像演出结束后清场并统计上座率）

---

## 5. 工程级优化地图（Prefill 侧：压 TTFT 的“算子与并行”）

Prefill 更接近“读完一整段提示词”的大矩阵乘，优化重点通常是算子效率、显存访问与并行方式。

1. FlashAttention / FlashInfer：用 fused、IO-aware attention 内核减少中间矩阵物化与 HBM 读写（为了把同样的注意力算得更省显存更快；就像不把整张表格打印出来再算，而是在流水线上边读边汇总）
2. 前缀/Prompt cache（prefix KV）：对重复前缀复用已算好的 KV（有的实现会做 prefix sharing/radix tree）（为了把最贵的 prefill 在跨请求场景摊薄，TTFT 收益常很可观；就像常用参考资料先做成速查表，后面查一次顶读一遍书）
3. Chunked / Disaggregated Prefill：对超长 prompt 分块 prefill，或把 prefill 与 decode 分离到不同 GPU（为了降低单卡峰值压力并改善尾延迟；就像先分批搬运大件行李，再在另一条传送带上分拣）
4. 量化与并行：权重量化（INT4/AWQ/GPTQ/FP8）+ 多 GPU 并行（TP/PP/EP）（为了减少带宽/GEMM 成本并把更大负载摊到多卡；就像把手册做成口袋版同时把装配线拆成多工位）

---

## 6. 工程级优化地图（Decode 侧：提吞吐的“KV 与调度”）

Decode 每步计算很小但要反复读写越来越长的 KV cache，优化重点通常是 KV 的组织方式、batch 的动态调度与“少走大模型步数”。

1. PagedAttention / RadixAttention：把 KV cache 分页/分块管理，减少碎片并支持高并发（为了在长上下文与多请求下稳定利用显存；就像用书架的分页目录管理散页，不要求整本书必须连续放）
2. 连续批处理 + 调度：动态进出 batch，并按长度/阶段/内存水位做路由与驱逐（为了避免“等最长序列”的空转并把吞吐与延迟做平衡；就像拼车滚动发车，同时由调度台实时改道分流）
3. Speculative Decoding：小 draft 模型先猜一段，大模型并行验证并一次性接受多个 token（为了用更少的大模型步数换同样输出长度；就像先让速记员打草稿，主笔一次性审核通过一段）
4. KV 量化（KV cache）：KV cache 用 INT8/FP8（甚至更低比特）降低带宽与显存占用（为了让 decode 这个“读 KV 的带宽瓶颈”更缓一些；就像把仓库通道拓宽不了，就先把箱子做薄一点）
