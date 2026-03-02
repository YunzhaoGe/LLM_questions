# Late Chunking (后期分块) 与 Contextual Retrieval：如何解决 RAG 中的 "Lost in the Middle" 问题？

在构建 RAG（检索增强生成）应用时，开发者常面临一个经典的困境：**切片（Chunking）的粒度**。

- 切得太细：语义完整性被破坏，代词（如 "它"、"该公司"）丢失了指代对象。
- 切得太粗：包含了过多无关信息（Noise），导致检索准确率下降（"Lost in the Middle" 现象）。

传统的 "Naive Chunking"（先切片，后 Embedding）往往导致切片丢失上下文信息。例如，切片中包含 "它的股价上涨了 5%"，但丢失了前文中 "Apple Inc." 的主语信息，导致检索时无法匹配到 "Apple 股价" 的查询。

本文介绍两种解决此问题的 SOTA（State-of-the-Art）技术方案：**Anthropic 的 Contextual Retrieval（上下文检索）** 和 **Jina AI 的 Late Chunking（后期分块）**。

---

## 1. 核心问题：为什么传统切片会 "失忆"？

传统 RAG 的索引流程通常是：

1.  **加载文档**：读取长文本。
2.  **切片 (Chunking)**：按固定字符数（如 512 tokens）或句子边界硬切。
3.  **Embedding**：对每个切片独立计算向量。
4.  **入库**：存入向量数据库。

**问题在于第 2 步**：一旦切断，切片就变成了一个孤岛。
假设原文是：

> "Google 在 2023 年发布了 Gemini 模型。**它**具有多模态能力。"

如果切片恰好在句号处断开：

- 切片 A: "Google 在 2023 年发布了 Gemini 模型。"
- 切片 B: "**它**具有多模态能力。"

当用户搜索 "Gemini 的功能" 时，切片 A 可能被检索到，但切片 B 因为只包含 "它" 而无法与 "Gemini" 建立语义关联，导致关键信息丢失。这被称为 **Context Loss（上下文丢失）**。

---

## 2. 方案一：Contextual Retrieval (Anthropic)

**思路：用 LLM 为每个切片 "人工" 补全上下文。**

Anthropic 在 2024 年提出的 Contextual Retrieval 是一种**显式**的上下文增强方法。

### 2.1 原理

在对文档进行切片后，不直接进行 Embedding，而是多加一步 **"Contextualization"（上下文生成）**：

1.  取出一个切片。
2.  将该切片**所在的整篇文档**（或大窗口上下文）一起发给 LLM（如 Claude 3 Haiku）。
3.  要求 LLM 生成一段简短的解释，说明该切片在全文中的具体含义（例如补全主语、时间、地点）。
4.  将生成的解释**拼接到切片开头**。
5.  对拼接后的文本进行 Embedding 和索引。

**处理后的切片 B 变成了：**

> [Context: 本切片讨论了 Google 发布的 Gemini 模型的功能。] > **它**具有多模态能力。

### 2.2 优缺点分析

- **优点**：
  - **大幅提升检索率**：Anthropic 报告称检索失败率降低了 49% (结合 Contextual BM25)。
  - **兼容性强**：生成的文本既增强了 **语义检索（Embedding）**，也增强了 **关键词检索（BM25）**。切片中现在显式包含了 "Google"、"Gemini" 等关键词。
  - **模型无关**：可以使用任何 Embedding 模型。
- **缺点**：
  - **成本高昂**：索引阶段需要对每个切片调用一次 LLM，Token 消耗巨大。
  - **索引速度慢**：LLM 生成速度远慢于简单的 Embedding 计算，适合离线批处理。

---

## 3. 方案二：Late Chunking (Jina AI)

**思路：先通过长窗口模型 "看" 完没全文，再切分向量。**

Jina AI 提出的 Late Chunking 是一种**隐式**的 Embedding 层优化方法，旨在解决 Naive Chunking 破坏语义环境的问题。

### 3.1 原理

Late Chunking 颠覆了 "先切片，再 Embedding" 的顺序，改为 **"先 Embedding，再切片"**。

1.  **全文 Embedding**：使用支持长上下文（如 8192 tokens）的 Embedding 模型（如 `jina-embeddings-v2`），一次性输入整篇文档（或长段落）。
2.  **Transformer 计算**：模型内部的 Transformer 层通过 Attention 机制，计算每个 Token 的向量表示。此时，**每个 Token 的向量已经融合了全文的上下文信息**（因为 Attention 允许 Token "看到" 全文）。
3.  **后期切片 (Late Chunking)**：在拿到所有 Token 的向量序列后，再根据边界（如句子、段落）对**向量序列**进行切分。
4.  **Pooling**：对切分出的向量片段进行 Mean Pooling，得到该切片的最终向量。

**效果：**
即使切片 B 的文本是 "它具有多模态能力"，但其对应的 Token 向量是在阅读了前文 "Google 发布 Gemini" 后计算出来的。因此，"它" 这个词的向量在数学空间上已经指向了 "Gemini" 的语义区域。

### 3.2 优缺点分析

- **优点**：
  - **无需 LLM，成本低**：不需要调用昂贵的生成模型，仅需 Embedding 模型推理，速度快。
  - **保留隐式上下文**：完美保留了长距离依赖，解决了代词指代问题。
  - **实现优雅**：是工程上的 "非侵入式" 优化，不改变数据结构。
- **缺点**：
  - **依赖特定模型**：必须使用支持 Late Chunking 的长上下文 Embedding 模型（通常是 BERT 架构的变体，如 Jina V2/V3），无法直接用于 OpenAI `text-embedding-3` 等黑盒 API。
  - **不增强关键词检索**：由于只是向量层面的优化，切片文本本身没有变化，因此无法改善 BM25 等基于关键词的检索效果。

---

## 4. 总结与选型建议

| 特性         | Contextual Retrieval (Anthropic)                   | Late Chunking (Jina AI)                                 |
| :----------- | :------------------------------------------------- | :------------------------------------------------------ |
| **核心机制** | **文本生成**：用 LLM 显式补全上下文文本            | **向量计算**：利用 Transformer Attention 隐式融合上下文 |
| **索引成本** | **高** (需要大量 LLM Token)                        | **低** (仅需 Embedding 推理)                            |
| **索引速度** | 慢 (LLM 生成延迟)                                  | 快                                                      |
| **适用场景** | 高精度要求、预算充足、需要混合检索 (Vector + BM25) | 追求高性价比、私有化部署、主要依赖向量检索              |
| **兼容性**   | 任意 Embedding 模型 / 任意数据库                   | 需特定 Embedding 模型 (如 Jina V2/V3)                   |

### 开发者建议

1.  **如果你的系统严重依赖关键词检索 (BM25)**：请选择 **Contextual Retrieval**。Late Chunking 无法解决 "文本中没出现关键词" 的问题。
2.  **如果你追求极致的性价比和速度**：请选择 **Late Chunking**。它在不增加额外 LLM 开销的情况下，显著提升了向量检索的质量。
3.  **混合方案**：在极高要求的场景下，可以二者结合——先用 LLM 生成摘要（Contextual），再用 Late Chunking 进行向量化，但这通常属于过度设计（Over-engineering）。

**一句话总结**：

- **Contextual Retrieval** 是给**切片加注脚**，让切片自己"学会说话"。
- **Late Chunking** 是让模型**先读懂全文**，再把理解"注入"到切片的向量中。
