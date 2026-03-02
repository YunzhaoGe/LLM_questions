# A-RAG ｜ Agentic RAG 新范式：给模型分层检索工具，让 LLM 自己决定怎么搜、搜多少、什么时候停

> **摘要**：传统 RAG（Retrieval-Augmented Generation）依赖预定义的静态流程，无法应对复杂多变的信息检索需求。2026 年 2 月提出的 A-RAG（Agentic RAG）框架，通过向 LLM 暴露分层检索工具（关键词搜索、语义搜索、切片阅读），让模型自主决定检索策略。这种“Service as Software”的转变，不仅提升了检索精度，更开启了 RAG 系统的 Test-Time Scaling 时代。

---

在 2026 年的今天，RAG（检索增强生成）已经从单纯的“外挂知识库”进化为复杂的 Agent 系统。最新的研究成果 **A-RAG (Agentic RAG)** 揭示了一个核心趋势：**不要试图用工程代码去“硬编码”检索流程，而是把检索工具交给模型，让它自己根据问题去规划。**

本文将从技术实现角度，解析 A-RAG 是如何通过分层检索接口（Hierarchical Retrieval Interfaces）实现这一范式转移的。

## 1. 传统 RAG 的“死板”与 Agentic RAG 的“灵活”

现有的 RAG 系统主要分为两类：

1.  **Naive RAG**：一次性检索（Single-shot）。用户提问 -> Embedding -> Top-K 检索 -> 生成。这种方式对复杂问题（如“比较 A 和 B 的三个区别”）往往无能为力，因为一次检索很难覆盖所有维度的信息。
2.  **Flow Engineering (Iterative RAG)**：通过人工设计的 Workflow（如 LangGraph）进行多步检索。虽然比 Naive RAG 强，但流程是写死的（Hard-coded）。如果预定义的流程是“先搜 A 再搜 B”，遇到不需要搜 B 的情况，它依然会照做，浪费 Token 且增加延迟。

**A-RAG 的核心理念是：把决策权还给模型。**

它不再预设“先搜后写”的线性逻辑，而是将检索能力封装成**Tools（工具）**，让 LLM 在一个 ReAct（Reasoning + Acting）循环中自主调用。

## 2. A-RAG 的三把“手术刀”：分层检索接口

A-RAG 并没有发明新的检索算法，而是改变了检索工具的**交互粒度**。它向 LLM 提供了三个层次的 API，对应人类获取信息的不同方式：

### 🛠️ 工具 1：`keyword_search(query)` —— 精确打击

- **功能**：基于 BM25 的关键词倒排索引检索。
- **场景**：当模型需要查找专有名词、特定 ID、或者非常具体的术语时。
- **类比**：你在书的索引页查找某个特定概念的页码。

### 🛠️ 工具 2：`semantic_search(query)` —— 模糊探索

- **功能**：基于 Dense Vector 的向量相似度检索。
- **场景**：当模型需要理解语义、查找相关概念或回答开放性问题时。
- **类比**：你在图书馆的分类书架前浏览，寻找“关于深度学习的书”。

### 🛠️ 工具 3：`chunk_read(chunk_id)` —— 深度阅读

- **功能**：获取特定文档切片（Chunk）的完整上下文或相邻切片。
- **场景**：当模型发现某条检索结果很有用，但通过 SearchResult 只看到了摘要或片段，需要阅读原文的上下文（Context Window）时。
- **类比**：你翻开书，仔细阅读某一页的具体内容。

## 3. 工作流：让模型自己“Loop”起来

在 A-RAG 中，LLM 不再是被动的“填空者”，而是主动的“调查员”。一个典型的执行流程如下：

1.  **Thought（思考）**：用户问了这个问题，我需要先确认实体 A 的定义。
2.  **Action（行动）**：调用 `keyword_search("实体A")`。
3.  **Observation（观察）**：获得了 5 条关于实体 A 的简短描述。
4.  **Thought（思考）**：描述 3 看起来最相关，但我需要更多细节来对比。
5.  **Action（行动）**：调用 `chunk_read("doc_123_chunk_45")` 读取上下文。
6.  **Observation（观察）**：获得了详细的技术参数。
7.  **Thought（思考）**：现在我有足够的信息了，可以停止检索并回答用户。
8.  **Final Answer（回答）**：输出最终答案。

在这个过程中，**“怎么搜”（选哪个工具）、“搜多少”（Loop 多少次）、“什么时候停”（判断信息充足）** 完全由 LLM 根据其内部推理能力（Internal Reasoning）决定。

## 4. Test-Time Scaling：算力换智能

A-RAG 的一个重要特性是支持 **Test-Time Scaling（推理时扩展）**。

- **传统 RAG**：性能上限受限于 Retriever 的准确率和固定的 Top-K。
- **A-RAG**：可以通过增加推理时的计算量（允许模型进行更多的思考和检索轮次）来换取更好的效果。

实验数据表明，随着允许的检索轮次增加，A-RAG 在复杂问答任务上的准确率呈现明显的上升趋势。这意味着，对于高价值的难题，我们可以让 Agent“多想一会儿，多翻几本书”，从而得到超越模型参数规模的回答质量。

## 5. 对开发者的启示

对于 AI 应用程序员来说，A-RAG 的出现意味着架构设计的重心转移：

1.  **从 Pipeline 到 Toolset**：停止编写冗长的 `if-else` 业务逻辑来控制检索流程。把精力花在打磨**检索工具的 API 定义**和**数据索引的质量**上。
2.  **Prompt Engineering 的新方向**：你的 Prompt 不再是教模型“如何生成答案”，而是教模型“如何使用工具”以及“如何判断信息是否充分”。
3.  **关注小模型的 Reasoning 能力**：A-RAG 依赖模型的推理和决策能力。随着 DeepSeek-R1 等具备强大推理能力（甚至通过蒸馏的小模型）的普及，在端侧或低成本场景部署 Agentic RAG 变得可行。

## 6. 决策指南：你该不该用 A-RAG？

虽然 A-RAG 看起来很美好，但在决定是否引入生产环境前，你需要冷静考虑以下三个 Trade-off（权衡）：

### ⚠️ 权衡 1：延迟与成本 (Latency & Cost)

A-RAG 的本质是用“时间换智能”。Naive RAG 一次检索 + 一次生成可能只需要 1 秒。而 A-RAG 可能需要 3-5 轮 ReAct 循环，每轮都要调用 LLM 和检索工具。

- **延迟**：如果你的应用对首字延迟（TTFT）极其敏感（如实时客服），A-RAG 可能太慢了。
- **成本**：多轮对话意味着 Input Token 成倍增加。你需要评估单次查询的价值是否值得消耗 5 倍以上的 Token。

### ⚠️ 权衡 2：模型智力门槛 (Model Capability)

不要指望 7B 甚至 14B 的小模型能完美驾驭 A-RAG。

- **原因**：A-RAG 极其依赖模型的 Instruction Following 和 Reasoning 能力。弱模型容易陷入死循环（反复搜同一个词）或过早停止（搜了一次就瞎编）。
- **建议**：生产环境建议至少使用 Claude 3.5 Sonnet, GPT-4o 或 DeepSeek-R1 等第一梯队模型。

### ⚠️ 权衡 3：不可控性 (Unpredictability)

- **Naive RAG**：结果是确定性的，Retriever 搜不到就是搜不到。
- **A-RAG**：是概率性的。同一个问题，Agent 每次走的检索路径可能不同。对于合规性要求极高（如金融、医疗）的场景，这种“自由发挥”可能带来幻觉或风控风险。

### ✅ 什么时候用？

- **适合**：深度研报生成、复杂代码库问答、法律案件分析、开放式探索。
- **不适合**：简单的 FAQ 问答、高并发低延迟的 C 端搜索、对幻觉零容忍的场景。

## 结语

A-RAG 不仅仅是一个新的缩写，它代表了 AI 应用开发从 **Software as a Service (SaaS)** 向 **Service as Software (Agentic Workflows)** 的转变。在这个新范式中，程序员构建环境和工具，而由 AI Agent 在其中自主探索，寻找答案。

> **参考文献**
>
> - [2602.03442] A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces (Feb 2026)
