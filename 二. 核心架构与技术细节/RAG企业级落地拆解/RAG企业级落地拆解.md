## 1. 数据摄入与预处理阶段（Offline/Ingestion Pipeline，影响知识库质量）

1. **数据源采集（Data Ingestion）**  
   从多源拉取原始数据：PDF/Word/网页/数据库/CSV/内部知识库/实时流（Kafka）。  
   使用工具如 Unstructured.io 或 LlamaParse 处理非结构化数据，提取文本/表格/图像。  
   （为了构建全面知识库；生产级：增量更新，每日/实时同步，避免全量重刷）

2. **数据清洗与规范化（Data Cleaning）**  
   移除噪声：HTML 标签、重复段落、敏感信息（PII 脱敏用 Presidio）。  
   标准化：Unicode 统一、语言检测（fastText）、OCR 纠错（Tesseract for 图像文本）。  
   （为了提升下游 embedding 质量；常见坑：脏数据导致召回低 20–30%）

3. **Chunking（数据切块）**  
   把长文档切成小块（chunks）：固定大小（512 token）、语义切分（SentenceTransformers 基于句子/段落边界）、层次切分（标题+子段）。  
   高级：递归 chunking（小块 → 大块），或基于 LLM 的智能切分（e.g., "总结这个段落的核心句"）。  
   元数据附加：来源、时间戳、标签。  
   （为了平衡召回粒度与上下文完整；优化：语义 chunking 提升 MRR 10–20%；工具：LlamaIndex NodeParser）

4. **Embedding 生成（Vectorization）**  
   用 embedding 模型（e.g., BGE-large、BCE、Voyage-2）把每个 chunk → 高维向量（768–4096 维）。  
   生产级：异步批量处理（Ray/Dask），多模态支持（CLIP for 图文）。  
   （为了量化语义相似；收益：用 domain-specific 微调 embedding 模型，提升 Recall@10 15–25%）

5. **Upsert 到向量数据库（Indexing）**  
   将 chunk + embedding + metadata 插入 Vector DB（如 Pinecone、Milvus、Weaviate、Qdrant）。  
   配置：HNSW/PQ/IVF 索引算法，namespace 分区（多租户）。  
   （为了高效存储/查询；生产级：分布式 shard，自动扩容，支持亿级向量）

**Ingestion 优化手段（2026 主流）**

- **增量/Delta 更新**：只处理新/改动数据，节省 80% 计算（Apache Nifi + CDC）。
- **多模态扩展**：文本+图像+表格 embedding，融合 CLIP + TableTransformer。
- **成本控制**：用廉价 embedding 模型（如 MiniLM）初筛，再用大模型精炼。
- **监控**：Ingestion 成功率、chunk 分布统计（Prometheus + Grafana）。

## 2. 查询阶段（Online/Serving Pipeline，影响实时性能）

6. **用户查询接收与预处理（Query Ingestion）**  
   接收用户 prompt，应用聊天模板，提取意图（分类器或 LLM zero-shot）。  
   （为了标准化输入；生产级：A/B 测试多版本 prompt）

7. **Query Embedding**  
   用相同 embedding 模型把 query → 向量。  
   高级：Query 重写（用 LLM 扩展同义词/子问题，如 "HyDE" 假设文档生成）。  
   （为了匹配知识库语义；优化：HyDE 提升 Recall 10–15%）

8. **初召回（Retrieval - Dense/Sparse/Hybrid）**

   - Dense：向量相似搜索（cosine/inner-product），top-K（10–100）。
   - Sparse：BM25/TF-IDF 关键词匹配（Elasticsearch）。
   - Hybrid：融合两者（alpha _ dense + (1-alpha) _ sparse），或 graph-based（知识图谱增强）。  
     过滤：metadata filter（时间/来源），post-filter（多样性 de-dupe）。  
     （为了广覆盖；收益：Hybrid 比纯 dense 提升 MRR 20%；工具：Pinecone hybrid index）

9. **Re-ranking（重排序）**  
   用跨编码器模型（e.g., Cross-Encoder/BGE-reranker）对 top-K 结果打分，重排。  
   高级：LLM-based reranker（"rank these docs by relevance to query"）。  
   （为了精炼相关性；收益：NDCG@10 提升 15–30%，减少噪声输入 LLM；工具：SentenceTransformers CrossEncoder）

10. **上下文组装（Context Assembly）**  
    把 top-M（5–20） reranked chunks 拼接成 context，添加引用/来源。  
    压缩：用 LLM 摘要 chunks（"summarize this doc relevant to query"）。  
    （为了喂给 LLM 的输入精炼；优化：压缩减 token 30–50%，降低成本/延迟）

## 3. 生成阶段（Generation with Retrieved Context）

11. **Prompt 构建（Augmented Prompt）**  
    模板：system + "基于以下上下文回答: [context]" + user query + "如果无关说不知道"。  
    高级：Few-shot 示例、CoT 引导。  
    （为了指导 LLM 忠实于检索结果，减少幻觉）

12. **LLM 调用（Inference）**  
    输入 augmented prompt → LLM 生成（vLLM/TensorRT for 加速）。  
    流式输出（streaming）以降低感知延迟。  
    （核心输出；生产级：fallback 到 base LLM 如果检索为空）

13. **后处理与验证（Post-Generation）**  
    事实检查：用 LLM 自评（"is this grounded in context?"）或外部 verifier（Entailment 模型）。  
    格式化：提取结构化输出（JSON）。  
    （为了质量把关；优化：自评过滤掉 20–40% 低质响应）

## 4. 评估与迭代阶段（Offline/Online Monitoring，闭环优化）

14. **Offline 评估**  
    用基准数据集（e.g., RAGAS）计算：Faithfulness（忠实度）、Answer Relevancy、Context Precision 等。  
    （为了量化 pipeline 效果；工具：DeepEval/RAGAS）

15. **Online 监控与 A/B 测试**  
    追踪：E2E 延迟、token 成本、用户反馈（thumbs up/down）。  
    A/B：测试不同 embedding/reranker/LLM 变体。  
    （为了持续迭代；生产级：集成 Sentry/ELK 日志）

16. **反馈循环（Online Learning）**  
    用户纠正/点赞 → 微调 embedding/LLM，或更新知识库。  
    高级：Active learning，选择高不确定样本重标。  
    （为了自适应；收益：长期 Recall 提升 10–20%）
