1. Data Collection（从各种来源如网页、书籍、代码库等收集海量文本数据，为了构建一个多样化、高质量的语料库，确保模型能学习广泛的知识和语言模式）。

2. Data Cleaning（移除噪声、重复、低质量或敏感内容，如 HTML 标签、垃圾文本、PII 数据，为了提高数据质量，减少偏见和错误，避免模型学习有害模式）。

3. Data Filtering（根据语言、主题、长度等标准过滤数据，为了聚焦于相关和高价值内容，确保训练效率和模型泛化能力）。

4. Tokenization Setup（定义词汇表和分词规则，如使用 BPE 或 WordPiece 算法，为了将文本转换为 token 序列，便于模型处理标准化输入）。

5. Vocabulary Building（从清洗后的数据中构建词汇表，通常 10k-100k 个 token，为了覆盖常见词、子词和特殊符号，优化表示效率）。

6. Data Tokenization（将所有文本转换为 token ID 序列，为了将自然语言映射到数值输入，便于嵌入和模型计算）。

7. Data Sharding and Batching（将 tokenized 数据分片并组织成批次，为了支持分布式训练，提高并行处理效率）。

8. Model Architecture Definition（设计 Transformer 结构，包括层数、注意力头、隐藏维度等，为了确定模型容量和计算复杂度，如 GPT 的解码器-only 架构）。

9. Parameter Initialization（使用如 Xavier 或 He 初始化随机设置权重，为了提供一个合理的起点，避免训练从零开始时的数值不稳定）。

10. Embedding Layer Setup（为 token 和位置创建嵌入矩阵，为了将离散输入转换为连续向量表示，捕捉语义和位置信息）。

11. Positional Encoding Addition（如 Sinusoidal 或 RoPE：将位置信息添加到嵌入中，为了让模型理解序列顺序，避免位置无关假设）。

12. Forward Pass - Attention Layers（通过多头自注意力计算查询-键-值交互，为了捕捉 token 间依赖，生成上下文增强表示）。

13. Forward Pass - Feed-Forward Layers（应用 MLP 层引入非线性变换，为了进一步精炼每个 token 的表示，提高表达能力）。

14. Residual Connections and Normalization（在每个子层后添加残差和层归一化，为了稳定梯度流，缓解深层网络的训练难度）。

15. Mask Application（在因果 LM 中使用因果掩码，为了确保模型只依赖过去 token，实现自回归预测）。

16. Output Projection（将最终隐藏状态投影到词汇表维度，为了生成 logits，用于预测下一个 token）。

17. Loss Calculation（如 Cross-Entropy：比较预测 logits 与真实下一个 token，为了量化模型错误，提供优化信号）。

18. Backward Pass（通过自动微分计算梯度，为了传播错误信号，回溯更新每个参数）。

19. Gradient Clipping（可选，裁剪梯度范数，为了防止梯度爆炸，提高训练稳定性）。

20. Optimizer Update（如 AdamW：使用梯度更新参数，结合学习率调度，为了逐步最小化损失，实现收敛）。

21. Mixed Precision Training（使用 FP16/bfloat16 进行计算，为了加速训练并减少内存使用，同时保持精度）。

22. Distributed Training Setup（如 Data Parallelism 或 Model Parallelism：在多 GPU/TPU 上分担计算，为了处理大规模模型和数据，提高吞吐量）。

23. Checkpointing（定期保存模型状态，为了允许恢复训练，防止中断损失）。

24. Training Loop（重复前向-后向-更新步骤，通过多个 epoch 遍历数据，为了逐步改进模型，直到损失收敛或达到预设迭代）。

25. Evaluation During Training（在验证集上计算指标如 Perplexity，为了监控过拟合，提供早停信号）。

26. Fine-Tuning Preparation（可选，为特定任务准备标签数据，如指令调优或 RLHF，为了适应下游应用，提高任务性能）。

27. Fine-Tuning Loop（类似预训练，但使用监督损失或奖励模型，为了微调参数，使模型更贴合特定需求）。

28. Quantization and Pruning（可选，后处理如 INT8 量化或稀疏化，为了减少模型大小，提高推理效率，而不显著牺牲性能）。

29. Final Evaluation（在基准测试如 GLUE、SuperGLUE 上评估，为了验证模型能力，提供性能指标）。

30. Deployment Preparation（导出模型格式如 ONNX 或 HuggingFace，为了便于集成到应用中，实现实际使用）。
