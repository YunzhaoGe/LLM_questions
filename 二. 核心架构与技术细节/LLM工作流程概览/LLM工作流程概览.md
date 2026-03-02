1. Tokenization（将输入提示词文本拆分成离散的 token 序列，为了将自然语言转换为模型可处理的标准化单元，避免歧义并便于后续嵌入）。

2. Vocabulary Lookup（在预训练的词汇表中查找每个 token 对应的唯一 ID，为了将 token 映射到数值索引，便于嵌入层的快速检索）。

3. Embedding（将每个 token ID 转换为高维空间的向量表示，为了捕捉词的语义、语法和上下文信息，使模型能处理相似词的相似性）。

4. Positional Encoding（如 RoPE：将旋转位置嵌入添加到嵌入向量中，为了注入序列中 token 的位置信息，让模型区分相同 token 在不同位置的含义，避免位置混淆）。

5. Input Projection（可选，对嵌入向量进行初始线性变换，为了调整维度或初始化隐藏状态，适应 Transformer 层的输入要求）。

6. Layer Normalization（对输入向量进行归一化处理，为了稳定数值分布，减少内部协变量偏移，提高模型训练和推理的稳定性）。

7. Causal Mask Preparation（生成因果掩码矩阵，为了在自注意力中防止未来 token 影响当前 token，确保生成过程是自回归的）。

8. Query, Key, Value Projection（通过线性层将输入向量投影成 Query、Key 和 Value 矩阵，为了在注意力机制中分别用于查询、键匹配和值聚合）。

9. Multi-Head Splitting（将 Q、K、V 矩阵拆分成多个注意力头，为了并行捕捉不同子空间的依赖关系，提高模型的表达能力）。

10. Scaled Dot-Product Attention（计算 Q 和 K 的点积并缩放，为了度量 token 间的相似度，避免高维导致的数值爆炸）。

11. Apply Causal Mask（在注意力分数上应用掩码，为了屏蔽未来 token 的信息，确保模型只依赖过去和当前上下文）。

12. Softmax Activation（对注意力分数应用 Softmax 函数，为了将分数转换为概率分布，便于后续加权求和）。

13. Attention Output（用注意力概率加权 V 矩阵，为了聚合相关 token 的信息，生成上下文增强的表示）。

14. Head Concatenation（将多头注意力的输出拼接起来，为了整合不同头的视角，形成完整的注意力输出）。

15. Output Projection（通过线性层投影多头输出，为了调整维度并进一步精炼注意力结果）。

16. Residual Connection（将注意力输出与原始输入相加，为了保留原始信息，缓解深层网络的梯度消失问题）。

17. Layer Normalization（再次对残差输出进行归一化，为了维持数值稳定性，继续后续处理）。

18. Feed-Forward Network - First Linear（应用第一层线性变换，为了引入非线性变换，扩展特征空间）。

19. Activation Function（如 GELU 或 ReLU：对线性输出应用激活函数，为了引入非线性，提高模型的表达能力）。

20. Feed-Forward Network - Second Linear（应用第二层线性变换，为了压缩回原始维度，精炼特征）。

21. Residual Connection（将前馈输出与注意力残差相加，为了进一步保留信息流，增强梯度传播）。

22. Layer Normalization（对前馈残差输出进行归一化，为了准备进入下一层或最终输出）。

（重复步骤 6-22，对于每个 Transformer 层，通常有多个层如 12-96 层不等，每层逐步精炼表示，为了构建深层上下文理解）。

23. Final Layer Normalization（对最后一个 Transformer 层的输出进行最终归一化，为了稳定最终隐藏状态，便于输出头处理）。

24. Output Linear Projection（将最终隐藏向量投影到词汇表大小的维度，为了生成每个可能 token 的 logits 分数，表示预测置信度）。

25. Bias Addition（可选，添加偏置项到 logits，为了调整整体分布，补偿线性投影的偏差）。

26. Softmax（对 logits 应用 Softmax 函数，为了将分数转换为概率分布，确保总和为 1，便于 token 选择）。

27. Temperature Scaling（可选，调整 Softmax 温度，为了控制输出的随机性，如低温趋向确定性输出）。

28. Top-K/Top-P Sampling（可选，过滤概率分布，只考虑 Top-K 或 Top-P 累积概率的 token，为了避免低概率噪声，提高生成质量）。

29. Token Selection（从概率分布中采样或取 argmax 选择下一个 token，为了生成第一个输出 token，启动自回归过程）。
