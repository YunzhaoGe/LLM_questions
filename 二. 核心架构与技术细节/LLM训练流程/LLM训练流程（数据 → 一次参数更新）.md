# LLM 训练全链路解析（数据 → 一次参数更新）

目标：从“拿到原始语料”开始，串起清洗与过滤、分词与批次组织、模型结构与参数初始化、Transformer 的前向计算（含因果 mask、残差与归一化）、loss 计算、反向传播与优化器更新（含梯度裁剪、学习率调度、混合精度、分布式与 checkpoint），直到完成“一次参数更新”。每一步都同时回答“做了什么/为什么这么做”，并配一句“就像……”的类比，帮助建立直观的全局理解。

---

## 1. 数据获取与质量控制

1.  **Data Collection**：从网页、书籍、代码库收集海量文本，并引入**合成数据（Synthetic Data）**（为了覆盖广泛领域并利用更强模型生成的高质量推理/代码数据提升上限；就像先把图书馆的书搬来，再请顶级教授专门编写一部分高难度教材）
2.  **Data Cleaning**：去掉噪声、重复与敏感内容（HTML、垃圾文本、PII 等）（为了降低错误模式与偏见被学进去的概率，提升训练信号纯度；就像先把食材里的沙子和坏掉的部分挑出来）
3.  **Data Filtering & Annealing**：按质量评分筛选，并在训练后期使用高质量“退火”数据（为了把算力花在“更值得学”的样本上，并在最后阶段通过优质数据“提纯”模型能力；就像考前冲刺只看错题集和核心考点，不再刷简单题）

---

## 2. 分词体系与训练样本组织

1.  **Tokenization Setup**：选择 BPE/Byte-level BPE，处理未知字符（为了把连续文本变成可计算的离散符号，兼顾压缩率与多语言支持；就像规定字典编目规则，且遇到生僻字能拆成偏旁部首认读）
2.  **Vocabulary Building**：构建词表（现代模型常见 **100k–150k+**）（为了在覆盖率与序列长度之间做权衡：大词表能缩短序列长度、提升推理速度；就像扩充“常用词组”库，一次能说完的就不用拆成几个字蹦）
3.  **Data Tokenization**：把文本编码为 `token_ids` 序列（为了把训练数据变成整数序列，便于 embedding 查表；就像把书页内容转换成条形码，流水线上才能快速扫描）
4.  **Data Sharding and Batching**：切分样本、打包 batch，并做**动态课程学习（Curriculum Learning）**（为了稳定显存占用并让模型由易到难学习；就像把整车货按箱打包，且先搬轻的再搬重的）

---

## 3. 模型定义与参数初始化（前沿架构标准）

1.  **Model Architecture Definition**：确定 Transformer 配置。
    - **Norm**：使用 **RMSNorm** 替代 LayerNorm（为了计算更简单且训练更稳定；就像只调音量大小而不调音色，效率更高）
    - **Activation**：使用 **SwiGLU** 替代 ReLU/GELU（为了更好的性能表现；就像换用了更灵敏的油门响应系统）
    - **Attention**：使用 **GQA (Grouped Query Attention)** 替代 MHA（为了在保持性能的同时大幅降低推理显存占用；就像多人共享一份复习资料，既省纸张又都能学到东西）
2.  **Parameter Initialization**：初始化权重（为了让激活与梯度的数值尺度合理；就像给机器先做校准，让刻度从一开始就对得上）
3.  **Embedding 与位置处理**：使用 **RoPE (Rotary Positional Embeddings)**（为了完美的相对位置外推能力；就像用旋转的角度来表示距离，无论转到哪里相对关系都不变）

---

## 4. 前向计算（得到 logits）

一次训练步通常把输入序列右移一位做 next-token prediction：

- 输入：`x = [x0, x1, ..., x(T-1)]`
- 标签：`y = [x1, x2, ..., xT]`

对每一层 `l = 1..L`：

1.  **RMSNorm**：`H = RMSNorm(X)`（为了稳定数值分布）
2.  **Attention (GQA)**：`O = Attn(H)`，配合 **FlashAttention** 加速（为了极速从上下文检索信息，不占额外显存；就像量子速读，瞬间看完相关段落）
3.  **Residual**：`X = X + O`（为了保留主干信息通道）
4.  **SwiGLU MLP**：`M = SwiGLU(RMSNorm(X))`，再 `X = X + M`（为了提供更强的非线性表达能力）

最后得到 `logits = X_L W_vocab`（为了把“当前理解”翻译成一张“候选词得分榜”）。

---

## 5. Loss 计算（把“猜得好不好”量化成标量）

1.  **Shift 对齐**：用 `logits[t]` 预测 `y[t]`（为了把训练目标变成“预测下一个 token”；就像用前一句话来猜下一句会怎么写）
2.  **Cross-Entropy**：`loss = CE(logits, y)`（常配合 z-loss 稳定训练）（为了把概率分布与真实标签的差距压成可优化的标量；就像把“答题对错与把握度”折算成一个统一的扣分规则）

---

## 6. 反向传播与参数更新（完成一次训练步）

1.  **Backward Pass**：对 `loss` 自动微分得到梯度 `∇W`。
2.  **Gradient Clipping**：裁剪梯度范数（为了防止梯度爆炸；就像下坡时给刹车设上限，避免失控）
3.  **Optimizer Update**：用 **AdamW** 或 **Came** 更新参数，配合 **Warmup + Cosine Decay**（为了让参数平滑启动并逐步精细收敛；就像起步时轻踩油门，高速后慢慢减速入库）
4.  **Mixed Precision**：使用 **BF16 (BFloat16)** 进行计算（为了防止 FP16 的溢出问题，训练更稳定；就像用刻度更粗但量程更大的尺子，既快又不容易爆表）

---

## 7. 规模化训练的工程配套

1.  **Distributed Training**：使用 **3D Parallelism (Data + Tensor + Pipeline)** 与 **ZeRO/FSDP**（为了把万亿参数切分到成千上万张卡上；就像把大工程拆成流水线，且每个工人只拿自己那部分图纸，省脑子又快）
2.  **Checkpointing**：定期保存（为了容灾恢复；就像玩游戏存盘）
3.  **Training Loop**：重复迭代，直到 Loss 收敛或达到 Token 数（通常为 Token 数的 1 Epoch 左右，避免过拟合）。

---

## 8. 评估、微调与部署前处理（前沿延伸）

1.  **Evaluation**：监控 Perplexity 与下游任务指标。
2.  **Post-Training (SFT & RL)**：
    - **SFT**：指令微调。
    - **Reasoning Distillation**：使用 **DeepSeek-R1** 类方法的推理数据蒸馏（为了让小模型学会深度思考；就像把天才的思考过程写下来给普通学生看）
    - **Alignment**：**DPO/GRPO** 替代传统的 PPO（为了更直接、稳定地对齐人类偏好；就像直接告诉学生“这个比那个好”，而不是打复杂的评分）
3.  **Quantization**：FP8/INT4 量化（为了极致的推理速度）。
4.  **Deployment**：导出为 GGUF/vLLM 支持格式。
