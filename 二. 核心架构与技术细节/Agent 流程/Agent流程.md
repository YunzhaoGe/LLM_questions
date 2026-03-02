## 1. 初始化与查询接收阶段（Setup & Input）

1. **Agent 系统初始化**  
   加载核心组件：

   - LLM 后端（e.g., Llama-3.1-70B、Qwen2.5-72B、o1-preview 等，支持 function calling）。
   - 工具集（tools）：预定义函数如 web_search、calculator、file_reader、API_call（e.g., weather API、booking API）。
   - 状态管理器（state）：内存缓冲区存储历史消息、工具输出、中间结果。
   - 配置：max_steps（防无限循环，通常 10–50）、temperature（0.2–0.7，低值更确定性）。  
     （为了准备好执行环境；就像给机器人装上大脑、工具箱和笔记本，避免从零开始）

2. **用户查询接收与解析**  
   接收原始输入（e.g., "帮我计划去北京的周末旅行，包括天气和酒店"）。  
   初步解析：用 LLM 或规则提取意图、参数（e.g., 地点=北京，时间=周末）。  
   如果是多轮对话，加载历史上下文。  
   （为了理解任务边界；生产级：用 intent classifier 防 off-topic 查询）

## 2. 规划阶段（Planning / Reasoning）

3. **初始规划生成**  
   用 LLM 生成初步计划（Plan-and-Execute 变体）：prompt 如 "分解这个任务为步骤: [user query]"。  
   输出：步骤列表（e.g., 1. 查询天气；2. 搜索酒店；3. 汇总行程）。  
   如果是 ReAct，直接跳到 reasoning。  
   （为了结构化复杂任务；优化：用 few-shot 示例提升计划质量，减少后续循环次数）

4. **当前步 Reasoning（思考）**  
   LLM 输入：system prompt + 历史状态 + 当前计划步 + "Think step by step: What to do next?"。  
   输出：reasoning 文本（e.g., "我需要先查天气，因为它影响行程"）。  
   （为了让 Agent “思考”而非盲目行动；就像人类先脑补再动手，减少错误）

## 3. 行动阶段（Acting / Tool Calling）

5. **工具选择与调用**  
   LLM 输出：决定用哪个工具（e.g., "call web_search with query='北京周末天气'"）。  
   框架解析：用 JSON schema 结构化工具调用（function calling）。  
   执行工具：实际调用（e.g., web_search 返回 "晴天，20-25°C"）。  
   处理错误：如果工具失败（e.g., API 超时），记录并重试/切换工具。  
   （为了获取外部信息；生产级：并行调用多工具，提升速度 2–3×）

6. **观察与状态更新（Observation）**  
   工具输出反馈给状态：e.g., "Observation: 北京周末晴天，温度 20-25°C"。  
   更新内存：追加到历史缓冲区，支持长上下文。  
   （为了闭环反馈；优化：用 KV cache / summary 压缩历史，防 token 爆炸）

## 4. 循环迭代阶段（Iteration Loop）

7. **自我评估与循环决策**  
   LLM 输入：reasoning + observation + "Is task complete? If not, next action."。  
   如果未完成：重复步骤 4–6（e.g., 下一步 "搜索酒店"）。  
   终止条件：达到目标、max_steps、检测到 "Final Answer" 信号。  
   （为了多步迭代；常见坑：无限循环，用反射机制如 "criticize previous step" 优化，准确率 +15–20%）

8. **中间反思（Reflection，可选高级）**  
   用另一个 LLM 或同一 LLM 审视历史： "Review: What's wrong? How to improve?"。  
   更新计划：e.g., "天气好，添加户外活动"。  
   （为了自适应纠错；2026 主流，借鉴 o1 的 multi-step thinking，提升复杂任务成功率 20–30%）

（重复步骤 3–8，直到任务完成。通常 3–10 循环，视复杂度而定；就像一个循环的“思考-行动-反馈”链条）

## 5. 输出与后处理阶段（Output & Post-Processing）

9. **最终合成与生成**  
   LLM 输入：全历史 + "Summarize final answer based on all info."。  
   输出：结构化响应（e.g., JSON: {"行程": "...", "天气": "...", "酒店": "..."}）。  
   （为了整合所有观察成连贯结果；优化：用 RAG 增强事实准确）

10. **验证与后处理**  
    自检：用 LLM verifier 检查幻觉/一致性（e.g., "Is this grounded in observations?"）。  
    格式化：添加引用、来源（e.g., "天气来自 web_search"）。  
    如果失败：回滚到上一步或报告错误。  
    （为了质量把关；生产级：集成人类反馈 loop，长期迭代 Agent）

11. **响应返回**  
    流式输出（streaming）或一次性返回最终结果。  
    日志记录：全执行轨迹（reasoning + actions + observations），供调试/审计。  
    （为了用户友好；优化：可视化 dashboard 如 LangSmith 追踪）
