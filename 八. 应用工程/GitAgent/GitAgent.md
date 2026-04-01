# GitAgent 编译工作流：从普通目录到可运行的 AI 代码

简单来说，GitAgent 就像是一个专门针对 AI Agent 的 **Webpack** 或 **代码生成器**。下面我们来看看，当你敲下 `gitagent export -f langchain` 时，这个黑盒里到底发生了什么：

---

## 1. 解析配置文件：加载环境变量与基础依赖

1. **读 YAML 拿元数据**：解析 `agent.yaml` 文件，提取出你要用的模型（如 `gpt-4o`）、API 密钥路径以及 Python 依赖包版本（为了生成最后的 `requirements.txt` 和初始化 LLM 客户端）。
2. **确定目标框架**：识别你要导出的是 LangChain 还是 OpenAI Assistants，然后加载对应的“代码生成模板”（为了决定最后写出的 `.py` 文件用哪种语法）。

---

## 2. 组装 Prompt（提示词）：把 Markdown 变成 System Message

1. **提取系统人设**：读取 `SOUL.md` 的纯文本内容，直接将其塞进大模型 API 的 `{"role": "system", "content": "..."}` 字段中（为了替代平时写死在代码里的冗长提示词）。
2. **拼接护栏规则**：读取 `DUTIES.md`（能做什么/不能做什么），把它作为补充指令追加到 System Prompt 的末尾，或者转换为执行工具前的 `if/else` 拦截逻辑（为了防止大模型乱调用危险接口）。

---

## 3. 转换工具函数（Tools）：通过反射生成 JSON Schema

1. **扫描 Python 函数**：遍历 `tools/` 目录下的普通 Python 函数（比如你写的一个 `get_weather(city: str)`）。
2. **提取签名与注释**：利用 Python 的反射机制（`inspect` 模块），自动读取函数的参数类型（Type Hints）和文档字符串（Docstring）。
3. **生成大模型认识的 Schema**：把上一步提取的信息，自动翻译成大模型 Function Calling 必须的严格 JSON 结构（为了让大模型知道你有哪些函数可用，以及该传什么参数）。

---

## 4. 挂载记忆状态：用文件读写代替数据库

1. **读取历史上下文**：在生成运行时代码时，注入一段文件读取逻辑：每次调用大模型前，先 `open('memory/context.md', 'r')` 读取历史记忆，拼接到对话记录中（为了让 Agent 知道之前的状态，而不需要你额外配置 Redis 或向量数据库）。
2. **写入与 Git 提交**：在 Agent 执行完任务后，将新的状态更新写回 `.md` 文件，并在后台自动执行 `git add .` 和 `git commit -m "update memory"`（为了让你在 Git 提交历史里，能像看代码 Diff 一样看 Agent 的“脑回路”变化）。

---

## 5. 渲染目标框架代码：套用模板生成胶水代码

1. **生成 LangChain 胶水代码**：如果你导出为 LangChain，黑盒会把你定义的工具和 Prompt 填入 `AgentExecutor` 和 `create_tool_calling_agent` 的标准模板中。
2. **生成 AutoGen 胶水代码**：如果你导出为 AutoGen，黑盒则会生成两个对话节点（如 `AssistantAgent` 和 `UserProxyAgent`），并把你的配置注入进去。
3. **组装路由与工作流**：把 `skills/` 里定义的复杂业务流，翻译成对应框架的链式调用（Chain）或图结构（Graph）代码。

---

## 6. 打包落盘：输出最终的工程目录

1. **生成入口文件**：在目标文件夹生成一个 `app.py` 或 `main.py`，里面已经 import 了所有需要的框架包，并初始化好了 Agent 实例。
2. **输出环境依赖**：生成配套的 `requirements.txt`。
3. **交付可用资产**：最终，你得到了一个完全独立、符合特定框架规范的 Python 项目，你可以直接 `pip install -r requirements.txt` 然后 `python app.py` 把项目跑起来。