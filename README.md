# D-Agent: 一个模块化的 AI 代理框架

D-Agent 是一个用于学习和实践现代 AI 代理（Agent）架构的沙盒项目。它从零开始构建了一个具备思考、决策和工具使用能力的智能代理，并重点演示了以下核心概念的集成：

* **代理架构 (Agent Architecture)**: 实现了一个基于 "思考 -> 行动" 循环的核心代理。
* **元认知协议 (MCP - Meta-Cognitive Protocol)**: 设计了一套标准化的消息和工具接口，实现系统各组件的解耦。
* **检索增强生成 (RAG - Retrieval-Augmented Generation)**: 集成了一个自动化的知识库检索流程，在代理决策前为其提供相关背景知识。

---

## 核心概念

本项目旨在将理论概念转化为可运行的代码：

1.  **MCP (Meta-Cognitive Protocol)**
    * 在本项目中，MCP 体现在 `mcp/` 目录下的 `protocol.py` 和 `interfaces.py`。
    * `MCPMessage` 定义了系统内所有组件（用户、代理、工具）之间通信的标准“信封”。
    * `BaseTool` 定义了所有工具必须遵守的“接口规范”，确保任何工具都能被代理动态加载和使用。

2.  **RAG (Retrieval-Augmented Generation)**
    * 本项目实现了一个**自动化的 RAG 预处理步骤**。
    * 在代理接收到任何用户请求时，它会**首先**使用 `RAGTool` 从本地知识库（向量数据库）中检索相关信息。
    * 这些检索到的信息会作为“背景资料”被注入到提示词中，帮助 LLM 做出更“知情”的决策。

3.  **Agent (智能代理)**
    * `agent.py` 中的 `SmartAgent` 是系统的“大脑”和“调度中心”。
    * 它**不直接执行**具体任务，而是通过 LLM 来**理解用户意图**，并**决策**下一步行动：是直接回答，还是调用一个合适的工具。

## 项目架构

```
D-Agent/
├── .env.template         # API 密钥的模板文件
├── .gitignore            # Git 忽略规则
├── agent.py              # SmartAgent 的核心实现
├── build_rag_index.py    # [运行一次] 用于构建 RAG 知识库的脚本
├── main.py               # 项目的主入口，支持交互和模拟模式
├── requirements.txt      # 项目的 Python 依赖
├── data/                 # 存放 RAG 的原始知识文档 (.txt)
│   ├── apple_intro.txt
│   └── google_intro.txt
├── mcp/                  # MCP 核心定义包
│   ├── __init__.py
│   ├── interfaces.py     # 定义了 BaseTool 等接口
│   └── protocol.py       # 定义了 MCPMessage 消息格式
└── tools/                # 存放所有可被代理动态加载的工具
    ├── __init__.py
    ├── calculator_tool.py
    └── rag_tool.py
```

---

## ✨ 功能特性

* **动态工具加载**: 代理在启动时会自动扫描 `tools/` 目录并加载所有符合 `BaseTool` 规范的工具。
* **RAG 增强决策**: 在每次决策前，代理会自动查询内部知识库，为 LLM 提供决策所需的上下文。
* **LLM 驱动的决策核心**: 使用 OpenAI 的 `gpt-4o` 等模型作为代理的“大脑”，负责理解意图、选择工具和提取参数。
* **标准化接口**: 所有工具和消息都遵循统一的 MCP 规范，易于扩展。
* **双模式运行**: 支持与代理直接对话的**交互模式**和用于测试的**模拟模式**。
* **安全的密钥管理**: 通过 `.env` 文件管理敏感的 API 密钥，避免硬编码。

---

## 🚀 快速开始

### 1. 环境准备

* 确保你已安装 [Git](https://git-scm.com/)。
* 确保你已安装 [Python 3.10+](https://www.python.org/) 和 [Conda](https://docs.conda.io/en/latest/miniconda.html)。

### 2. 项目设置

1.  **克隆仓库**
    ```bash
    git clone https://github.com/Erber102/D-Agent.git
    cd D-Agent
    ```

2.  **创建并激活 Conda 环境**
    ```bash
    conda create --name agent python=3.10 -y
    conda activate agent
    ```

3.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

4.  **配置 API 密钥**
    * 复制模板文件：
        ```bash
        cp .env.template .env
        ```
    * 编辑新创建的 `.env` 文件，填入你的 [OpenAI API Key](https://platform.openai.com/api-keys)。

5.  **构建 RAG 知识库**
    这是一个一次性的步骤，用于处理 `data/` 目录下的文档并创建向量数据库。
    ```bash
    python build_rag_index.py
    ```
    运行成功后，你会在项目根目录下看到一个 `rag_db` 文件夹。

### 3. 运行代理

项目支持两种运行模式：

* **交互模式 (默认)**
    直接运行 `main.py`，即可开始与代理对话。
    ```bash
    python main.py
    ```
    **示例对话:**
    ```
    > 👤 You: 苹果公司是谁创立的？
    > 🧠 Agent is thinking...
    > ✅ Problem Solved!
    > 🤖 Agent: 根据知识库信息，苹果公司是由史蒂夫·乔布斯、斯蒂夫·沃兹尼亚克和罗恩·韦恩等人创立的。
    ```

* **模拟模式**
    使用 `--simulate` 标志来运行预设的测试用例。
    ```bash
    python main.py --simulate
    ```

---

## 🔧 如何扩展：添加新工具

本框架的设计使得添加新工具变得非常简单：

1.  **创建工具文件**: 在 `tools/` 目录下创建一个新的 Python 文件，例如 `my_new_tool.py`。

2.  **实现工具类**: 在文件中，创建一个继承自 `mcp.interfaces.BaseTool` 的类。

3.  **遵循接口规范**:
    * 在 `__init__` 方法中，定义工具的 `name`, `description` 和 `parameters`。这是给 LLM 看的“工具说明书”。
    * 实现 `execute` 方法，编写工具的核心逻辑。该方法的返回值必须是一个包含 `status` 字段的字典。

    **模板:**
    ```python
    # tools/my_new_tool.py
    from typing import Dict, Any
    from mcp.interfaces import BaseTool

    class MyNewTool(BaseTool):
        def __init__(self):
            name = "my_new_tool"
            description = "这里是新工具的详细描述，告诉LLM它能做什么。"
            parameters = [
                {
                    "name": "param1",
                    "type": "string",
                    "description": "参数1的描述。"
                }
            ]
            super().__init__(name, description, parameters)

        def execute(self, **kwargs: Any) -> Dict[str, Any]:
            param1_value = kwargs.get("param1")
            # ...你的工具逻辑...
            return {"status": "success", "result": "任务完成！"}
    ```

4.  **完成**！下次启动 `main.py` 时，`SmartAgent` 会自动发现并加载你的新工具。

---

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE)。