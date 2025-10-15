import os
import importlib
import inspect
import json
from typing import Dict, Optional, Any

from dotenv import load_dotenv
from openai import OpenAI

from memory import ConversationMemory
from mcp.protocol import MCPMessage
from mcp.interfaces import BaseTool
# 导入 RAGTool 以便特殊处理
from tools.rag_tool import RAGTool

load_dotenv()

class SmartAgent:
    def __init__(self, agent_id="smart_agent_001", tools_package_path="tools"):
        self.agent_id = agent_id
        self.memory = ConversationMemory()

        provider = os.getenv("LLM_PROVIDER", "openai").lower() # 默认为 openai，并转为小写

        api_key = None
        base_url = None
        
        # 根据 provider 的值，从 .env 加载对应的配置
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            self.model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
            # OpenAI 官方服务不需要 base_url
        
        elif provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            self.model_name = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
            base_url = "https://api.deepseek.com" # DeepSeek 的 URL 是固定的
            
        elif provider == "custom":
            api_key = os.getenv("CUSTOM_LLM_API_KEY")
            self.model_name = os.getenv("CUSTOM_LLM_MODEL_NAME")
            base_url = os.getenv("CUSTOM_LLM_BASE_URL") # 用户自己提供 URL
            if not base_url:
                raise ValueError("LLM_PROVIDER is 'custom', but CUSTOM_LLM_BASE_URL is not set!")
        
        else:
            raise ValueError(f"Unsupported LLM provider '{provider}'. Please check your .env file.")

        if not api_key:
            raise ValueError(f"API key for provider '{provider}' is not set in .env file.")
        
        # 3. 使用加载到的配置来初始化 OpenAI 客户端
        # 无论用户选择哪个 provider，我们最终都用同一个 OpenAI 客户端对象
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url # 如果是 OpenAI 官方，base_url 会是 None，客户端会自动处理
        )
        
        print(f"Agent '{self.agent_id}' is online, powered by '{provider}' with model '{self.model_name}'.")

        # self.tools 现在只包含“可选工具”
        self.tools: Dict[str, BaseTool] = {}
        # self.rag_tool 是一个特殊的、自动调用的工具
        self.rag_tool: Optional[RAGTool] = None
        self._load_and_register_tools(tools_package_path)
        
        print(f"  - Automatic RAG tool loaded: {'Yes' if self.rag_tool else 'No'}")
        print(f"  - Selectable tools loaded: {list(self.tools.keys())}")


    def _load_and_register_tools(self, package_path: str):
        """加载所有工具，并对 RAGTool 进行特殊注册。"""
        # (加载逻辑与之前相同)
        if not os.path.exists(package_path): return
        for filename in os.listdir(package_path):
            if filename.endswith("_tool.py"):
                module_name = f"{package_path}.{filename[:-3]}"
                try:
                    module = importlib.import_module(module_name)
                    for _, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, BaseTool) and obj is not BaseTool:
                            tool_instance = obj()
                            # --- 关键改动：分离 RAG 工具 ---
                            if isinstance(tool_instance, RAGTool):
                                self.rag_tool = tool_instance
                            else:
                                self.tools[tool_instance.name] = tool_instance
                except Exception as e:
                    print(f"Error loading tool from {filename}: {e}")

    def _reason(self, goal: str, retrieved_context:str) -> Dict[str, Any]:
        """
        ReAct 循环中的“思考”步骤。
        这个方法将代替旧的 _decide_next_step。
        """
        history = self.memory.format_for_prompt()
        tool_descriptions = [tool.get_mcp_description() for tool in self.tools.values()]
        tools_json_string = json.dumps(tool_descriptions, indent=2, ensure_ascii=False)

        system_prompt = f"""
        # 使命
        你是一个高度智能的自主代理，你的任务是利用一系列强大的工具来完成用户的 [最终目标]。
        你将在一个 "思考 -> 行动 -> 观察" 的循环中工作。你的首要目标是高效、准确地达成最终的成功结果。
        1.  **思考 (Thought)**: 首先，你必须分析 [最终目标] 并回顾完整的 [历史记录]（包括你之前所有的行动和观察结果）。为你的下一步行动制定一个清晰、简洁的计划。你的思考过程需要被明确地表达出来。
        2.  **行动 (Action)**: 根据你的思考，从 [可用工具] 列表中选择最合适的一个工具，并确定其所需的输入参数。
        3.  **观察 (Observation)**: 在你行动后，系统会向你提供该行动的结果。这个新的观察结果将被添加到历史记录中，用于指导你接下来的思考。

        [可用工具]
        {tools_json_string}

        # 关键规则与限制 (必须严格遵守)
        - **基于事实行动**: 你的所有行动都必须**只**基于 [历史记录] 中已有的信息，或用户在 [最终目标] 中明确提供的信息。
        - **杜绝幻觉 (NO HALLUCINATION)**: **严禁**凭空捏造 URL、文件名或任何其他事实。如果你不知道某个信息（比如一个URL），你的首要任务是使用一个工具（比如 `web_surfer_tool`）去**找到它**。不要猜测。
        - **必须处理错误**: 如果 '观察' (Observation) 的结果显示 `status: "error"`，你**必须**在下一步的 '思考' (Thought) 中解决它。
        - **对于 404 Not Found 错误**: 这意味着URL是无效的或已失效。**不要**再次尝试同一个URL。你的下一步应该是使用 `web_surfer_tool` 寻找正确的URL。
        - **对于其他工具错误**: 分析错误信息。如果是参数问题，就在下一步行动中修正 `action_input`。如果某个工具持续失败，请考虑更换策略。
        - **工具协同**: 理解工具如何协同工作。使用 `web_surfer_tool` 来发现信息和URL；使用 `link_extractor_tool` 来探索已知网页以进行导航；使用 `web_browser_tool` 来阅读已确认有效的URL内容；使用 `file_writer_tool` 来保存信息。
        - **任务完成**: 只有当 [最终目标] 被完全、彻底地满足时，你才应该使用特殊行动 `finish`。你在结束前的最后一个 '思考' (thought) 应该为用户提供一个全面的结果总结。

        # 输出格式
        你的回应**必须**是一个**单一、有效**的JSON对象，其结构如下。不要在JSON对象的前后添加任何额外的文本。
        {{
            "thought": "对当前情况的详细分析，以及你为下一步行动制定的计划。解释你*为什么*要选择这个行动。",
            "action": "你将要使用的工具名称（从 [可用工具] 列表中选择），或者 'finish'（如果目标已完成）。",
            "action_input": {{
                "param1": "value1",
                "param2": "value2"
            }}
        }}

        [最终目标]
        {goal}

        [历史记录]
        {history}

        [背景信息]
        {retrieved_context if retrieved_context else "无相关背景信息。"}
        """
        
        # 注意：这里的 user_prompt 留空，因为所有信息都在 system_prompt 里了
        user_prompt = "Please proceed with the next step." 
 
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        decision = json.loads(response.choices[0].message.content)
        return decision
    
    def run(self, goal: str):
        """
        执行 ReAct 循环来完成一个复杂的目标。
        """
        self.memory.clear()
        self.memory.add_message("user", f"My goal is: {goal}")
        
        max_turns = 50 # 设置一个最大循环次数，防止无限循环
        

        for i in range(max_turns):
            print(f"\n--- Turn {i+1}/{max_turns} ---")
            
            # 1. 思考 (Reason)
            print("🤔 Thinking...")
            
            rag_result = self.rag_tool.execute(query=goal.data["query"]) 
            retrieved_context = rag_result.get("retrieved_context", "")

            # 把 context 作为参数传递给 _reason
            decision = self._reason(goal, retrieved_context) 
            thought = decision.get("thought", "No thought provided.")
            action = decision.get("action")
            action_input = decision.get("action_input", {})
            print(f"Thought: {thought}")
            self.memory.add_message("assistant", f"Thought: {thought}")

            # 2. 检查是否完成
            if action == "finish":
                print("✅ Task Finished.")
                self.memory.add_message("assistant", f"Final Answer: {thought}")
                return thought

            # 3. 行动 (Act)
            if action in self.tools:
                print(f"🎬 Acting: Using tool '{action}' with input {action_input}")
                self.memory.add_message("assistant", f"Action: Using tool {action} with input {json.dumps(action_input)}")
                
                # 4. 观察 (Observe)
                tool_result = self.tools[action].execute(**action_input)
                observation = f"Tool {action} returned: {json.dumps(tool_result, ensure_ascii=False)}"
                print(f"👀 Observation: {observation}")
                self.memory.add_message("system", observation)
            else:
                observation = f"Error: Unknown action '{action}'. Available tools are: {list(self.tools.keys())}"
                print(f"👀 Observation: {observation}")
                self.memory.add_message("system", observation)

        print("⚠️ Reached max turns. Stopping.")
        return "The agent reached the maximum number of turns without finishing the task."