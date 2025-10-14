import os
import importlib
import inspect
import json
from typing import Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI

from mcp.protocol import MCPMessage
from mcp.interfaces import BaseTool
# 导入 RAGTool 以便特殊处理
from tools.rag_tool import RAGTool

load_dotenv()

class SmartAgent:
    def __init__(self, agent_id="smart_agent_001", tools_package_path="tools"):
        self.agent_id = agent_id
        provider = os.getenv("LLM_PROVIDER", "openai").lower() # 默认为 openai，并转为小写

        if provider == "openai":
            print("Initializing agent with OpenAI...")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("LLM_PROVIDER is 'openai', but OPENAI_API_KEY is not set!")
            
            self.model_name = "gpt-4o" # 或者从 .env 读取
            self.client = OpenAI(api_key=api_key)

        elif provider == "deepseek":
            print("Initializing agent with DeepSeek...")
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("LLM_PROVIDER is 'deepseek', but DEEPSEEK_API_KEY is not set!")

            self.model_name = "deepseek-chat" # 或者从 .env 读取
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
            
        else:
            # 如果用户在 .env 里写了不支持的 provider，就报错
            raise ValueError(f"Unsupported LLM provider '{provider}' configured in .env file.")

        print(f"Agent '{self.agent_id}' is online, powered by {provider.capitalize()} model '{self.model_name}'.")

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

    def _generate_direct_answer(self, query: str) -> str:
        """发起一个简单的、无工具约束的 LLM 调用来生成最终答案。"""
        print(f"[{self.agent_id}] Generating final answer using general knowledge...")
        try:
            # 这里的 Prompt 非常简单
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"生成答案时出错: {e}"

    def handle_message(self, incoming_message: MCPMessage):
        """
        核心流程：检索 -> 决策 -> 行动
        """
        print(f"\n[{self.agent_id}] --- New Task Received ---")

        # =======================================================
        # 步骤 1: 准备用户的查询
        # =======================================================
        user_query = incoming_message.data.get("query", "")
        if not user_query:
            user_query = json.dumps(incoming_message.data)

        # =======================================================
        # 步骤 2: 检索 (在决策前执行！)
        # =======================================================
        retrieved_context = ""
        if self.rag_tool:
            print(f"[{self.agent_id}] Pre-processing: Automatically querying knowledge base...")
            rag_result = self.rag_tool.execute(query=user_query)
            if rag_result.get("status") == "success":
                retrieved_context = rag_result.get("retrieved_context", "")
                print(f"[{self.agent_id}] Retrieval finished. Context is ready.")
            else:
                print(f"[{self.agent_id}] Retrieval error: {rag_result.get('message')}")

        # =======================================================
        # 步骤 3: 决策 (带着检索到的上下文去思考)
        # =======================================================
        print(f"[{self.agent_id}] Thinking with context...")
        decision = self._decide_next_step(incoming_message.task, incoming_message.data, retrieved_context)
        
        action = decision.get("action")
        llm_thought = decision.get("reasoning", "No reasoning provided.")
        print(f"[{self.agent_id}] LLM Router Decision: Action is '{action}'. Reason: '{llm_thought}'")

        # =======================================================
        # 步骤 4: 行动 (根据清晰的决策来执行)
        # =======================================================
        final_thought = f"Router thought: {llm_thought}"

        if action == "use_tool":
            chosen_tool_name = decision.get("tool_name")
            arguments = decision.get("arguments", {})
            if chosen_tool_name in self.tools:
                response_data = self.tools[chosen_tool_name].execute(**arguments)
            else:
                response_data = {"status": "error", "message": f"Router decided to use tool '{chosen_tool_name}', but it was not found."}

        elif action == "use_knowledge_base":
            # 知识库有答案，需要发起第二次 LLM 调用来组织语言
            final_query = f"Based on the following context:\n---\n{retrieved_context}\n---\nPlease provide a comprehensive answer to the user's question: {user_query}"
            final_answer = self._generate_direct_answer(final_query)
            response_data = {"status": "success", "result": final_answer}

        elif action == "use_general_knowledge":
            # 知识库没有答案，但可以用通用知识回答
            final_answer = self._generate_direct_answer(user_query)
            response_data = {"status": "success", "result": final_answer}
            
        else: # action == "cannot_answer" or any other case
            response_data = {"status": "rejected", "result": "I've analyzed the request, but I cannot handle it with my current capabilities."}
        
        # =======================================================
        # 步骤 5: 响应
        # =======================================================
        response_msg = MCPMessage(
            sender_id=self.agent_id,
            receiver_id=incoming_message.sender_id,
            task=f"response_to:{incoming_message.task}",
            data=response_data,
            thought=final_thought
        )
        
        print(f"[{self.agent_id}] --- Task Finished ---")
        return response_msg

    def _decide_next_step(self, user_task: str, user_data: dict, retrieved_context: str):
        tool_descriptions = [tool.get_mcp_description() for tool in self.tools.values()]
        tools_json_string = json.dumps(tool_descriptions, indent=2, ensure_ascii=False)

        # 简化后的 Prompt，只做决策，不生成答案
        system_prompt = f"""
        你是一个智能 Agent 的路由决策核心。你的任务是分析用户请求、背景信息和可用工具，然后决定下一步的行动路径。

        行动路径选项:
        1. "use_knowledge_base": 如果背景信息非常相关且足以回答问题。
        2. "use_general_knowledge": 如果背景信息不相关，但这是一个可以用通用知识回答的常识、编程或创意性问题。
        3. "use_tool": 如果用户的请求明确需要一个工具来执行（例如计算）。
        4. "cannot_answer": 如果以上都不适用。

        [背景信息]
        {retrieved_context if retrieved_context else "无相关背景信息。"}

        [可用工具列表]
        {tools_json_string if tools_json_string else "无可用工具。"}

        你的输出必须是一个严格的 JSON 对象，格式如下:
        {{
            "action": "你选择的行动路径",
            "tool_name": "如果 action 是 'use_tool'，这里是工具名称",
            "arguments": {{...}},
            "reasoning": "你的决策理由。"
        }}
        """
        user_request = f"用户任务: '{user_task}', 相关数据: {json.dumps(user_data, ensure_ascii=False)}"

        # 这里的 LLM 调用逻辑不变
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_request}
            ],
            response_format={"type": "json_object"}
        )
        decision = json.loads(response.choices[0].message.content)
        return decision