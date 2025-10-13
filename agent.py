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
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY 未在 .env 文件中设置！")
        self.client = OpenAI(api_key=openai_api_key)

        # self.tools 现在只包含“可选工具”
        self.tools: Dict[str, BaseTool] = {}
        # self.rag_tool 是一个特殊的、自动调用的工具
        self.rag_tool: Optional[RAGTool] = None
        self._load_and_register_tools(tools_package_path)
        
        print(f"Agent '{self.agent_id}' is online.")
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

    def handle_message(self, incoming_message: MCPMessage):
        """
        新的核心流程：检索 -> 思考 -> 行动
        """
        print(f"\n[{self.agent_id}] --- New Task Received ---")
        
        # 步骤 1: 准备用户的查询
        user_query = incoming_message.data.get("query", "")
        if not user_query:
            # 如果是其他任务类型（如 'calculate'），也用其数据作为查询内容
            user_query = json.dumps(incoming_message.data)

        # 步骤 2: 自动调用 RAG 工具进行检索
        retrieved_context = ""
        if self.rag_tool:
            print(f"[{self.agent_id}] Pre-processing: Automatically querying knowledge base...")
            rag_result = self.rag_tool.execute(query=user_query)
            if rag_result.get("status") == "success":
                retrieved_context = rag_result.get("retrieved_context", "")
                print(f"[{self.agent_id}] Retrieval finished. Context found.")
            else:
                print(f"[{self.agent_id}] Retrieval error: {rag_result.get('message')}")

        # 步骤 3: 带着上下文信息，让 LLM 思考和决策
        print(f"[{self.agent_id}] Thinking with context...")
        decision = self._decide_next_step(incoming_message.task, incoming_message.data, retrieved_context)
        llm_thought = decision.get("reasoning", "No reasoning provided.")
        chosen_tool_name = decision.get("tool_name")
        arguments = decision.get("arguments", {})
        print(f"[{self.agent_id}] LLM Decision: Use tool '{chosen_tool_name}'. Reason: '{llm_thought}'")

        # 步骤 4: 行动 (与之前相同)
        # ... (这部分代码无需改动) ...
        if chosen_tool_name in self.tools:
            tool_to_execute = self.tools[chosen_tool_name]
            execution_result = tool_to_execute.execute(**arguments)
            response_data = execution_result
            final_thought = f"I executed the '{chosen_tool_name}' tool. LLM thought: {llm_thought}"
        else:
            # 如果LLM决定不需要工具，我们可以在这里直接利用上下文来回答
            if chosen_tool_name == "respond_directly" and retrieved_context:
                 response_data = {"status": "success", "result": decision.get("direct_answer", "I found some information but couldn't form a final answer.")}
            else:
                response_data = {"status": "rejected", "result": "I don't have a tool to handle this request."}
            final_thought = f"I concluded my action based on the context. LLM thought: {llm_thought}"

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
        """
        新的决策函数，Prompt 中加入了检索到的上下文。
        """
        tool_descriptions = [tool.get_mcp_description() for tool in self.tools.values()]
        tools_json_string = json.dumps(tool_descriptions, indent=2, ensure_ascii=False)

        system_prompt = f"""
        你是一个智能 Agent 的决策核心。在做决定前，你已经从知识库中检索到了一些相关背景信息。
        你的任务是：
        1. 分析用户的请求和背景信息。
        2. 如果背景信息足以直接回答用户问题，请选择名为 "respond_directly" 的特殊工具，并在 "direct_answer" 字段中提供答案。
        3. 如果需要进行额外操作（如计算），请从可用工具列表中选择一个。
        4. 如果两者都不适用，请选择 "none"。

        [背景信息]
        {retrieved_context if retrieved_context else "无相关背景信息。"}

        [可用工具列表 (除直接回答外)]
        {tools_json_string if tools_json_string else "无其他可用工具。"}

        你的输出必须是一个严格的 JSON 对象，格式如下:
        {{"tool_name": "工具名称", "arguments": {{...}}, "reasoning": "你的决策理由。", "direct_answer": "如果直接回答，请在此提供答案"}}
        """
        
        user_request = f"用户任务: '{user_task}', 相关数据: {json.dumps(user_data, ensure_ascii=False)}"

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_request}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            # ... (错误处理不变) ...
            return {"tool_name": "none", "arguments": {}, "reasoning": f"LLM 调用失败: {e}"}