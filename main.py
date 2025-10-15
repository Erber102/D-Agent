import json
import argparse  # 导入命令行参数解析库

from agent import SmartAgent
from mcp.protocol import MCPMessage

def run_simulation_case(agent: SmartAgent, user_task: str, user_data: dict):
    """
    运行一个独立的模拟测试用例并打印结果。
    （这是我们之前的 run_simulation 函数，稍微改了名字以示区分）
    """
    print("="*60)
    print(f"SIMULATION CASE: User requests task '{user_task}'")
    
    user_request = MCPMessage(
        sender_id="simulation_client",
        receiver_id=agent.agent_id,
        task=user_task,
        data=user_data
    )
    
    response = agent.handle_message(user_request)
    
    print("\n--- Final MCP Response from Agent ---")
    print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
    print("="*60 + "\n")

def run_all_simulations(agent: SmartAgent):
    """
    一个“套件”，用于运行所有预定义的模拟用例。
    """
    print("🚀 Starting Simulation Suite...")
    
    # ... 保留 calculator_tool 和失败的测试用例 ...
    run_simulation_case(
        agent=agent,
        user_task="calculate", 
        user_data={"expression": "3.14 * 10**2"}
    )
    
    # 一个知识库查询
    run_simulation_case(
        agent=agent,
        user_task="ask_about_company",
        user_data={"query": "苹果公司是谁创立的？"}
    )
    
    run_simulation_case(
        agent=agent,
        user_task="get_weather_forecast",
        user_data={"location": "London", "days": 3}
    )
    
    print("✅ Simulation Suite Finished.")


def start_interactive_mode(agent: SmartAgent):
    """
    启动交互模式，接收用户输入并尝试解决问题。
    """
    print("\n🚀 Starting Interactive Mode with SmartAgent...")
    print("Agent is ready. Ask me anything!")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        try:
            # 1. 接受用户的询问
            user_input = input("\n👤 You: ")

            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 Agent shutting down. Goodbye!")
                break
            
            if not user_input:
                continue

            # 2. 将用户输入封装成 MCP 消息，并调用 Agent 尝试解决
            print("🧠 Agent is thinking...")
            user_request = MCPMessage(
                sender_id="user_interactive_client",
                receiver_id=agent.agent_id,
                # 我们使用一个通用的 'user_query' 任务，让 LLM 自行解析意图
                task="user_query", 
                data={"query": user_input}
            )
            final_result = agent.run(user_request)
            print("\n--- Final Result ---")
            print(f"🤖 Agent: {final_result}")

        except KeyboardInterrupt:
            print("\n👋 Agent shutting down. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred in the main loop: {e}")


if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Run the SmartAgent in interactive or simulation mode.")
    parser.add_argument(
        '--simulate',
        action='store_true',  # 当出现 --simulate 参数时，其值为 True
        help="Run the predefined simulation suite instead of interactive mode."
    )
    args = parser.parse_args()

    # 初始化 Agent (无论哪种模式都需要)
    print("Initializing SmartAgent, please wait...")
    my_agent = SmartAgent()
    
    # 根据命令行参数决定运行哪个模式
    if args.simulate:
        run_all_simulations(my_agent)
    else:
        start_interactive_mode(my_agent)