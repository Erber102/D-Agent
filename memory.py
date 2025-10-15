from typing import List, Dict, Any

class ConversationMemory:
    """一个简单的内存组件，用于存储对话历史。"""
    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def add_message(self, role: str, content: str):
        """添加一条消息到历史记录。"""
        self.history.append({"role": role, "content": content})

    def get_history(self) -> List[Dict[str, Any]]:
        """获取完整的历史记录。"""
        return self.history
    
    def format_for_prompt(self) -> str:
        """将历史记录格式化为字符串，以便放入 Prompt。"""
        if not self.history:
            return "No history yet."
        
        formatted_history = ""
        for msg in self.history:
            formatted_history += f"{msg['role'].capitalize()}: {msg['content']}\n"
        return formatted_history.strip()

    def clear(self):
        """清空历史记录。"""
        self.history = []