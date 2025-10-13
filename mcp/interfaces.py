from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseTool(ABC):
    """
    所有工具类必须遵守的接口规范 (Interface Specification)。
    """
    def __init__(self, name: str, description: str, parameters: List[Dict[str, Any]]):
        self.name = name
        self.description = description
        self.parameters = parameters

    def get_mcp_description(self) -> Dict[str, Any]:
        """
        生成该工具的 JSON 描述，用于给 LLM '看'。
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }

    @abstractmethod
    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """
        工具的核心执行逻辑。子类必须实现此方法。
        """
        pass