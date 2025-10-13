import numexpr
from typing import Dict, Any
from mcp.interfaces import BaseTool

class CalculatorTool(BaseTool):
    def __init__(self):
        name = "calculator_tool"
        description = "一个用于执行数学计算的工具。适用于任何需要加、减、乘、除等数学运算的场景。"
        parameters = [
            {
                "name": "expression",
                "type": "string",
                "description": "需要计算的数学表达式，例如 '1024 / 4 + 50'。"
            }
        ]
        super().__init__(name, description, parameters)

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """执行计算并返回结果"""
        expression = kwargs.get("expression")
        if not expression:
            return {"status": "error", "message": "缺少 'expression' 参数。"}
        
        try:
            result = numexpr.evaluate(expression).item()
            return {"status": "success", "result": result}
        except Exception as e:
            return {"status": "error", "message": f"计算出错: {str(e)}"}