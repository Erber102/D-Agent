import os
from typing import Dict, Any
from mcp.interfaces import BaseTool

# 在文件顶部定义一个默认的输出路径常量，方便管理
DEFAULT_OUTPUT_PATH = "./workplace"

class FileWriterTool(BaseTool):
    def __init__(self):
        name = "file_writer_tool"
        description = "一个用于将内容写入本地文件的工具。可以指定一个可选的文件路径（route），如果不指定，将写入默认的 'output' 文件夹。"

        parameters = [
            {
                "name": "filename",
                "type": "string",
                "description": "要写入的文件的名称，例如 'hello.txt'。"
            },
            {
                "name": "content",
                "type": "string",
                "description": "要写入文件的文本内容。"
            },
            {
                "name": "route",
                "type": "string",
                "description": "【可选】创建文件的路径（文件夹），例如 './documents'。如果省略，将使用默认路径。"
            }
        ]
        super().__init__(name, description, parameters)

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        filename = kwargs.get("filename")
        content = kwargs.get("content")
        # 获取可选的 route 参数
        route = kwargs.get("route") 

        if not filename or content is None:
            return {"status": "error", "message": "缺少 'filename' 或 'content' 参数。"}
        
        # 决定最终的文件路径
        # 如果用户提供了 route，就使用它；否则，使用我们定义的默认路径
        final_path = route if route else DEFAULT_OUTPUT_PATH
        
        try:
            # 5. [关键] 确保目标文件夹存在，如果不存在就创建它
            # exist_ok=True 表示如果文件夹已存在，不要报错
            os.makedirs(final_path, exist_ok=True)

            # 6. 使用 os.path.join() 安全地拼接路径和文件名，以兼容不同操作系统
            full_file_path = os.path.join(final_path, filename)

            with open(full_file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # 在成功信息中返回完整路径，让用户知道文件到底存在了哪里
            return {"status": "success", "result": f"成功将内容写入文件 '{full_file_path}'。"}
        except OSError as e:
            return {"status": "error", "message": f"创建目录 '{final_path}' 时出错: {str(e)}"}
        except Exception as e:
            return {"status": "error", "message": f"写入文件 '{full_file_path}' 时出错: {str(e)}"}