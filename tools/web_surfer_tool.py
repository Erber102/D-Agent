# tools/web_surfer_tool.py
import os
from typing import Dict, Any
from serpapi import GoogleSearch
from mcp.interfaces import BaseTool

class WebSurferTool(BaseTool):
    def __init__(self):
        name = "web_surfer_tool"
        description = (
            "一个用于在互联网上执行关键词搜索的工具。"
            "当你需要查找信息但没有具体的URL时，或者当一个URL无效(如404错误)需要寻找正确页面时，应使用此工具。"
        )
        parameters = [
            {
                "name": "search_query",
                "type": "string",
                "description": "需要在搜索引擎中查询的关键词或问题。"
            }
        ]
        super().__init__(name, description, parameters)

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        search_query = kwargs.get("search_query")
        if not search_query:
            return {"status": "error", "message": "缺少 'search_query' 参数。"}

        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return {"status": "error", "message": "未在 .env 文件中找到 SERPAPI_API_KEY。"}

        try:
            params = {
                "engine": "google",
                "q": search_query,
                "api_key": api_key
            }
            search = GoogleSearch(params)
            results = search.get_dict()

            # 提取前3个有用的有机结果
            organic_results = results.get("organic_results", [])
            if not organic_results:
                return {"status": "success", "result": "搜索引擎没有返回有效结果。"}

            summary = []
            for result in organic_results[:3]:
                title = result.get("title", "N/A")
                link = result.get("link", "N/A")
                snippet = result.get("snippet", "N/A").replace("\n", " ")
                summary.append(f"Title: {title}\nLink: {link}\nSnippet: {snippet}")

            return {"status": "success", "result": "\n\n".join(summary)}

        except Exception as e:
            return {"status": "error", "message": f"执行网络搜索时出错: {e}"}