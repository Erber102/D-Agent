import os
from typing import Dict, Any
from serpapi import GoogleSearch
import requests
from mcp.interfaces import BaseTool

BROWSER_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
}

import requests
from bs4 import BeautifulSoup
from typing import Dict, Any
from mcp.interfaces import BaseTool

class WebBrowserTool(BaseTool):
    def __init__(self):
        name = "web_browser_tool"
        description = (
            "一个用于获取和读取网页具体文本内容的工具。"
            "当你需要回答基于特定URL内容的问题时，或者当搜索结果提供了一个你需要深入阅读的链接时，应使用此工具。"
        )
        parameters = [
            {
                "name": "url",
                "type": "string",
                "description": "需要访问和读取的完整网页URL。"
            },
            {
                "name": "word_limit",
                "type": "integer",
                "description": "【可选】指定返回内容的最大单词数量，默认为1000，以防止内容过长。"
            }
        ]
        super().__init__(name, description, parameters)

    def _clean_html_content(self, html: str, word_limit: int) -> str:
        """使用 BeautifulSoup 清理HTML并提取文本。"""
        soup = BeautifulSoup(html, 'html.parser')

        # 移除所有脚本和样式元素，因为它们不包含有用信息
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()

        # 获取文本并处理空白
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        clean_text = '\n'.join(line for line in lines if line)

        # 截断文本到指定的单词数量
        words = clean_text.split()
        if len(words) > word_limit:
            truncated_text = ' '.join(words[:word_limit]) + "..."
            return truncated_text
        return clean_text

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        url = kwargs.get("url")
        word_limit = kwargs.get("word_limit", 1000)

        if not url:
            return {"status": "error", "message": "缺少 'url' 参数。"}
        
        try:
            response = requests.get(url, headers=BROWSER_HEADERS, timeout=15, stream=True)
            response.raise_for_status()

            # --- 核心改动：在这里检查文件类型 ---
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type.lower():
                # 如果文件类型不是 HTML 网页，就直接报告并停止
                return {
                    "status": "error", 
                    "message": f"URL 指向的不是一个网页，而是一个 '{content_type}' 类型的文件，无法读取文本内容。"
                }

            # --- 核心改动：更智能的编码处理 ---
            # 确保我们能正确读取内容
            response.encoding = response.apparent_encoding
            html_content = response.text

            content = self._clean_html_content(html_content, word_limit)

            if not content:
                return {"status": "success", "result": "网页内容为空或无法提取有效文本。"}

            return {"status": "success", "result": content}

        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": f"访问URL时发生网络错误: {e}"}
        except Exception as e:
            return {"status": "error", "message": f"处理网页内容时发生未知错误: {e}"}

    