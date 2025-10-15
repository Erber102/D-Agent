# tools/link_extractor_tool.py
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, Set
from urllib.parse import urljoin
from mcp.interfaces import BaseTool

class LinkExtractorTool(BaseTool):
    def __init__(self):
        name = "link_extractor_tool"
        description = (
            "一个用于从指定URL的网页内容中提取所有唯一超链接的工具。"
            "当你需要在一个网页上寻找导航链接或下一步的跳转目标时，应使用此工具。"
        )
        parameters = [
            {
                "name": "url",
                "type": "string",
                "description": "需要提取链接的完整网页URL。"
            }
        ]
        super().__init__(name, description, parameters)

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        url = kwargs.get("url")
        if not url:
            return {"status": "error", "message": "缺少 'url' 参数。"}

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            
            unique_links: Set[str] = set()
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                # 将相对路径 (如 /docs/quickstart) 转换为绝对路径
                absolute_url = urljoin(url, href)
                # 简单的过滤，去掉无用的锚点和脚本链接
                if absolute_url.startswith('http') and '#' not in absolute_url:
                    unique_links.add(absolute_url)

            if not unique_links:
                return {"status": "success", "result": "在页面上没有找到有效的外部链接。"}

            # 为了不让结果过长，可以限制返回的数量
            link_list = list(unique_links)[:15]

            return {"status": "success", "result": "\n".join(link_list)}

        except Exception as e:
            return {"status": "error", "message": f"提取链接时出错: {e}"}