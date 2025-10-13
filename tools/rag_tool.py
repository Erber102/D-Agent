# tools/rag_tool.py
import os
from typing import Dict, Any

from langchain_huggingface import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

PERSIST_DIRECTORY = "./rag_db"

class RAGTool(BaseTool):
    def __init__(self):
        name = "internal_knowledge_retriever"
        description = "从内部知识库检索背景信息的工具。此工具会被自动调用。"
        # 注意：由于此工具是自动调用的，parameters 实际上不会被 LLM 用来“选择”，但保留它是为了接口统一
        parameters = [
            {
                "name": "query",
                "type": "string",
                "description": "用户的原始查询。"
            }
        ]
        super().__init__(name, description, parameters)
        
        if not os.path.exists(PERSIST_DIRECTORY):
            raise FileNotFoundError(f"RAG数据库 '{PERSIST_DIRECTORY}' 未找到。请先运行 'build_rag_index.py'。")
            
        print("RAGTool: 正在加载向量数据库...")
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        self.retriever = db.as_retriever(search_kwargs={"k": 2})
        print("RAGTool: 数据库加载完成，准备就绪。")
        

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """从向量数据库中检索相关信息，只返回文本块内容。"""
        query = kwargs.get("query")
        if not query:
            return {"status": "success", "retrieved_context": "No query provided for retrieval."}
        
        try:
            retrieved_docs = self.retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            if not context:
                return {"status": "success", "retrieved_context": "知识库中没有找到相关信息。"}
            
            return {"status": "success", "retrieved_context": context}
        except Exception as e:
            return {"status": "error", "message": f"检索时出错: {str(e)}"}