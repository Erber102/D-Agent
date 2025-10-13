from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 定义数据库的持久化路径
PERSIST_DIRECTORY = "./rag_db"
DATA_PATH = "./data"

def build_index():
    print("开始构建向量索引...")
    
    # 1. 加载文档
    print(f"从 '{DATA_PATH}' 加载文档...")
    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", show_progress=True)
    documents = loader.load()
    if not documents:
        print("错误：在 data 目录中没有找到任何 .txt 文件。")
        return

    # 2. 文本分割
    print("分割文档...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)

    # 3. 初始化嵌入模型
    print("初始化嵌入模型 (all-MiniLM-L6-v2)...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. 创建并持久化向量数据库
    print(f"创建向量数据库并将其持久化到 '{PERSIST_DIRECTORY}'...")
    db = Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIRECTORY)
    db.persist()
    
    print("\n索引构建完成！")
    print(f"数据库已保存在 '{PERSIST_DIRECTORY}' 文件夹中。")

if __name__ == "__main__":
    build_index()