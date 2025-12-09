from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding  # pip install llama-index-embeddings-openai
import os

from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件中的环境变量     OPENAI_API_BASE=https:xxxx  OPENAI_API_KEY=xxxx

# 加载本地嵌入模型
# import os
# os.environ['HF_ENDPOINT']= 'https://hf-mirror.com' # 如果万一被屏蔽，可以设置镜像

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用并行 tokenization
# 配置远程 embedding（使用 bge-large-zh-v1.5）
embed_model = OpenAIEmbedding(
    model   =os.getenv("EMBEDDING_API_MODEL") ,   # 与服务端 model 字段一致
    api_key =os.getenv("EMBEDDING_API_KEY") ,     # api key
    api_base=os.getenv("EMBEDDING_API_BASE")      # 你的 BGE 服务地址
)


# 加载数据
documents = SimpleDirectoryReader(input_files=["90-文档-Data/黑悟空/设定.txt"]).load_data()

# 构建索引
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model
)

# 创建问答引擎
query_engine = index.as_query_engine()

# 开始问答
print(query_engine.query("黑神话悟空中有哪些战斗工具?"))


from llama_index.core import Settings

# 打印组件信息
print("===== 默认配置 =====")
print(f"默认嵌入模型类型: {index._embed_model}")
print(f"默认向量存储类型: {index.vector_store.class_name}")
# print(f"默认向量db数据: {index._vector_store}")
print(f"默认LLM模型: {Settings.llm}")
# print(f"模型名称: {Settings}")