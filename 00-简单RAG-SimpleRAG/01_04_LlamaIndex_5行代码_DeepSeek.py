# 第一行代码：导入相关的库
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# pip install llama-index-embeddings-huggingface

#参考文档：https://docs.llamaindex.ai/en/stable/examples/llm/deepseek/
#pip install llama-index-llms-deepseek
from llama_index.llms.deepseek import DeepSeek

from dotenv import load_dotenv
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 加载环境变量
load_dotenv()

# 加载本地嵌入模型
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh")

# 创建 Deepseek LLM
llm = DeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# llm = Vllm(model="facebook/opt-125m")
# llm = Vllm(
#     model="facebook/opt-125m",
#     tensor_parallel_size=4,
#     max_new_tokens=100,
#     vllm_kwargs={"swap_space": 1, "gpu_memory_utilization": 0.5},
# )

# llm = OpenAI(
#     api_base="https://api.deepseek.com/v1",  # DeepSeek 的 API 地址
#     api_key=os.getenv("DEEPSEEK_API_KEY")
#     model="deepseek-chat",
# )

# 第二行代码：加载数据
documents = SimpleDirectoryReader(input_files=["90-文档-Data/黑悟空/设定.txt"]).load_data() 

# 第三行代码：构建索引
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model
)

# 第四行代码：创建问答引擎
query_engine = index.as_query_engine(
    llm=llm
)

# 第五行代码: 开始问答
print(query_engine.query("黑神话悟空中有哪些战斗工具，用中文描述?"))



# 打印组件信息
print("\n===== 配置信息 =====")
print(f"嵌入模型类型: {index._embed_model}")
print(f"向量存储类型: {index.vector_store.class_name}")
print(f"LLM模型: {llm.model}")
# print(f"模型名称: {Settings}")
