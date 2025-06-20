"""
注意：运行此代码前，请确保已在环境变量中设置OpenAI API密钥。
在Linux/Mac系统中，可以通过以下命令设置： export OPENAI_API_KEY='sk-proxxxxxx'
在Windows  系统中，可以通过以下命令设置： set OPENAI_API_KEY=your-api-key
"""
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件中的环境变量     OPENAI_API_BASE=https:xxxx  OPENAI_API_KEY=xxxx

import os
OPENAI_API_BASE     = os.getenv("OPENAI_API_BASE")
print(f"请求的API为: {OPENAI_API_BASE}")


# 第一行代码：导入相关的库  pip install llama-index
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader ,ServiceContext
# 第二行代码：加载数据
documents = SimpleDirectoryReader(input_files=["90-文档-Data/黑悟空/设定.txt"]).load_data() 
# 第三行代码：构建索引  
index = VectorStoreIndex.from_documents(documents)
# 第四行代码：创建问答引擎
query_engine = index.as_query_engine()
# 第五行代码: 开始问答
print(query_engine.query("用中文回答 ，黑神话悟空中有哪些战斗工具?"))



from llama_index.core import Settings

# 打印组件信息
print("===== 默认配置 =====")
print(f"默认嵌入模型类型: {index._embed_model}")
print(f"默认向量存储类型: {index.vector_store.class_name}")
# print(f"默认向量db数据: {index._vector_store}")
print(f"默认LLM模型: {Settings.llm}")
# print(f"模型名称: {Settings}")


"""
在 LlamaIndex 中，嵌入模型（Embedding Model）、向量存储（Vector Store） 和 推理模型（LLM） 的默认配置取决于你的环境和依赖安装情况。以下是详细说明：
一、默认组件分类
1. 嵌入模型（Embedding Model）
优先级 1：若安装了 openai 库且设置了 OPENAI_API_KEY，则默认使用： text-embedding-ada-002（OpenAI 提供的文本嵌入模型）。
优先级 2：否则使用开源模型：sentence-transformers/all-MiniLM-L6-v2（轻量级语义嵌入模型）。
2. 向量存储（Vector Store）
默认选择：
SimpleVectorStore（LlamaIndex 内置的纯内存存储方案）。
无需额外依赖，数据存储在 Python 列表中，适合小规模数据和快速原型。
3. 推理模型（LLM）
优先级 1：若安装了 openai 库且设置了 OPENAI_API_KEY，则默认使用： gpt-3.5-turbo（OpenAI 的聊天模型）。
优先级 2：若未安装 OpenAI 依赖，则无默认模型，需手动配置（如使用开源 LLM）。
"""