#todo
import pandas as pd
import numpy as np
import requests
from sklearn.cluster import KMeans
from pymilvus.model.dense import JinaEmbeddingFunction #pip install "pymilvus[model]"

from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件中的环境变量     OPENAI_API_BASE=https:xxxx  OPENAI_API_KEY=xxxx
import os

# 2. 读取游戏描述数据
df = pd.read_csv("90-文档-Data/灭神纪/游戏描述.csv")
texts = df['description'].tolist()

# 3. 获取文本嵌入  JinaEmbeddingFunction 不在本地运行模型，它是一个 API 客户端。
jina_ef = JinaEmbeddingFunction(
    model_name="jina-embeddings-v3", # Defaults to `jina-embeddings-v3`
    api_key=JINAAI_API_KEY, # Provide your Jina AI API key
    task="retrieval.passage", # Specify the task
    dimensions=1024, # Defaults to 1024
)


docs_embeddings = jina_ef.encode_documents(texts)

# Print embeddings
print("Embeddings:", docs_embeddings)
# Print dimension and shape of embeddings
print("Dim:", jina_ef.dim, docs_embeddings[0].shape)
