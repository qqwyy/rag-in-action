# https://github.com/milvus-io/milvus-lite
# https://github.com/milvus-io/pymilvus
# Lite 目前支持以下环境：
# Ubuntu >= 20.04（x86_64 和 arm64）
# MacOS >= 11.0（Apple Silicon M1/M2 和 x86_64）
# 注意：目前尚不支持 Windows 系统。
# 安装：pip install -U pymilvus[milvus-lite]


from pymilvus import MilvusClient
import numpy as np

client = MilvusClient("./milvus_demo.db")
client.create_collection(
    collection_name="demo_collection",
    dimension=384  # The vectors we will use in this demo has 384 dimensions
)

# Text strings to search from.
docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]
# For illustration, here we use fake vectors with random numbers (384 dimension).

vectors = [[ np.random.uniform(-1, 1) for _ in range(384) ] for _ in range(len(docs)) ]
data = [ {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"} for i in range(len(vectors)) ]
res = client.insert(
    collection_name="demo_collection",
    data=data
)

# This will exclude any text in "history" subject despite close to the query vector.
res = client.search(
    collection_name="demo_collection",
    data=[vectors[0]],
    filter="subject == 'history'",
    limit=2,
    output_fields=["text", "subject"],
)
print(f"查询1：{res}")

# a query that retrieves all entities matching filter expressions.
res = client.query(
    collection_name="demo_collection",
    filter="subject == 'history'",
    output_fields=["text", "subject"],
)
print(f"查询2：{res}")

# delete
res = client.delete(
    collection_name="demo_collection",
    filter="subject == 'history'",
)
print(f"删除：{res}")