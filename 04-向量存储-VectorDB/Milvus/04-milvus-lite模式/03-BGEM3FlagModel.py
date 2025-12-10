from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件中的环境变量     OPENAI_API_BASE=https:xxxx  OPENAI_API_KEY=xxxx
import os
from FlagEmbedding import BGEM3FlagModel

# 初始化模型  首次运行会自动从 Hugging Face 下载模型（约 2.2GB）
# pip install -U FlagEmbedding
# https://milvus.io/docs/zh/embed-with-bgm-m3.md     bge-m3有2个G  换一个轻量的       bge-small-zh-v1.5  bge-large-zh-v1.5
model = BGEM3FlagModel(
    'BAAI/bge-m3',
    use_fp16=False,  # CPU 必须为 False；GPU 可设为 True
    device='cpu'     # 或 'cuda:0'
)

# 输入文本
texts = ["孙悟空大战白骨精", "人工智能改变世界"]

# 编码（只取稠密向量）
embeddings = model.encode(
    texts,
    batch_size=12,
    max_length=512,
    return_dense=True,
    return_sparse=False,      # 暂时不取稀疏
    return_colbert_vecs=False
)

#放开：如果长度超过一定阈值（默认是 前 3 个 + 后 3 个），NumPy 会自动用 ... 省略中间部分
import numpy as np
np.set_printoptions(threshold=np.inf)  

# 获取稠密向量（numpy array）
dense_vectors = embeddings['dense_vecs']  # shape: (n, 1024)
print("第一个向量:",dense_vectors[0])
print("第二个向量:",dense_vectors[1])
# 获取维度
dim = dense_vectors.shape[1]  # → 1024

print(f"向量维度: {dim}")
print(f"向量数量: {len(dense_vectors)}")
print(f"第一个向量前5维: {dense_vectors[0][:5]}")