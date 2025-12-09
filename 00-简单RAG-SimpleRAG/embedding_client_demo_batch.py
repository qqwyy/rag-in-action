# demo.py

from embedding_client import get_embedding_response, extract_embeddings_from_response
import os
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件中的环境变量     OPENAI_API_BASE=https:xxxx  OPENAI_API_KEY=xxxx
        # 配置参数
MODEL = os.getenv("HTTP_EMBEDDING_API_MODEL")
texts = ["文本内容  测试文案11","晴空万里22"]  #批量 多个文本   请求后返回多个向量
AUTH_KEY = os.getenv("HTTP_EMBEDDING_API_KEY") 
API_URL=os.getenv("HTTP_EMBEDDING_API_BASE")

        # 1. 获取原始响应
print("正在调用 embedding API...")
raw_response = get_embedding_response(model=MODEL,input_text=texts,base_url=API_URL,auth_key=AUTH_KEY )

        # （可选）打印原始响应结构
print("原始响应:", raw_response)

        # 2. 提取向量
vectors = extract_embeddings_from_response(raw_response)

        # 3. 输出结果
print(f"✅ 成功获取 {len(vectors)} 个向量")
if vectors:
    vec1 = vectors[0]
    print(f"第一个向量维度: {len(vec1)}")
    print(f"第一个前5维示例: {[x for x in vec1[:5]]}")

    vec2 = vectors[1]
    print(f"第二个向量维度: {len(vec2)}")
    print(f"第二个前5维示例: {[x for x in vec2[:5]]}")