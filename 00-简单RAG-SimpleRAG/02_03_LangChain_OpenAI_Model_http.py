# 1. 加载文档
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from embedding_client import get_embedding_response, extract_embeddings_from_response
load_dotenv()  # 加载 .env 文件中的环境变量     OPENAI_API_BASE=https:xxxx  OPENAI_API_KEY=xxxx

loader = TextLoader("90-文档-Data/黑悟空/设定2.txt", encoding="utf-8")
docs = loader.load()  # 返回 List[Document]

# 2. 按字符数分块（chunk_size=50 字符，重叠 5 字符）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,  # 默认就是 len（按字符数），可省略
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
)

# 3. 执行分块
all_splits = text_splitter.split_documents(docs)  # 返回 List[Document]

print(f"docs数量：{len(docs)}")
# print(f"docs内容：{docs}")
    

print(f"all_splits数量：{len(all_splits)}")

# 这里会分割成：39片
texts_for_embedding  = [doc.page_content for doc in all_splits]
raw_response         = get_embedding_response(model=os.getenv("HTTP_EMBEDDING_API_MODEL"),input_text=texts_for_embedding,base_url=os.getenv("HTTP_EMBEDDING_API_BASE"),auth_key=os.getenv("HTTP_EMBEDDING_API_KEY") )
raw_response_embeddings_list = extract_embeddings_from_response(raw_response)
# print("向量原始响应:", raw_response)
print(f"向量数量:{len(raw_response_embeddings_list)}")



from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection,MilvusClient, DataType
import numpy as np

client = MilvusClient(
    uri     = os.getenv("MILVUS_URL"),
    db_name = os.getenv("MILVUS_TEST_DB1")
)

COLLECTION_NAME = "search000203_ann_demo"

# 如果集合已存在，则删除
if client.has_collection(COLLECTION_NAME):
    client.drop_collection(COLLECTION_NAME)


# 2. 创建 schema
schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
schema.add_field(field_name="id"       , datatype=DataType.INT64        , is_primary=True)
schema.add_field(field_name="text"     , datatype=DataType.VARCHAR      , max_length=65535 )
schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR , dim=1024        )

# 3. 创建集合
client.create_collection(collection_name=COLLECTION_NAME, schema=schema)

num_vectors = len(raw_response_embeddings_list)
entities = [{"text": texts_for_embedding[i], "embedding": raw_response_embeddings_list[i]} for i in range(num_vectors)]


client.insert(collection_name=COLLECTION_NAME, data=entities)

# 5. 创建索引
index_params = MilvusClient.prepare_index_params()
index_params.add_index(
    field_name="embedding",
    metric_type="L2",
    index_type="FLAT",
    index_name="embedding_index",
    params={}
)
client.create_index(
    collection_name=COLLECTION_NAME,
    index_params=index_params,
    sync=True
)

# 6. 加载集合
client.load_collection(collection_name=COLLECTION_NAME)



# 查询（用 query vector）
# query_vec = [np.random.rand(dim).tolist()]
# 5. 构建用户查询
question = "用中文回答 ，黑神话悟空中有哪些战斗工具?"
question_raw_response = get_embedding_response(model=os.getenv("HTTP_EMBEDDING_API_MODEL"),input_text=question,base_url=os.getenv("HTTP_EMBEDDING_API_BASE"),auth_key=os.getenv("HTTP_EMBEDDING_API_KEY") )
query_vec             = extract_embeddings_from_response(question_raw_response)



search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}


results = client.search(
    collection_name=COLLECTION_NAME,
    data= query_vec,
    anns_field="embedding",
    limit=3,
    search_params={"metric_type": "L2"},
    output_fields=["text"]
)


retrieved_docs = []
for hits in results:          # results 是一个列表，每个元素对应一个查询向量的结果
    for hit in hits:          # hits 是 TopK 结果（如 limit=2，则最多2个）
        text = hit.entity.get("text")  # 获取存储的原文
        retrieved_docs.append(text)

# 现在 texts 是一个字符串列表
print(f"找到的相似文档数量：{len(retrieved_docs)}") 
print(f"找到的相似文档明细：{retrieved_docs}")  # 例如: ["黑悟空手持金箍棒", "游戏场景包括花果山"]

# 6. 在向量存储中搜索相关文档，并准备上下文内容
# retrieved_docs = vector_store.similarity_search(question, k=3)
# query_vec = [0.5]*1024  # 查询向量（也由外部生成）
# retrieved_docs = vector_store.similarity_search_by_vector(query_vec, k=1)

docs_content = "\n\n".join(retrieved_docs)

# 7. 构建提示模板
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""
                基于以下上下文，回答问题。如果上下文中没有相关信息，
                请说"我无法从提供的上下文中找到相关信息"。
                上下文: {context}
                问题: {question}
                回答:"""
                )

# 8. 使用大语言模型生成答案
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo")
answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(f"大模型回答内容==={answer.content}")








# # 1. 预计算好的数据
# texts = ["文本A", "文本B"]
# embeddings_list = [[0.1]*1024, [0.9]*1024]  # 你的向量
# # 2. 构建向量库
# db = InMemoryVectorStore.from_texts(texts,embeddings=embeddings_list)

# # 3. 搜索（不调用任何 embedding 模型）
# query_vec = [0.5]*1024  # 查询向量（也由外部生成）
# results = db.similarity_search_by_vector(query_vec, k=1)
# print(results[0].page_content)