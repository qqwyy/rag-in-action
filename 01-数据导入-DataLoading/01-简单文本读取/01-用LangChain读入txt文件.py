# 读取单个txt文件
import os
from langchain_community.document_loaders import TextLoader
# 获取当前脚本文件所在的目录
script_dir = os.path.dirname(__file__)
print(f"获取当前脚本文件所在的目录：{script_dir}") 
# 结合相对路径构建完整路径
file_dir = os.path.join(script_dir, '../../90-文档-Data/黑悟空/设定.txt')

# 使用 TextLoader 加载，并指定 UTF-8 编码
loader = TextLoader(file_dir,encoding="utf-8")
docs   = loader.load()

print("=" * 60)
print(f"===>总文档数：{len(docs)}")  # 输出文档总数
print("=" * 60)
for i, doc in enumerate(docs, start=1):
    file_path = doc.metadata.get("source", "未知路径")
    file_name = os.path.basename(file_path)
    content_preview = doc.page_content[:100]
    print(f"第{i}份文档")
    print(f"  文件名：{file_name}")
    print(f"  路径：{file_path}")
    print(f"  内容预览：{content_preview}")
    print("-" * 60)
