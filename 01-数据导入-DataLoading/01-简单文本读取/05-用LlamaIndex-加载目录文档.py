import os
from llama_index.core import SimpleDirectoryReader
# 使用 SimpleDirectoryReader 加载目录中的文件
dir_reader = SimpleDirectoryReader(
     input_dir="90-文档-Data/黑悟空"
     ,required_exts=[".md", ".txt"]
     ,errors="ignore"  # 自动跳过失败文件
)
docs       = dir_reader.load_data()
# 查看加载的文档数量和内容
print("=" * 60)
print(f"===>总文档数：{len(docs)}")  # 输出文档总数
print("=" * 60)
for i, doc in enumerate(docs, start=1):
    file_path = doc.metadata.get("file_path", "未知路径")
    file_name = file_path.split("/")[-1]  # 或用 os.path.basename，但路径可能是正斜杠
    content_preview = doc.text[:100]  # llama_index 中文本内容在 .text 属性中

    print(f"第{i}份文档")
    print(f"  文件名：{file_name}")
    print(f"  路径：{file_path}")
    print(f"  内容预览：{content_preview}")
    print("-" * 60)

# 仅加载某一个特定文件
dir_reader = SimpleDirectoryReader(input_files=["90-文档-Data/黑悟空/设定.txt"])
docs = dir_reader.load_data()

print("=" * 60)
print(f"===>总文档数：{len(docs)}")  # 输出文档总数
print("=" * 60)
for i, doc in enumerate(docs, start=1):
    file_path = doc.metadata.get("file_path", "未知路径")
    file_name = file_path.split("/")[-1]  # 或用 os.path.basename，但路径可能是正斜杠
    content_preview = doc.text[:100]  # llama_index 中文本内容在 .text 属性中

    print(f"第{i}份文档")
    print(f"  文件名：{file_name}")
    print(f"  路径：{file_path}")
    print(f"  内容预览：{content_preview}")
    print("-" * 60)


