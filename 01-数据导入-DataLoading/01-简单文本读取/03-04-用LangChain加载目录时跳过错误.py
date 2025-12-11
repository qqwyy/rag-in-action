from langchain_community.document_loaders import DirectoryLoader, TextLoader
# 加载目录下所有文件，跳过出错文件，因为有些文件是图片，TextLoader 无法加载
import os
# 获取当前脚本文件所在的目录
script_dir = os.path.dirname(__file__)
print(f"获取当前脚本文件所在的目录：{script_dir}") 
# 结合相对路径构建完整路径
data_dir = os.path.join(script_dir, '../../90-文档-Data/黑悟空')

# 加载目录下所有 Markdown 文件
loader = DirectoryLoader(data_dir             ,
                         silent_errors=True   , #如果某个文件加载失败，不会抛出异常，而是跳过该文件，并继续加载其他文件。错误信息通常会被记录（如打印警告）
                         loader_cls=TextLoader,
                         loader_kwargs={"encoding": "utf-8"}  # 👈 关键：指定编码
                        )

docs = loader.load()
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
