# 读取单个txt文件
import os
import sys
from langchain_community.document_loaders import TextLoader

# 获取当前脚本文件所在的目录
script_dir = os.path.dirname(__file__)
print(f"📂 当前脚本所在目录：{script_dir}")

# 支持命令行参数：如果提供了文件路径参数，使用它；否则使用默认路径
if len(sys.argv) > 1:
    file_path = sys.argv[1]
    print(f"📝 使用命令行参数指定的文件：{file_path}")
else:
    # 结合相对路径构建完整路径
    file_path = os.path.join(script_dir, '../../90-文档-Data/黑悟空/设定.txt')
    print(f"📝 使用默认文件路径：{file_path}")

# 验证文件是否存在
if not os.path.exists(file_path):
    print(f"❌ 错误：文件不存在 - {file_path}")
    sys.exit(1)

if not os.path.isfile(file_path):
    print(f"❌ 错误：路径不是文件 - {file_path}")
    sys.exit(1)

# 使用 TextLoader 加载，并指定 UTF-8 编码
try:
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    print("✅ 文件加载成功！")
except FileNotFoundError:
    print(f"❌ 错误：文件未找到 - {file_path}")
    sys.exit(1)
except UnicodeDecodeError:
    print(f"❌ 错误：文件编码不是 UTF-8，请检查文件编码")
    sys.exit(1)
except Exception as e:
    print(f"❌ 加载文件时出错：{e}")
    sys.exit(1)

print("=" * 60)
print(f"===> 总文档数：{len(docs)}")
print("=" * 60)

for i, doc in enumerate(docs, start=1):
    source_path = doc.metadata.get("source", "未知路径")
    file_name = os.path.basename(source_path)
    content_preview = doc.page_content[:100]
    content_length = len(doc.page_content)
    
    print(f"📄 第 {i} 份文档")
    print(f"  文件名：{file_name}")
    print(f"  路径：{source_path}")
    print(f"  内容长度：{content_length} 字符")
    print(f"  内容预览：{content_preview}...")
    print("-" * 60)
