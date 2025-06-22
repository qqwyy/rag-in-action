"""
使用Ollama本地运行大语言模型，无需OpenAI API密钥。

1. 安装Ollama Server:
   - Windows: 访问 https://ollama.com/download 下载安装包
   - Linux/Mac: 运行 curl -fsSL https://ollama.com/install.sh | sh

2. 下载并运行模型:
   - 打开终端，运行以下命令下载模型:
   查看已安装模型：ollama list
   显示：deepseek-r1:1.5b     安装路径一般在：～/.ollama/models
   运行：ollama run deepseek-r1:1.5b

     ollama pull qwen:7b  # 下载通义千问7B模型
     # 或
     ollama pull llama2:7b  # 下载Llama2 7B模型
     # 或
     ollama pull mistral:7b  # 下载Mistral 7B模型

3. 设置环境变量:
   - 在.env文件中添加:
     OLLAMA_MODEL=qwen:7b  # 或其他已下载的模型名称
"""

# 第一行代码：导入相关的库
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama # 需要pip install llama-index-llms-ollama
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 加载本地嵌入模型
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh")

# 创建 Ollama LLM, 默认URL：http://localhost:11434
llm = Ollama(
    model   =os.getenv("OLLAMA_MODEL"),
    base_url=os.getenv("OLLAMA_API_BASE"), 
    request_timeout=30.0, 
    context_window = 4000
)


# 第五行代码: 开始问答
resp  =  llm.complete ("黑神话悟空中有哪些战斗工具?")
print(resp)
