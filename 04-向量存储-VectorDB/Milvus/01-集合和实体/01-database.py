# 数据库官方文档：https://milvus.io/docs/v2.5.x/manage_databases.md
# 安装 milvus 版本：v2.5.19 https://github.com/milvus-io/milvus/releases/tag/v2.5.19
# 安装依赖：pip install pymilvus==2.5.18   pymilvus和 DB 版本的关系：https://pypi.org/project/pymilvus/2.5.18/
# pip show pymilvus  # 查看当前 SDK 版本
'''
# 安装Milvus服务端 db版本

# 创建工作目录
mkdir milvus && cd milvus

wget https://github.com/milvus-io/milvus/releases/download/v2.5.19/milvus-standalone-docker-compose.yml -O docker-compose.yml

sudo docker compose up -d

Creating milvus-etcd  ... done
Creating milvus-minio ... done
Creating milvus-standalone ... done

验证运行状态：
docker compose ps

停止容器：
docker compose down

如需清除数据（谨慎操作）：
docker compose down -v



官方安装指南：https://milvus.io/docs/install_standalone-docker.md
Milvus GitHub：https://github.com/milvus-io/milvus


管理后端：attu  Attu GitHub: https://github.com/zilliztech/attu
# 创建工作目录  
mkdir milvus-attu && cd milvus-attu

启动：
docker run -d \
  --name attu \
  -p 3000:3000 \
  -e MILVUS_URL=host.docker.internal:19530 \
  zilliz/attu:v2.4.8

停止：docker stop attu
重启：docker start attu

停止并彻底删除容器（谨慎操作）：
docker stop attu && docker rm attu


# 查看正在运行的容器（应看不到 attu）
docker ps
# 查看所有容器（包括已停止的，确认是否还存在）
docker ps -a | grep attu


注：在 Mac 上使用 docker run 单独启动 Attu 时，必须使用 host.docker.internal，而不是 localhost 或 127.0.0.1。
UI 界面连接 也要使用   host.docker.internal：19530

'''

from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件中的环境变量     OPENAI_API_BASE=https:xxxx  OPENAI_API_KEY=xxxx
import os
from pymilvus import MilvusClient, exceptions

# ——————————————
# 1. 连接 Milvus Standalone
# ——————————————
# uri: 协议+地址+端口，默认为 http://localhost:19530  token: "用户名:密码"，默认 root:Milvus
client = MilvusClient(
    uri=os.getenv("MILVUS_URL"),
    token="root:Milvus"
)

    


# ——————————————
# 2. 创建数据库 my_database_1（无额外属性）  API文档：https://milvus.io/api-reference/pymilvus/v2.5.x/MilvusClient/Database/create_database.md
# ——————————————
db_list = client.list_databases()
db1 = "oldwang_database_1"

if db1 in db_list:
    print(f"ℹ️ 数据库 '{db1}' 已存在!!")
else:
    client.create_database(db_name=db1)
    print(f"✅ 数据库 '{db1}' 创建成功")

# ——————————————
# 3. 创建数据库 my_database_2（设置副本数为 3） #https://milvus.io/api-reference/pymilvus/v2.5.x/MilvusClient/Database/create_database.md
# ——————————————

db2 = "oldwang_database_2"

if db2 in db_list:
    print(f"ℹ️ 数据库 '{db2}' 已存在!!")
else:
    client.create_database(db_name=db2,properties={"database.replica.number": 3})
    print(f"✅ 数据库 '{db2}' 创建成功，副本数=3")

# ——————————————
# 4. 列出所有数据库 https://milvus.io/api-reference/pymilvus/v2.5.x/MilvusClient/Database/list_databases.md
# ——————————————
db_list = client.list_databases()
print("当前所有数据库,如下：")
for index, db in enumerate(db_list):
    print(f"{index+1}: {db}")
# for db in db_list:
#     print(db)


# ——————————————
# 5. 查看数据库详情 #https://milvus.io/api-reference/pymilvus/v2.5.x/MilvusClient/Database/describe_database.md
# ——————————————

default_info = client.describe_database(db_name=db2)
print("db2数据库详情：", default_info)

# ——————————————
# 6. 修改 my_database_1 属性：限制最大集合数为 10   https://milvus.io/api-reference/pymilvus/v2.5.x/MilvusClient/Database/alter_database_properties.md
# ——————————————


default_info = client.describe_database(db_name=db1)
print("修改前", default_info)

client.alter_database_properties(
    db_name=db1,
    properties={"database.max.collections": 10}
)

default_info = client.describe_database(db_name=db1)
print("修改后", default_info)
print("✓ 已为 ",db1," 限制最大集合数为 10")


# ——————————————
# 7. 删除 my_database_1 的 max.collections 限制  https://milvus.io/api-reference/pymilvus/v2.5.x/MilvusClient/Database/drop_database_properties.md
# ——————————————

default_info = client.describe_database(db_name=db1)
print("修改前", default_info)

client.drop_database_properties(
    db_name=db1,
    property_keys=["database.max.collections"]
)
default_info = client.describe_database(db_name=db1)
print("修改后", default_info)

print("✓ 已移除 my_database_1 的最大集合数限制")

# ——————————————
# 8. 切换到 my_database_2（后续所有操作都作用于该库） https://milvus.io/api-reference/pymilvus/v2.5.x/MilvusClient/Database/using_database.md
# ——————————————

client.use_database(db_name=db2)
print("✓ 已切换当前数据库为 db2")


# ——————————————
# 9. 删除数据库 db2
#    （注意：如果库内有 Collection，需先 client.drop_collection() 将其清空） https://milvus.io/api-reference/pymilvus/v2.5.x/MilvusClient/Database/drop_database.md
# ——————————————

wyydb3 = "oldwang_database_3"

client.create_database(db_name=wyydb3)
print(f"ℹ️ 数据库 '{wyydb3}' 已创建成功!!")

if db2 in db_list:
    client.drop_database(db_name=wyydb3)
    print(f"ℹ️ 数据库 '{wyydb3}' 已删除成功!!")

