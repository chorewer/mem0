import os
import sys
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from mem0.vector_stores.opensearch import OpenSearchDB
from mem0.embeddings.volce_embedding import VolceEmbedding
from mem0.configs.embeddings.base import BaseEmbedderConfig

# 加载环境变量
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

def debug_vector_insert():
    print("🔍 调试向量插入问题")
    
    # 配置
    host = os.getenv("HOST_OPENSEARCH")
    port = os.getenv("PORT_OPENSEARCH")
    user = os.getenv("USER_OPENSEARCH")
    password = os.getenv("PASSWORD_OPENSEARCH")
    index = "debug_test_index"
    
    volce_api_key = os.getenv("VOLCE_EMBED_API_KEY", "")
    volce_endpoint = os.getenv("VOLCE_EMBEDDING_ENDPOINT", "")
    volce_model = os.getenv("VOLCE_MODEL", "intfloat/multilingual-e5-small")
    
    print(f"OpenSearch: {host}:{port}")
    print(f"索引: {index}")
    
    # 初始化服务
    config = BaseEmbedderConfig(
        volce_api_key=volce_api_key,
        volce_endpoint=volce_endpoint,
        volce_model=volce_model
    )
    embedder = VolceEmbedding(config=config)
    
    db = OpenSearchDB(
        host=host,
        port=port,
        user=user,
        password=password,
        verify_certs=False,
        use_ssl=True,
        collection_name=index,
        embedding_model_dims=384,
    )
    
    # 步骤1: 生成向量
    print("\n=== 步骤1: 生成向量 ===")
    test_text = "这是一个测试文本用于调试"
    print(f"测试文本: {test_text}")
    
    embedding = embedder.embed(test_text)
    print(f"生成的向量:")
    print(f"  类型: {type(embedding)}")
    print(f"  维度: {len(embedding) if embedding else 'None'}")
    if embedding:
        print(f"  前3维: {embedding[:3]}")
        print(f"  是否包含None: {None in embedding}")
        print(f"  所有值都是数字: {all(isinstance(x, (int, float)) for x in embedding)}")
    
    # 步骤2: 准备数据
    print("\n=== 步骤2: 准备插入数据 ===")
    vectors = [embedding]
    payloads = [{"text": test_text, "category": "debug"}]
    ids = ["debug_1"]
    
    print(f"vectors类型: {type(vectors)}")
    print(f"vectors长度: {len(vectors)}")
    print(f"第一个向量类型: {type(vectors[0])}")
    print(f"第一个向量维度: {len(vectors[0]) if vectors[0] else 'None'}")
    
    # 步骤3: 创建索引
    print("\n=== 步骤3: 创建索引 ===")
    try:
        db.create_index()
        print("✅ 索引创建成功")
    except Exception as e:
        print(f"❌ 索引创建失败: {e}")
        return
    
    # 步骤4: 手动构造插入体验插入过程
    print("\n=== 步骤4: 手动模拟插入过程 ===")
    try:
        for i, (vec, id_) in enumerate(zip(vectors, ids)):
            print(f"\n处理向量 {i+1}:")
            print(f"  向量ID: {id_}")
            print(f"  向量类型: {type(vec)}")
            print(f"  向量维度: {len(vec) if vec else 'None'}")
            print(f"  向量是否为None: {vec is None}")
            
            if vec is not None:
                print(f"  向量前3维: {vec[:3]}")
                print(f"  向量包含None值: {None in vec}")
                
                # 构造body
                body = {
                    "vector_field": vec,
                    "payload": payloads[i],
                    "id": id_,
                }
                
                print(f"  插入体结构: {list(body.keys())}")
                print(f"  vector_field类型: {type(body['vector_field'])}")
                print(f"  vector_field维度: {len(body['vector_field']) if body['vector_field'] else 'None'}")
                print(f"  vector_field是否为None: {body['vector_field'] is None}")
                
                # 尝试插入
                print(f"  正在插入到OpenSearch...")
                try:
                    result = db.client.index(index=db.collection_name, body=body)
                    print(f"  ✅ 插入成功: {result}")
                except Exception as e:
                    print(f"  ❌ 插入失败: {e}")
                    print("  详细错误信息:")
                    import traceback
                    traceback.print_exc()
            else:
                print("  ❌ 向量为None，跳过插入")
    
    except Exception as e:
        print(f"❌ 模拟插入过程失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 步骤5: 使用原始insert方法
    print("\n=== 步骤5: 使用原始insert方法 ===")
    try:
        # 重新生成一个新的向量测试
        new_embedding = embedder.embed("另一个测试文本")
        new_vectors = [new_embedding]
        new_payloads = [{"text": "另一个测试文本", "category": "debug2"}]
        new_ids = ["debug_2"]
        
        print(f"新向量类型: {type(new_embedding)}")
        print(f"新向量维度: {len(new_embedding) if new_embedding else 'None'}")
        
        result = db.insert(vectors=new_vectors, payloads=new_payloads, ids=new_ids)
        print(f"✅ 原始insert方法成功")
    except Exception as e:
        print(f"❌ 原始insert方法失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 清理
    print("\n=== 清理测试索引 ===")
    try:
        db.delete_col()
        print("✅ 测试索引已删除")
    except Exception as e:
        print(f"⚠️ 清理失败: {e}")

if __name__ == "__main__":
    debug_vector_insert() 