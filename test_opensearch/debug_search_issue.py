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

def debug_search_issue():
    print("🔍 调试搜索返回0结果的问题")
    
    # 配置
    host = os.getenv("HOST_OPENSEARCH")
    port = os.getenv("PORT_OPENSEARCH")
    user = os.getenv("USER_OPENSEARCH")
    password = os.getenv("PASSWORD_OPENSEARCH")
    index = os.getenv("INDEX", "mem0serve")
    
    volce_api_key = os.getenv("VOLCE_EMBED_API_KEY", "")
    volce_endpoint = os.getenv("VOLCE_EMBEDDING_ENDPOINT", "")
    volce_model = os.getenv("VOLCE_MODEL", "intfloat/multilingual-e5-small")
    
    print(f"目标索引: {index}")
    
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
    
    # 步骤1: 检查索引中的数据
    print("\n=== 步骤1: 检查索引中的数据 ===")
    try:
        # 获取索引中的所有文档
        all_docs_result = db.list(limit=100)
        print(f"db.list()返回类型: {type(all_docs_result)}")
        
        # 处理嵌套列表结构
        if isinstance(all_docs_result, list) and len(all_docs_result) > 0:
            if isinstance(all_docs_result[0], list):
                # 嵌套列表，取第一个列表
                all_docs = all_docs_result[0]
                print(f"展开嵌套列表，文档数量: {len(all_docs)}")
            else:
                all_docs = all_docs_result
                print(f"直接列表，文档数量: {len(all_docs)}")
        else:
            all_docs = []
            print(f"空结果或其他类型")
        
        print(f"索引中共有 {len(all_docs)} 个文档")
        
        if len(all_docs) > 0:
            print("前3个文档:")
            for i, doc in enumerate(all_docs[:3]):
                print(f"  {i+1}. 文档类型: {type(doc)}")
                if hasattr(doc, 'id'):
                    print(f"     ID: {doc.id}")
                    print(f"     Payload: {doc.payload}")
                else:
                    print(f"     数据: {doc}")
        else:
            print("❌ 索引为空，需要先插入数据")
            return
    except Exception as e:
        print(f"❌ 获取文档列表失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤2: 检查索引mapping
    print("\n=== 步骤2: 检查索引mapping ===")
    try:
        mapping_response = db.client.indices.get_mapping(index=index)
        print(f"索引mapping:")
        import json
        print(json.dumps(mapping_response, indent=2))
    except Exception as e:
        print(f"❌ 获取mapping失败: {e}")
    
    # 步骤3: 直接查询检查
    print("\n=== 步骤3: 直接查询检查 ===")
    try:
        # 使用match_all查询
        query = {"query": {"match_all": {}}, "size": 5}
        response = db.client.search(index=index, body=query)
        
        print(f"match_all查询结果:")
        print(f"  总数: {response['hits']['total']['value']}")
        print(f"  返回: {len(response['hits']['hits'])}")
        
        if response['hits']['hits']:
            print("示例文档结构:")
            for i, hit in enumerate(response['hits']['hits'][:2]):
                print(f"  文档 {i+1}:")
                print(f"    _source keys: {list(hit['_source'].keys())}")
                if 'vector_field' in hit['_source']:
                    vec_field = hit['_source']['vector_field']
                    print(f"    vector_field类型: {type(vec_field)}")
                    print(f"    vector_field维度: {len(vec_field) if isinstance(vec_field, list) else 'Not a list'}")
                else:
                    print("    ❌ 没有找到vector_field字段")
    except Exception as e:
        print(f"❌ 直接查询失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 步骤4: 测试向量搜索
    print("\n=== 步骤4: 测试向量搜索 ===")
    try:
        # 生成查询向量
        query_text = "测试查询"
        query_vector = embedder.embed(query_text)
        
        print(f"查询文本: {query_text}")
        print(f"查询向量维度: {len(query_vector)}")
        
        # 手动构造KNN查询
        knn_query = {
            "size": 5,
            "query": {
                "knn": {
                    "vector_field": {
                        "vector": query_vector,
                        "k": 5,
                    }
                }
            }
        }
        
        print(f"KNN查询结构: {list(knn_query.keys())}")
        
        # 执行搜索
        response = db.client.search(index=index, body=knn_query)
        
        print(f"KNN搜索结果:")
        print(f"  总数: {response['hits']['total']['value']}")
        print(f"  返回: {len(response['hits']['hits'])}")
        
        if response['hits']['hits']:
            for i, hit in enumerate(response['hits']['hits']):
                print(f"  {i+1}. Score: {hit['_score']}, ID: {hit['_source'].get('id', 'N/A')}")
        
    except Exception as e:
        print(f"❌ KNN搜索失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 步骤5: 测试OpenSearchDB的search方法
    print("\n=== 步骤5: 测试OpenSearchDB的search方法 ===")
    try:
        query_text = "OpenAI GPT-4"
        query_vector = embedder.embed(query_text)
        
        print(f"使用OpenSearchDB.search方法")
        print(f"查询文本: {query_text}")
        
        results = db.search(query=query_text, vectors=query_vector, limit=5)
        
        print(f"搜索结果:")
        print(f"  返回数量: {len(results)}")
        
        for i, result in enumerate(results):
            print(f"  {i+1}. ID: {result.id}, Score: {result.score}")
            print(f"      Payload keys: {list(result.payload.keys())}")
        
    except Exception as e:
        print(f"❌ OpenSearchDB搜索失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_search_issue() 