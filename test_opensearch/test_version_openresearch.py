import requests
import os
import json
from dotenv import load_dotenv


load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

host = os.getenv("HOST_OPENSEARCH")
port = os.getenv("PORT_OPENSEARCH")
user = os.getenv("USER_OPENSEARCH")
password = os.getenv("PASSWORD_OPENSEARCH")
index = "remote_sementic_dc"
search_pipeline = "search_pipeline_dc"

# 通用请求参数
headers = {"Content-Type": "application/json"}
auth = (user, password)

print("=== 检查集群健康状态 ===")
response = requests.get(f"https://{host}:{port}/_cluster/health", headers=headers, auth=auth, verify=False)
print(json.dumps(response.json(), indent=2, ensure_ascii=False))

print(f"\n=== 检查索引 '{index}' 的mapping ===")
try:
    response = requests.get(f"https://{host}:{port}/{index}/_mapping", headers=headers, auth=auth, verify=False)
    if response.status_code == 200:
        mapping = response.json()
        print(json.dumps(mapping, indent=2, ensure_ascii=False))
        
        # 检查是否有向量字段
        if index in mapping:
            properties = mapping[index]["mappings"].get("properties", {})
            vector_fields = []
            for field_name, field_config in properties.items():
                if field_config.get("type") == "knn_vector":
                    vector_fields.append({
                        "field": field_name,
                        "config": field_config
                    })
            
            if vector_fields:
                print(f"\n找到向量字段:")
                for vf in vector_fields:
                    print(f"  - 字段名: {vf['field']}")
                    print(f"    配置: {json.dumps(vf['config'], indent=4, ensure_ascii=False)}")
            else:
                print("\n未找到向量字段")
    else:
        print(f"获取mapping失败: {response.status_code} - {response.text}")
except Exception as e:
    print(f"获取mapping时出错: {e}")

print(f"\n=== 检查索引 '{index}' 的设置 ===")
try:
    response = requests.get(f"https://{host}:{port}/{index}/_settings", headers=headers, auth=auth, verify=False)
    if response.status_code == 200:
        settings = response.json()
        print(json.dumps(settings, indent=2, ensure_ascii=False))
    else:
        print(f"获取设置失败: {response.status_code} - {response.text}")
except Exception as e:
    print(f"获取设置时出错: {e}")

print(f"\n=== 检查搜索管道 '{search_pipeline}' ===")
try:
    response = requests.get(f"https://{host}:{port}/_search/pipeline/{search_pipeline}", headers=headers, auth=auth, verify=False)
    if response.status_code == 200:
        pipeline = response.json()
        print(json.dumps(pipeline, indent=2, ensure_ascii=False))
        
        # 分析管道中的处理器
        if search_pipeline in pipeline:
            processors = pipeline[search_pipeline].get("request_processors", [])
            response_processors = pipeline[search_pipeline].get("response_processors", [])
            
            print(f"\n管道分析:")
            print(f"  请求处理器数量: {len(processors)}")
            print(f"  响应处理器数量: {len(response_processors)}")
            
            for i, processor in enumerate(processors):
                processor_type = list(processor.keys())[0]
                print(f"  请求处理器 {i+1}: {processor_type}")
                if processor_type == "text_embedding":
                    print(f"    模型ID: {processor[processor_type].get('model_id', 'N/A')}")
                    print(f"    字段映射: {processor[processor_type].get('field_map', 'N/A')}")
    else:
        print(f"获取搜索管道失败: {response.status_code} - {response.text}")
except Exception as e:
    print(f"获取搜索管道时出错: {e}")

print(f"\n=== 检查所有ML模型 ===")
try:
    response = requests.get(f"https://{host}:{port}/_plugins/_ml/models", headers=headers, auth=auth, verify=False)
    if response.status_code == 200:
        models = response.json()
        print(json.dumps(models, indent=2, ensure_ascii=False))
    else:
        print(f"获取ML模型失败: {response.status_code} - {response.text}")
except Exception as e:
    print(f"获取ML模型时出错: {e}")

print(f"\n=== 测试搜索功能 ===")
try:
    # 测试基本搜索
    search_query = {
        "query": {
            "match_all": {}
        },
        "size": 1
    }
    
    response = requests.post(f"https://{host}:{port}/{index}/_search", 
                           headers=headers, auth=auth, verify=False, json=search_query)
    if response.status_code == 200:
        result = response.json()
        print(f"索引中共有 {result['hits']['total']['value']} 条文档")
        if result['hits']['hits']:
            print("示例文档结构:")
            print(json.dumps(result['hits']['hits'][0], indent=2, ensure_ascii=False))
    else:
        print(f"搜索失败: {response.status_code} - {response.text}")
except Exception as e:
    print(f"搜索时出错: {e}")