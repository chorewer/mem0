import os
import sys
import requests
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from mem0.embeddings.volce_embedding import VolceEmbedding
from mem0.configs.embeddings.base import BaseEmbedderConfig

# 加载环境变量
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

def test_volce_embedding():
    print("🧪 测试 Volce Embedding 功能")
    
    # 获取配置
    volce_api_key = os.getenv("VOLCE_EMBED_API_KEY", "")
    volce_endpoint = os.getenv("VOLCE_EMBEDDING_ENDPOINT", "")
    volce_model = os.getenv("VOLCE_MODEL", "intfloat/multilingual-e5-small")
    
    print(f"\n配置信息:")
    print(f"  API Key: {'已设置' if volce_api_key else '❌ 未设置'}")
    print(f"  Endpoint: {volce_endpoint if volce_endpoint else '❌ 未设置'}")
    print(f"  Model: {volce_model}")
    
    if not volce_api_key:
        print("\n❌ 请设置 VOLCE_EMBED_API_KEY 环境变量")
        return
    
    if not volce_endpoint:
        print("\n❌ 请设置 VOLCE_EMBEDDING_ENDPOINT 环境变量")
        return
    
    # 测试直接 API 调用
    print("\n=== 直接测试 API 调用 ===")
    try:
        test_text = "这是一个测试文本"
        request_body = {
            "model": volce_model,
            "input": [test_text]
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {volce_api_key}"
        }
        
        print(f"请求 URL: {volce_endpoint}")
        print(f"请求体: {request_body}")
        print(f"请求头: {headers}")
        
        response = requests.post(volce_endpoint, headers=headers, json=request_body)
        
        print(f"响应状态码: {response.status_code}")
        print(f"响应头: {response.headers}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API 调用成功")
            print(f"响应数据结构: {list(result.keys())}")
            
            if "data" in result and len(result["data"]) > 0:
                embedding = result["data"][0]["embedding"]
                print(f"向量维度: {len(embedding)}")
                print(f"向量前5维: {embedding[:5]}")
            else:
                print("❌ 响应数据格式不正确")
                print(f"完整响应: {result}")
        else:
            print(f"❌ API 调用失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"❌ 直接 API 调用失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试 VolceEmbedding 类
    print("\n=== 测试 VolceEmbedding 类 ===")
    try:
        # 创建配置
        config = BaseEmbedderConfig(
            volce_api_key=volce_api_key,
            volce_endpoint=volce_endpoint,
            volce_model=volce_model
        )
        
        # 创建 embedder
        embedder = VolceEmbedding(config=config)
        
        # 检查初始化是否正确
        print(f"Embedder API Key: {'已设置' if embedder.api_key else '未设置'}")
        print(f"Embedder Endpoint: {embedder.endpoint}")
        print(f"Embedder Model: {embedder.model}")
        
        # 测试 embedding
        test_texts = [
            "OpenAI 发布了 GPT-4",
            "向量数据库用于相似性搜索",
            "深度学习神经网络"
        ]
        
        for i, text in enumerate(test_texts):
            print(f"\n测试文本 {i+1}: {text}")
            try:
                embedding = embedder.embed(text)
                
                if embedding is None:
                    print("❌ 返回 None")
                elif isinstance(embedding, list) and len(embedding) > 0:
                    print(f"✅ 成功生成向量，维度: {len(embedding)}")
                    print(f"   向量类型: {type(embedding)}")
                    print(f"   前3维: {embedding[:3]}")
                else:
                    print(f"❌ 返回数据格式不正确: {type(embedding)}, {embedding}")
                    
            except Exception as e:
                print(f"❌ 生成失败: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"❌ VolceEmbedding 类测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_volce_embedding() 