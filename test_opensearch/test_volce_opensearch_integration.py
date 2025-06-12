import os
import sys
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from mem0.vector_stores.opensearch import OpenSearchDB
from mem0.embeddings.volce_embedding import VolceEmbedding
from mem0.configs.embeddings.base import BaseEmbedderConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# 加载环境变量
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

class VolceOpenSearchIntegrationTest:
    def __init__(self):
        """初始化测试环境"""
        # OpenSearch 配置
        self.host = os.getenv("HOST_OPENSEARCH")
        self.port = os.getenv("PORT_OPENSEARCH")
        self.user = os.getenv("USER_OPENSEARCH")
        self.password = os.getenv("PASSWORD_OPENSEARCH")
        self.index = os.getenv("INDEX", "test_volce_integration")
        
        # Volce embedding 配置
        self.volce_api_key = os.getenv("VOLCE_EMBED_API_KEY", "")
        self.volce_endpoint = os.getenv("VOLCE_EMBEDDING_ENDPOINT", "")
        self.volce_model = os.getenv("VOLCE_MODEL", "intfloat/multilingual-e5-small")
        
        # 打印配置信息用于调试
        print(f"调试信息:")
        print(f"  Volce API Key: {'已设置' if self.volce_api_key else '未设置'}")
        print(f"  Volce Endpoint: {self.volce_endpoint}")
        print(f"  Volce Model: {self.volce_model}")
        
        # 初始化 embedding 服务
        self.embedding_config = BaseEmbedderConfig(
            volce_api_key=self.volce_api_key,
            volce_endpoint=self.volce_endpoint,
            volce_model=self.volce_model
        )
        self.embedder = VolceEmbedding(config=self.embedding_config)
        
        # 初始化 OpenSearch 数据库
        self.db = OpenSearchDB(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            verify_certs=False,
            use_ssl=True,
            collection_name=self.index,
            embedding_model_dims=384,  # multilingual-e5-small 是 384 维
        )
        
        # 测试数据
        self.test_documents = [
            {
                "text": "OpenAI发布了GPT-4，这是一个强大的大语言模型，具有出色的推理能力。",
                "category": "AI技术",
                "source": "OpenAI",
                "user_id": "user1"
            },
            {
                "text": "向量数据库是存储和检索高维向量数据的专门数据库，常用于相似性搜索。",
                "category": "数据库技术",
                "source": "技术文档",
                "user_id": "user2"
            },
            {
                "text": "机器学习模型的训练需要大量的计算资源和高质量的数据集。",
                "category": "机器学习",
                "source": "学术论文",
                "user_id": "user1"
            },
            {
                "text": "自然语言处理技术在搜索引擎、聊天机器人等应用中发挥重要作用。",
                "category": "NLP",
                "source": "技术博客",
                "user_id": "user3"
            },
            {
                "text": "深度学习神经网络可以自动学习数据中的复杂模式和特征。",
                "category": "深度学习",
                "source": "教程",
                "user_id": "user2"
            }
        ]

    def test_embedding_generation(self):
        """测试 Volce embedding 生成"""
        print("\n=== 测试 Volce Embedding 生成 ===")
        try:
            test_text = "这是一个测试文本，用于验证 embedding 生成功能。"
            print(f"测试文本: {test_text}")
            
            # 检查配置
            if not self.volce_endpoint:
                print("❌ Volce endpoint 未配置")
                return None
            if not self.volce_api_key:
                print("❌ Volce API key 未配置")
                return None
                
            print(f"正在调用 Volce API: {self.volce_endpoint}")
            embedding = self.embedder.embed(test_text)
            
            if embedding is None:
                print("❌ Embedding 生成返回 None")
                return None
            
            print(f"✅ Embedding 生成成功")
            print(f"   向量维度: {len(embedding)}")
            print(f"   向量类型: {type(embedding)}")
            print(f"   向量前5维: {embedding[:5]}")
            
            return embedding
        except Exception as e:
            print(f"❌ Embedding 生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def test_opensearch_setup(self):
        """测试 OpenSearch 设置"""
        print("\n=== 测试 OpenSearch 设置 ===")
        try:
            # 创建索引
            self.db.create_index()
            print("✅ OpenSearch 索引创建成功")
            
            # 列出集合
            collections = self.db.list_cols()
            print(f"✅ 当前集合: {collections}")
            
            return True
        except Exception as e:
            print(f"❌ OpenSearch 设置失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_document_indexing(self):
        """测试文档索引"""
        print("\n=== 测试文档索引 ===")
        try:
            vectors = []
            payloads = []
            ids = []
            
            print(f"当前使用的索引名称: {self.db.collection_name}")
            
            for i, doc in enumerate(self.test_documents):
                print(f"正在处理文档 {i+1}: {doc['text'][:50]}...")
                
                # 生成 embedding
                try:
                    embedding = self.embedder.embed(doc["text"], memory_action="add")
                    
                    # 严格验证向量
                    if embedding is None:
                        print(f"❌ 文档 {i+1} embedding 生成失败 - 返回None")
                        continue
                    
                    if not isinstance(embedding, list):
                        print(f"❌ 文档 {i+1} embedding 类型错误: {type(embedding)}")
                        continue
                    
                    if len(embedding) == 0:
                        print(f"❌ 文档 {i+1} embedding 为空列表")
                        continue
                    
                    if len(embedding) != 384:
                        print(f"❌ 文档 {i+1} embedding 维度错误: {len(embedding)}, 期望384")
                        continue
                    
                    if None in embedding:
                        print(f"❌ 文档 {i+1} embedding 包含None值")
                        continue
                    
                    if not all(isinstance(x, (int, float)) for x in embedding):
                        print(f"❌ 文档 {i+1} embedding 包含非数值")
                        continue
                    
                    print(f"   ✅ 文档 {i+1}: 向量验证通过 (维度: {len(embedding)})")
                    
                    vectors.append(embedding)
                    payloads.append({
                        "text": doc["text"],
                        "category": doc["category"],
                        "source": doc["source"],
                        "user_id": doc["user_id"]
                    })
                    ids.append(f"doc_{i+1}")
                    
                except Exception as e:
                    print(f"❌ 文档 {i+1} embedding 生成异常: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if not vectors:
                print("❌ 没有成功生成任何向量")
                return []
            
            print(f"\n准备插入数据:")
            print(f"  成功处理的文档数: {len(vectors)}")
            print(f"  目标索引: {self.db.collection_name}")
            print(f"  索引配置的向量维度: {self.db.embedding_model_dims}")
            
            # 最后一次验证所有向量
            for i, vec in enumerate(vectors):
                if vec is None or len(vec) != 384:
                    print(f"❌ 插入前发现问题向量 {i}: type={type(vec)}, len={len(vec) if vec else 'None'}")
                    return []
            
            # 批量插入
            print(f"正在插入 {len(vectors)} 个向量到 OpenSearch...")
            result = self.db.insert(vectors=vectors, payloads=payloads, ids=ids)
            print(f"✅ 成功索引 {len(vectors)} 个文档")
            
            return ids
        except Exception as e:
            print(f"❌ 文档索引失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def test_semantic_search(self):
        """测试语义搜索"""
        print("\n=== 测试语义搜索 ===")
        
        search_queries = [
            "大语言模型的能力",
            "向量搜索技术",
            "深度学习训练",
            "NLP应用场景"
        ]
        
        try:
            for query in search_queries:
                print(f"\n🔍 查询: {query}")
                
                # 生成查询向量
                query_embedding = self.embedder.embed(query, memory_action="search")
                
                if query_embedding is None:
                    print(f"   ❌ 查询向量生成失败")
                    continue
                
                # 执行搜索
                results = self.db.search(
                    query=query,
                    vectors=query_embedding,
                    limit=3
                )
                
                print(f"   找到 {len(results)} 个相关结果:")
                for i, result in enumerate(results):
                    print(f"   {i+1}. [Score: {result.score:.4f}] {result.payload['text'][:60]}...")
                    print(f"       类别: {result.payload['category']}, 来源: {result.payload['source']}")
                
        except Exception as e:
            print(f"❌ 语义搜索失败: {e}")
            import traceback
            traceback.print_exc()

    def test_filtered_search(self):
        """测试带过滤器的搜索"""
        print("\n=== 测试过滤搜索 ===")
        try:
            query = "人工智能技术"
            query_embedding = self.embedder.embed(query, memory_action="search")
            
            if query_embedding is None:
                print("❌ 查询向量生成失败")
                return
            
             # 按用户过滤
            user_filter = {"user_id": "user1"}
            results = self.db.search(
                query=query,
                vectors=query_embedding,
                limit=5,
                filters=user_filter
            )
            
            print(f"🔍 查询: {query} (用户: user1)")
            print(f"   找到 {len(results)} 个结果:")
            for i, result in enumerate(results):
                print(f"   {i+1}. [Score: {result.score:.4f}] {result.payload['text'][:60]}...")
                print(f"       用户: {result.payload['user_id']}, 类别: {result.payload['category']}")
            
            # 按类别过滤
            category_filter = {"category": "AI技术"}
            results = self.db.search(
                query=query,
                vectors=query_embedding,
                limit=5,
                filters=category_filter
            )
            
            print(f"\n🔍 查询: {query} (类别: AI技术)")
            print(f"   找到 {len(results)} 个结果:")
            for i, result in enumerate(results):
                print(f"   {i+1}. [Score: {result.score:.4f}] {result.payload['text'][:60]}...")
                print(f"       类别: {result.payload['category']}")
                
        except Exception as e:
            print(f"❌ 过滤搜索失败: {e}")
            import traceback
            traceback.print_exc()

    def test_document_operations(self, doc_ids: List[str]):
        """测试文档操作"""
        print("\n=== 测试文档操作 ===")
        try:
            if not doc_ids:
                print("⚠️ 没有可操作的文档ID")
                return
            
            # 获取单个文档
            doc_id = doc_ids[0]
            result = self.db.get(doc_id)
            if result:
                print(f"✅ 获取文档成功: {result.payload['text'][:50]}...")
            
            # 更新文档
            new_text = "这是一个更新后的文档内容，测试文档更新功能。"
            new_embedding = self.embedder.embed(new_text, memory_action="update")
            
            if new_embedding is None:
                print("❌ 更新文档的 embedding 生成失败")
                return
                
            new_payload = {
                "text": new_text,
                "category": "更新测试",
                "source": "测试更新",
                "user_id": "admin",
                "updated": True
            }
            
            self.db.update(vector_id=doc_id, vector=new_embedding, payload=new_payload)
            print(f"✅ 文档更新成功: {doc_id}")
            
            # 验证更新
            updated_result = self.db.get(doc_id)
            if updated_result:
                print(f"   更新后内容: {updated_result.payload['text'][:50]}...")
            
            # 列出所有文档
            all_docs = self.db.list(limit=10)
            print(f"✅ 共有 {len(all_docs)} 个文档")
            
        except Exception as e:
            print(f"❌ 文档操作失败: {e}")
            import traceback
            traceback.print_exc()

    def test_performance(self):
        """测试性能"""
        print("\n=== 测试性能 ===")
        import time
        
        try:
            # 测试 embedding 生成速度
            texts = ["测试文本" + str(i) for i in range(10)]
            
            start_time = time.time()
            embeddings = []
            for text in texts:
                embedding = self.embedder.embed(text)
                if embedding is not None:
                    embeddings.append(embedding)
            embedding_time = time.time() - start_time
            
            print(f"✅ Embedding 生成: {len(embeddings)}/{len(texts)} 个文本，耗时 {embedding_time:.2f}s")
            if embeddings:
                print(f"   平均每个: {embedding_time/len(embeddings):.3f}s")
            
            # 测试搜索速度
            if embeddings:
                query_embedding = embeddings[0]
                start_time = time.time()
                for _ in range(5):
                    results = self.db.search(query="性能测试", vectors=query_embedding, limit=3)
                search_time = time.time() - start_time
                
                print(f"✅ 搜索性能: 5次搜索，耗时 {search_time:.2f}s")
                print(f"   平均每次: {search_time/5:.3f}s")
            
        except Exception as e:
            print(f"❌ 性能测试失败: {e}")
            import traceback
            traceback.print_exc()

    def cleanup(self):
        """清理测试数据"""
        print("\n=== 清理测试数据 ===")
        try:
            # 注意：这会删除整个索引
            choice = input("是否要清理测试数据？(y/N): ").lower()
            if choice == 'y':
                self.db.reset()
                print("✅ 测试数据清理完成")
            else:
                print("⚠️ 跳过数据清理")
        except Exception as e:
            print(f"❌ 清理失败: {e}")

    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始 Volce + OpenSearch 综合测试...\n")
        print(f"配置信息:")
        print(f"  OpenSearch: {self.host}:{self.port}")
        print(f"  索引: {self.index}")
        print(f"  Volce模型: {self.volce_model}")
        print(f"  向量维度: 384")
        
        # 1. 测试 embedding 生成
        embedding = self.test_embedding_generation()
        if not embedding:
            print("❌ Embedding 测试失败，停止后续测试")
            return
        
        # 2. 测试 OpenSearch 设置
        if not self.test_opensearch_setup():
            print("❌ OpenSearch 设置失败，停止后续测试")
            return
        
        # 3. 测试文档索引
        doc_ids = self.test_document_indexing()
        if not doc_ids:
            print("❌ 文档索引失败，停止后续测试")
            return
        
        # 4. 测试语义搜索
        self.test_semantic_search()
        
        # 5. 测试过滤搜索
        self.test_filtered_search()
        
        # 6. 测试文档操作
        self.test_document_operations(doc_ids)
        
        # 7. 测试性能
        self.test_performance()
        
        # 8. 清理（可选）
        self.cleanup()
        
        print("\n🎉 所有测试完成！")

if __name__ == "__main__":
    tester = VolceOpenSearchIntegrationTest()
    tester.run_all_tests() 