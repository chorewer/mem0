from mem0.vector_stores.opensearch import OpenSearchDB
import os
import logging

# 配置logging以显示内层的logger信息
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # 输出到控制台
    ]
)

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

host = os.getenv("HOST_OPENSEARCH")
port = os.getenv("PORT_OPENSEARCH")
index = os.getenv("INDEX")
search_pipeline = os.getenv("SEARCH_PIPELINE")
user = os.getenv("USER_OPENSEARCH")
password = os.getenv("PASSWORD_OPENSEARCH")

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

def test_create_index():
    """测试创建索引"""
    print("\n=== 测试创建索引 ===")
    try:
        db.create_index()
        print("✅ 创建索引成功")
    except Exception as e:
        print(f"❌ 创建索引失败: {e}")

def test_list_collections():
    """测试列出所有集合"""
    print("\n=== 测试列出所有集合 ===")
    try:
        collections = db.list_cols()
        print(f"✅ 获取集合列表成功: {collections}")
        return collections
    except Exception as e:
        print(f"❌ 获取集合列表失败: {e}")
        return []

def test_insert_vectors():
    """测试插入向量"""
    print("\n=== 测试插入向量 ===")
    try:
        # 生成测试向量数据 - 修改为384维
        vectors = [
            [0.1 * i for i in range(384)],  # 向量1
            [0.2 * i for i in range(384)],  # 向量2
            [0.3 * i for i in range(384)]   # 向量3
        ]
        
        payloads = [
            {"user_id": "user1", "text": "这是第一个测试文档", "category": "test"},
            {"user_id": "user2", "text": "这是第二个测试文档", "category": "demo"},
            {"user_id": "user1", "text": "这是第三个测试文档", "category": "test"}
        ]
        
        ids = ["test_vec_1", "test_vec_2", "test_vec_3"]
        
        result = db.insert(vectors=vectors, payloads=payloads, ids=ids)
        print("✅ 插入向量成功")
        return vectors, payloads, ids
    except Exception as e:
        print(f"❌ 插入向量失败: {e}")
        return [], [], []

def test_get_vector(vector_id):
    """测试根据ID获取向量"""
    print(f"\n=== 测试获取向量 ID: {vector_id} ===")
    try:
        result = db.get(vector_id)
        if result:
            print(f"✅ 获取向量成功: ID={result.id}, Score={result.score}")
            print(f"   Payload: {result.payload}")
        else:
            print("⚠️ 向量不存在")
        return result
    except Exception as e:
        print(f"❌ 获取向量失败: {e}")
        return None

def test_search_vectors(query_vector):
    """测试搜索相似向量"""
    print("\n=== 测试搜索相似向量 ===")
    try:
        # 不带过滤器的搜索
        results = db.search(query="测试查询", vectors=query_vector, limit=5)
        print(f"✅ 无过滤器搜索成功，找到 {len(results)} 个结果:")
        for i, result in enumerate(results):
            print(f"   {i+1}. ID: {result.id}, Score: {result.score}")
        
        # 带过滤器的搜索
        filters = {"user_id": "user1"}
        filtered_results = db.search(query="测试查询", vectors=query_vector, limit=5, filters=filters)
        print(f"✅ 带过滤器搜索成功，找到 {len(filtered_results)} 个结果:")
        for i, result in enumerate(filtered_results):
            print(f"   {i+1}. ID: {result.id}, Score: {result.score}")
        
        return results
    except Exception as e:
        print(f"❌ 搜索向量失败: {e}")
        return []

def test_update_vector(vector_id):
    """测试更新向量"""
    print(f"\n=== 测试更新向量 ID: {vector_id} ===")
    try:
        new_vector = [0.5 * i for i in range(384)]  # 修改为384维
        new_payload = {"user_id": "user1", "text": "这是更新后的文档", "category": "updated", "timestamp": "2024-01-01"}
        
        db.update(vector_id=vector_id, vector=new_vector, payload=new_payload)
        print("✅ 更新向量成功")
        
        # 验证更新结果
        updated_result = db.get(vector_id)
        if updated_result:
            print(f"   更新后的Payload: {updated_result.payload}")
    except Exception as e:
        print(f"❌ 更新向量失败: {e}")

def test_list_all_vectors():
    """测试列出所有向量"""
    print("\n=== 测试列出所有向量 ===")
    try:
        # 列出所有向量
        all_vectors = db.list()
        print(f"✅ 列出所有向量成功，共 {len(all_vectors)} 个")
        
        # 带过滤器列出向量
        filtered_vectors = db.list(filters={"user_id": "user1"}, limit=10)
        print(f"✅ 带过滤器列出向量成功，共 {len(filtered_vectors)} 个")
        
        return all_vectors
    except Exception as e:
        print(f"❌ 列出向量失败: {e}")
        return []

def test_delete_vector(vector_id):
    """测试删除向量"""
    print(f"\n=== 测试删除向量 ID: {vector_id} ===")
    try:
        db.delete(vector_id)
        print("✅ 删除向量成功")
        
        # 验证删除结果
        deleted_result = db.get(vector_id)
        if deleted_result is None:
            print("✅ 确认向量已被删除")
        else:
            print("⚠️ 向量似乎还存在")
    except Exception as e:
        print(f"❌ 删除向量失败: {e}")

def test_reset_index():
    """测试重置索引"""
    print("\n=== 测试重置索引 ===")
    try:
        db.reset()
        print("✅ 重置索引成功")
    except Exception as e:
        print(f"❌ 重置索引失败: {e}")

def run_all_tests():
    """运行所有测试"""
    print("🚀 开始运行OpenSearchDB接口测试...\n")
    
    # 1. 测试创建索引
    test_create_index()
    
    # 2. 测试列出集合
    collections = test_list_collections()
    
    # 3. 测试插入向量
    vectors, payloads, ids = test_insert_vectors()
    
    if vectors:  # 如果插入成功，继续其他测试
        # 4. 测试获取向量
        test_get_vector(ids[0])
        
        # 5. 测试搜索向量
        test_search_vectors(vectors[0])
        
        # 6. 测试列出所有向量
        test_list_all_vectors()
        
        # 7. 测试更新向量
        test_update_vector(ids[0])
        
        # 8. 测试删除向量
        test_delete_vector(ids[2])  # 删除第三个向量
        
        # 9. 最后测试重置索引（可选，会清空所有数据）
        # test_reset_index()
    
    print("\n🎉 所有测试完成！")

if __name__ == "__main__":
    run_all_tests()