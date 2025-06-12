"""
OpenSearch适配器 - 在不修改核心库的情况下修复搜索和列表功能的问题
"""
import logging
from typing import List, Dict, Optional, Any
from mem0.vector_stores.opensearch import OpenSearchDB, OutputData

logger = logging.getLogger(__name__)


class OpenSearchAdapter:
    """OpenSearch适配器，修复搜索和列表功能的问题"""
    
    def __init__(self, **kwargs):
        """初始化适配器，包装原始的OpenSearchDB"""
        self.db = OpenSearchDB(**kwargs)
        self.client = self.db.client
        self.collection_name = self.db.collection_name
        self.embedding_model_dims = self.db.embedding_model_dims
    
    def create_index(self):
        """创建索引"""
        return self.db.create_index()
    
    def insert(self, vectors: List[List[float]], payloads: Optional[List[Dict]] = None, ids: Optional[List[str]] = None):
        """插入向量"""
        return self.db.insert(vectors, payloads, ids)
    
    def get(self, vector_id: str):
        """获取向量"""
        return self.db.get(vector_id)
    
    def update(self, vector_id: str, vector: Optional[List[float]] = None, payload: Optional[Dict] = None):
        """更新向量"""
        return self.db.update(vector_id, vector, payload)
    
    def delete(self, vector_id: str):
        """删除向量"""
        return self.db.delete(vector_id)
    
    def reset(self):
        """重置索引"""
        return self.db.reset()
    
    def list_cols(self):
        """列出所有集合"""
        return self.db.list_cols()
    
    def list(self, filters: Optional[Dict] = None, limit: Optional[int] = None) -> List[OutputData]:
        """修复的列表方法 - 处理嵌套列表问题"""
        try:
            result = self.db.list(filters, limit)
            
            # 处理嵌套列表结构
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    # 嵌套列表，展开第一层
                    return result[0]
                else:
                    # 正常列表
                    return result
            else:
                return []
        except Exception as e:
            logger.error(f"列表查询失败: {e}")
            return []
    
    def search(self, query: str, vectors: List[float], limit: int = 5, filters: Optional[Dict] = None) -> List[OutputData]:
        """修复的搜索方法 - 使用正确的KNN查询格式"""
        try:
            # 准备过滤条件
            filter_clauses = []
            if filters:
                for key in ["user_id", "run_id", "agent_id"]:
                    value = filters.get(key)
                    if value:
                        filter_clauses.append({"term": {f"payload.{key}.keyword": value}})
            
            # 构建正确的KNN查询
            query_body = {
                "size": limit,
                "knn": {
                    "vector_field": {
                        "vector": vectors,
                        "k": limit,
                    }
                }
            }
            
            # 添加过滤条件
            if filter_clauses:
                query_body["query"] = {"bool": {"filter": filter_clauses}}
            
            # 执行搜索
            response = self.client.search(index=self.collection_name, body=query_body)
            
            hits = response["hits"]["hits"]
            results = [
                OutputData(
                    id=hit["_source"].get("id"), 
                    score=hit["_score"], 
                    payload=hit["_source"].get("payload", {})
                )
                for hit in hits
            ]
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            # 如果KNN查询失败，尝试使用原始方法
            try:
                return self.db.search(query, vectors, limit, filters)
            except Exception as e2:
                logger.error(f"原始搜索方法也失败: {e2}")
                return []
    
    def search_alternative(self, query: str, vectors: List[float], limit: int = 5, filters: Optional[Dict] = None) -> List[OutputData]:
        """备用搜索方法 - 使用script_score查询"""
        try:
            # 准备过滤条件
            filter_clauses = []
            if filters:
                for key in ["user_id", "run_id", "agent_id"]:
                    value = filters.get(key)
                    if value:
                        filter_clauses.append({"term": {f"payload.{key}.keyword": value}})
            
            # 使用script_score查询作为备用方案
            query_body = {
                "size": limit,
                "query": {
                    "script_score": {
                        "query": {"bool": {"filter": filter_clauses}} if filter_clauses else {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'vector_field') + 1.0",
                            "params": {"query_vector": vectors}
                        }
                    }
                }
            }
            
            response = self.client.search(index=self.collection_name, body=query_body)
            
            hits = response["hits"]["hits"]
            results = [
                OutputData(
                    id=hit["_source"].get("id"), 
                    score=hit["_score"], 
                    payload=hit["_source"].get("payload", {})
                )
                for hit in hits
            ]
            return results
            
        except Exception as e:
            logger.error(f"备用搜索方法失败: {e}")
            return []
    
    def debug_index(self) -> Dict[str, Any]:
        """调试索引信息"""
        try:
            info = {}
            
            # 获取索引mapping
            mapping = self.client.indices.get_mapping(index=self.collection_name)
            info["mapping"] = mapping
            
            # 获取索引设置
            settings = self.client.indices.get_settings(index=self.collection_name)
            info["settings"] = settings
            
            # 获取文档数量
            count_response = self.client.count(index=self.collection_name)
            info["total_docs"] = count_response["count"]
            
            # 获取样本文档
            sample_response = self.client.search(
                index=self.collection_name, 
                body={"query": {"match_all": {}}, "size": 3}
            )
            info["sample_docs"] = sample_response["hits"]["hits"]
            
            return info
        except Exception as e:
            logger.error(f"调试索引信息失败: {e}")
            return {"error": str(e)} 