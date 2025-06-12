import requests
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

endpoint = os.getenv("ENDPOINT_OPENSEARCH")
index = os.getenv("INDEX")
search_pipeline = os.getenv("SEARCH_PIPELINE")
user = os.getenv("USER_OPENSEARCH")
password = os.getenv("PASSWORD_OPENSEARCH")

request_body = {
    "size": 10,
    "query": {
        "hybrid": {
            "queries": [
                {
                    "match": {
                        "qa": "测试查询"
                    }
                },
                {
                    "remote_neural": {
                        "qa_knn": {
                            "query_text": "测试查询",
                            "k": 10
                        }
                    }
                }
            ]
        }
    }
}

headers = {"Content-Type": "application/json"}

response = requests.post(
    f"{endpoint}/{index}/_search?search_pipeline={search_pipeline}", 
    json=request_body, 
    headers=headers,
    auth=(user, password),
    verify=False
)

print(response.json())