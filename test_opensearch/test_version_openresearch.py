import requests
import os
from dotenv import load_dotenv


load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

endpoint = os.getenv("ENDPOINT_OPENSEARCH")
user = os.getenv("USER_OPENSEARCH")
password = os.getenv("PASSWORD_OPENSEARCH")

response = requests.get(
    f"{endpoint}/_cluster/health", 
    headers={"Content-Type": "application/json"},
    auth=(user, password),
    verify=False
)

print(response.json())