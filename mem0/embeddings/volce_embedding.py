import os
from typing import Literal, Optional
import requests
from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase


class VolceEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)
        self.api_key = config.volce_api_key or ""
        self.model = config.volce_model or "intfloat/multilingual-e5-small"
        self.endpoint = config.volce_endpoint or ""

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get the embedding for the given text using Volce.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        text = text.replace("\n", " ")
        request_body = {
            "model": self.model,
            "input": [text]
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        response = requests.post(self.endpoint, headers=headers, json=request_body)
        return response.json()["data"][0]["embedding"]

