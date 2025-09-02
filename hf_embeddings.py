# hf_embeddings.py
import os
import requests
from typing import List

import os
import requests

class HuggingFaceAPIEmbeddings:
    def __init__(self):
        self.api_token = os.getenv("HF_API_TOKEN")
        self.model = os.getenv("HF_EMBEDDING_MODEL")
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model}"
        self.headers = {"Authorization": f"Bearer {self.api_token}"}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        resp = requests.post(self.api_url, headers=self.headers, json={"inputs": text})
        resp.raise_for_status()
        data = resp.json()
        # Some HF models return [ [vector] ], others [vector]
        return data[0] if isinstance(data[0][0], float) else data[0][0]