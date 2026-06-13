from typing import List
import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from src.config import settings

class EmbeddingManager:

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"Error while loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, docs_to_embed: List[Document]) -> np.ndarray:
        if not self.model:
            raise ValueError("No model")

        texts = [doc.page_content for doc in docs_to_embed]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings

EmbeddingManager = EmbeddingManager()



