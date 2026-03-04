from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return np.array(self.model.encode(texts, show_progress_bar=True))

    def embed_query(self, query):
        return np.array(self.model.encode([query]))[0]
