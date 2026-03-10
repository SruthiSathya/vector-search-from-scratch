import os
os.environ["OMP_NUM_THREADS"] = "1"
import faiss
import numpy as np

class VectorIndexHNSW:
    def __init__(self, dimension, M=32, ef_construction=200, ef_search=50):
        """
        dimension: dimensionality of embeddings
        M: number of neighbors per node in HNSW graph (tradeoff between accuracy and speed)
        ef_construction: controls index construction time/accuracy
        ef_search: controls search accuracy/speed at query time
        """
        self.dimension = dimension
        self.text_chunks = []

        # Create HNSW index (L2 distance)
        self.index = faiss.IndexHNSWFlat(dimension, M)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search

    def add(self, embeddings, chunks):
        """
        Add embeddings and corresponding text chunks to the index.
        embeddings: numpy array of shape (num_vectors, dimension)
        chunks: list of text corresponding to each vector
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D")

        if embeddings.shape[1] != self.dimension:
            raise ValueError("Embedding dimension mismatch")

        self.index.add(embeddings)
        self.text_chunks.extend(chunks)

    def search(self, query_embedding, top_k=5):
        """
        Search the index using HNSW for top_k nearest neighbors (L2 distance)
        query_embedding: list or numpy array of length = dimension
        """
        query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            if idx == -1:
                continue
            results.append(self.text_chunks[idx])

        return results
