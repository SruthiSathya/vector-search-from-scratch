from vector_search.pdf_loader import load_pdfs_from_directory
from vector_search.chunking import chunk_text
from vector_search.embeddings import EmbeddingModel
from vector_search.index import VectorIndex
import numpy as np

DATA_PATH = "data/papers"

def build_index():
    print("Loading PDFs...")
    docs = load_pdfs_from_directory(DATA_PATH)

    all_chunks = []

    for name, text in docs:
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    model = EmbeddingModel()
    embeddings = model.embed_documents(all_chunks)

    dimension = embeddings.shape[1]
    index = VectorIndex(dimension)

    index.add(embeddings, all_chunks)

    return index, model


def query_loop(index, model):
    while True:
        query = input("\nEnter query (or 'exit'): ")
        if query == "exit":
            break

        query_embedding = model.embed_query(query)
        results = index.search(query_embedding, top_k=3)

        print("\nTop Results:\n")
        for i, r in enumerate(results):
            print(f"--- Result {i+1} ---")
            print(r[:500])
            print()


if __name__ == "__main__":
    index, model = build_index()
    query_loop(index, model)
