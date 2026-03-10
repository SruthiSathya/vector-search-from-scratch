from vector_search.chunking.chunking import chunk_text
from vector_search.embeddings.embeddings import EmbeddingModel
from vector_search.index.index import VectorIndex
from vector_search.index.index_from_scratch import VectorIndexFromScratchImplementation
from vector_search.index.index_hnsw import VectorIndexHNSW
from vector_search.index.index_hnsw_from_scratch import HNSWIndexFromScratch, visualize_hnsw
from vector_search.index.ivf_from_scratch import IVFIndexFromScratch
from vector_search.index.ivf_pq import IVFPQIndex
from vector_search.pdf_loader import load_pdfs_from_directory

from benchmark import run_benchmark
from benchmark_queries import benchmark_queries

DATA_PATH = "data/papers"


def choose_index():
    print("\nSelect Vector Index:")
    print("1. Brute Force (From Scratch)")
    print("2. FAISS Flat (Exact)")
    print("3. FAISS HNSW (Approximate)")
    print("4. Benchmark ALL Indexes")
    print("5. IVF from scratch")
    print("6. IVF + Product Quantizer")
    print("7. HNSW from scratch")

    choice = input("Enter choice: ")

    mapping = {
        "1": "bruteforce",
        "2": "faiss_flat",
        "3": "faiss_hnsw",
        "4": "benchmark_all",
        "5": "ivf_scratch",
        "6": "ivf_pq",
        "7": "hnsw_scratch",
    }

    return mapping.get(choice, "faiss_flat")


def load_and_embed():
    print("Loading PDFs...")

    docs = load_pdfs_from_directory(DATA_PATH)

    all_chunks = []

    for name, text in docs:
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    model = EmbeddingModel()
    embeddings = model.embed_documents(all_chunks)

    return all_chunks, embeddings, model


def build_index(index_type, dimension):

    if index_type == "bruteforce":
        print("Using Brute Force Index (From Scratch)")
        return VectorIndexFromScratchImplementation(dimension)

    elif index_type == "faiss_flat":
        print("Using FAISS Flat Index")
        return VectorIndex(dimension)

    elif index_type == "faiss_hnsw":
        print("Using FAISS HNSW Index")
        return VectorIndexHNSW(dimension)

    elif index_type == "ivf_scratch":
        print("Using IVF Index From Scratch")
        return IVFIndexFromScratch(dimension, num_clusters=20)

    elif index_type == "ivf_pq":
        print("Using IVF + Product Quantization")
        return IVFPQIndex(dimension, num_clusters=20)

    elif index_type == "hnsw_scratch":
        print("Using HNSW From Scratch")
        return HNSWIndexFromScratch()

    else:
        raise ValueError("Unknown index type")


def benchmark_all_indexes(chunks, embeddings, model):

    dimension = embeddings.shape[1]

    indexes = {
        "Brute Force": VectorIndexFromScratchImplementation(dimension),
        "FAISS Flat": VectorIndex(dimension),
        "FAISS HNSW": VectorIndexHNSW(dimension),
        "HNSW From Scratch": HNSWIndexFromScratch(),
        "IVF Scratch": IVFIndexFromScratch(dimension, num_clusters=20),
        "IVF + PQ": IVFPQIndex(dimension, num_clusters=20),
    }

    for name, index in indexes.items():

        print("\n==============================")
        print(f"Benchmarking: {name}")
        print("==============================")

        if name == "HNSW From Scratch":
            for vec in embeddings:
                index.add(vec)
        else:
            index.add(embeddings, chunks)

        run_benchmark(index, model, benchmark_queries, k=3)


def query_loop(index, model):

    while True:

        query = input("\nEnter query (or 'exit'): ")

        if query == "exit":
            break

        query_embedding = model.embed_query(query)

        if hasattr(index, "nodes"):  # HNSW from scratch visualization

            results, visited, edges = index.search(
                query_embedding,
                top_k=3,
                return_path=True
            )

            visualize_hnsw(index, visited, edges)

        else:

            results = index.search(query_embedding, top_k=3)

        print("\nTop Results:\n")

        for i, r in enumerate(results):
            print(f"--- Result {i+1} ---")
            print(str(r)[:500])
            print()


if __name__ == "__main__":

    choice = choose_index()

    chunks, embeddings, model = load_and_embed()

    dimension = embeddings.shape[1]

    if choice == "benchmark_all":

        benchmark_all_indexes(chunks, embeddings, model)

    else:

        index = build_index(choice, dimension)

        if choice == "hnsw_scratch":
            for vec in embeddings:
                index.add(vec)
        else:
            index.add(embeddings, chunks)

        print("\nRunning Benchmark...\n")

        run_benchmark(index, model, benchmark_queries, k=3)

        query_loop(index, model)
