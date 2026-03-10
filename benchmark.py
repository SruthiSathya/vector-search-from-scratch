import time

import numpy as np

def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    hits = sum(
        1
        for r in retrieved_k
        if any(str(rel) in str(r) for rel in relevant)
    )
    return hits / k


def recall_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    hits = sum(
        1
        for r in retrieved_k
        if any(str(rel) in str(r) for rel in relevant)
    )
    return hits / len(relevant)


def reciprocal_rank(retrieved, relevant):
    for i, r in enumerate(retrieved):
        if any(str(rel) in str(r) for rel in relevant):
            return 1 / (i + 1)
    return 0

def run_benchmark(index, model, queries, k=3):
    p_scores = []
    r_scores = []
    rr_scores = []
    latencies = []

    for q in queries:
        query = q["query"]
        relevant = q["relevant_chunks"]

        q_embedding = model.embed_query(query)
        start = time.time()
        retrieved = index.search(q_embedding, top_k=k)
        end = time.time()
        latency = end - start

        p = precision_at_k(retrieved, relevant, k)
        r = recall_at_k(retrieved, relevant, k)
        rr = reciprocal_rank(retrieved, relevant)

        p_scores.append(p)
        r_scores.append(r)
        rr_scores.append(rr)
        latencies.append(latency)

    print("\n===== BENCHMARK RESULTS =====")
    print(f"Precision@{k}: {np.mean(p_scores):.3f}")
    print(f"Recall@{k}: {np.mean(r_scores):.3f}")
    print(f"MRR: {np.mean(rr_scores):.3f}")
    print(f"Avg Query Latency: {np.mean(latencies):.4f} sec")
