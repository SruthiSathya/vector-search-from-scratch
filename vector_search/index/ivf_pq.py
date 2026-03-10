from vector_search.index.ivf_from_scratch import IVFIndexFromScratch
from vector_search.index.pq import ProductQuantizer

class IVFPQIndex:

    def __init__(self, dimension, num_clusters=20):

        self.dimension = dimension
        self.ivf = IVFIndexFromScratch(dimension, num_clusters)

        self.pq = ProductQuantizer(dimension)

        self.text_chunks = []
        self.vectors = []

    def add(self, embeddings, chunks):

        vectors = embeddings.tolist()

        self.text_chunks = chunks
        self.vectors = vectors

        self.ivf.add(embeddings, chunks)

        self.pq.train(vectors)

        self.pq.encode(vectors)

    def search(self, query_embedding, top_k=5, n_probe=1):

        candidates = []

        centroid_distances = [
            sum((a-b)**2 for a,b in zip(query_embedding,c))
            for c in self.ivf.centroids
        ]

        closest_clusters = sorted(
            range(len(centroid_distances)),
            key=lambda i: centroid_distances[i]
        )[:n_probe]

        for cid in closest_clusters:
            candidates.extend(self.ivf.inverted_lists[cid])

        scored = []

        for idx in candidates:

            code = self.pq.codes[idx]

            dist = self.pq.distance(query_embedding, code)

            scored.append((dist, idx))

        scored.sort(key=lambda x: x[0])

        return [
            self.text_chunks[idx]
            for _, idx in scored[:top_k]
        ]
