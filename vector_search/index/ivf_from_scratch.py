import random
import math

def l2(a, b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

class IVFIndexFromScratch:

    def __init__(self, dimension, num_clusters=10):
        self.dimension = dimension
        self.num_clusters = num_clusters

        self.centroids = []
        self.inverted_lists = {}
        self.text_chunks = []
        self.vectors = []

    def _kmeans(self, vectors, iterations=10):

        # initialize random centroids
        self.centroids = random.sample(vectors, self.num_clusters)

        for _ in range(iterations):

            clusters = {i: [] for i in range(self.num_clusters)}

            for v in vectors:

                distances = [l2(v,c) for c in self.centroids]
                closest = distances.index(min(distances))

                clusters[closest].append(v)

            new_centroids = []

            for i in range(self.num_clusters):

                if clusters[i]:

                    mean = [
                        sum(v[d] for v in clusters[i]) / len(clusters[i])
                        for d in range(self.dimension)
                    ]

                    new_centroids.append(mean)

                else:
                    new_centroids.append(random.choice(vectors))

            self.centroids = new_centroids

    def add(self, embeddings, chunks):

        vectors = embeddings.tolist()

        self.vectors = vectors
        self.text_chunks = chunks

        print("Training IVF clusters...")
        self._kmeans(vectors)

        self.inverted_lists = {i: [] for i in range(self.num_clusters)}

        for idx, v in enumerate(vectors):

            distances = [l2(v,c) for c in self.centroids]
            cluster_id = distances.index(min(distances))

            self.inverted_lists[cluster_id].append(idx)

    def search(self, query_embedding, top_k=5, n_probe=1):

        centroid_distances = [l2(query_embedding,c) for c in self.centroids]

        closest_clusters = sorted(
            range(len(centroid_distances)),
            key=lambda i: centroid_distances[i]
        )[:n_probe]

        candidates = []

        for cluster_id in closest_clusters:
            candidates.extend(self.inverted_lists[cluster_id])

        scored = []

        for idx in candidates:

            dist = l2(query_embedding, self.vectors[idx])
            scored.append((dist, idx))

        scored.sort(key=lambda x: x[0])

        results = [
            self.text_chunks[idx]
            for _, idx in scored[:top_k]
        ]

        return results
