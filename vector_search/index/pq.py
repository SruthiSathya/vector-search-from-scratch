import random
import math

def l2(a, b):
    return sum((x-y)**2 for x,y in zip(a,b))

class ProductQuantizer:

    def __init__(self, dimension, m=8, k=256):
        """
        dimension: vector dimension
        m: number of subspaces
        k: centroids per subspace
        """
        self.dimension = dimension
        self.m = m
        self.k = k

        self.sub_dim = dimension // m
        self.codebooks = []
        self.codes = []

    def _kmeans(self, vectors, k, iterations=8):

        centroids = random.sample(vectors, k)

        for _ in range(iterations):

            clusters = [[] for _ in range(k)]

            for v in vectors:
                distances = [l2(v,c) for c in centroids]
                idx = distances.index(min(distances))
                clusters[idx].append(v)

            new_centroids = []

            for i in range(k):

                if clusters[i]:
                    mean = [
                        sum(v[d] for v in clusters[i]) / len(clusters[i])
                        for d in range(len(vectors[0]))
                    ]
                else:
                    mean = random.choice(vectors)

                new_centroids.append(mean)

            centroids = new_centroids

        return centroids

    def train(self, vectors):

        print("Training Product Quantizer...")

        for i in range(self.m):

            sub_vectors = [
                v[i*self.sub_dim:(i+1)*self.sub_dim]
                for v in vectors
            ]

            centroids = self._kmeans(sub_vectors, self.k)

            self.codebooks.append(centroids)

    def encode(self, vectors):

        self.codes = []

        for v in vectors:

            code = []

            for i in range(self.m):

                subvector = v[i*self.sub_dim:(i+1)*self.sub_dim]

                centroids = self.codebooks[i]

                distances = [l2(subvector,c) for c in centroids]

                code.append(distances.index(min(distances)))

            self.codes.append(code)

    def distance(self, query, code):

        total = 0

        for i in range(self.m):

            subvector = query[i*self.sub_dim:(i+1)*self.sub_dim]

            centroid = self.codebooks[i][code[i]]

            total += l2(subvector, centroid)

        return total
