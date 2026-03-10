import math

class VectorIndexFromScratchImplementation:
    '''
    This class contains implementation for vector indexes with python and math library
    '''
    def __init__(self, dimension):
        # Store expected vector dimension
        self.dimension = dimension
        
        # Store vectors like:
        # [
        #   [2, 3, 4],
        #   [1, 1, 1],
        #   [7, 8, 9]
        # ]
        self.vectors = []
        
        # Store matching text chunks
        self.text_chunks = []

    def add(self, embeddings, chunks):
        """
        Add vectors and corresponding text chunks to the index.

        Args:
            embeddings (list of list of floats): Each chunk of text is represented as a vector. 
            chunks (list of str): The text chunks corresponding to each vector.

        Goal:
            - Store the vectors in `self.vectors`.
            - Store the corresponding text in `self.text_chunks`.
            - Keep index alignment: vector[i] corresponds to text_chunks[i].

        Notes:
            - Validates that each vector has the expected dimension.
            - Copies vectors to avoid shared mutable state.
            - Supports multiple add() calls while maintaining correct alignment.
            If we overwrote instead of appending the vector list will be replaced instead of appeneded.
        """

        for vector in embeddings:
            if len(vector) != self.dimension:
                raise ValueError("Embedding dimension mismatch")
            self.vectors.append(vector.copy()) #  avoiding shared mutable state
        self.text_chunks.extend(chunks)

    def search_l2(self, query_embedding, top_k=5):
        """
        Basic Idea
        - Compute L2 distance for a given query vector and all vectors that we have, 
        - Store the distances as an array of tuple (distance, idx)
        - Then find the top K nearest neighbors
        """
        if len(query_embedding) != self.dimension:
            raise ValueError("Query dimension mismatch")
        distances = []
        for i, vector in enumerate(self.vectors):
            # Euclidean (L2) distance = (a1 - b1)*(a1 - b1) + (a2 - b2)*(a2 - b2)...
            total = 0
            for j in range(self.dimension):
                difference = query_embedding[j] - vector[j]
                total += difference * difference
            total = math.sqrt(total)
            distances.append((total, i)) # store as tuple (distance, index)
        distances.sort(key=lambda x: x[0]) # sort by shortest distance
        results = []
        for k in range(min(top_k, len(distances))):
            _, index = distances[k]
            results.append(self.text_chunks[index])
        return results

    @staticmethod
    def _normalize(vector):
        norm = math.sqrt(sum(x*x for x in vector))
        if norm == 0:
            return vector  # or raise error
        return [x / norm for x in vector]

    def search_cosine(self, query_embedding, top_k=5):
        """
        Search using cosine similarity.
        Returns top_k most similar text chunks.
        """
        # Normalize the query
        query_norm = self._normalize(query_embedding)

        scores = []
        for i, vector in enumerate(self.vectors):
            # Normalize stored vector (or pre-normalize on add for speed)
            stored_norm = self._normalize(vector)
            # Cosine similarity = dot product of normalized vectors
            dot = sum(q*v for q, v in zip(query_norm, stored_norm))
            scores.append((dot, i))

        # Higher dot = more similar
        scores.sort(key=lambda x: -x[0])
        return [self.text_chunks[i] for _, i in scores[:top_k]]

    def search_l1(self, query_embedding, top_k=5):
        """
        Search using L1 (Manhattan) distance.
        Returns top_k most similar text chunks.
        """
        if len(query_embedding) != self.dimension:
            raise ValueError("Query dimension mismatch")

        distances = []
        for i, vector in enumerate(self.vectors):
            # L1 distance = sum of absolute differences
            dist = sum(abs(q - v) for q, v in zip(query_embedding, vector))
            distances.append((dist, i))

        # Smaller distance = more similar
        distances.sort(key=lambda x: x[0])
        return [self.text_chunks[i] for _, i in distances[:top_k]]

    def search(self, query_embedding, top_k=5, metric="cosine"):
        """
        Unified search interface used by benchmarks and query loops.
        """

        if metric == "cosine":
            return self.search_cosine(query_embedding, top_k)

        elif metric == "l2":
            return self.search_l2(query_embedding, top_k)

        elif metric == "l1":
            return self.search_l1(query_embedding, top_k)

        else:
            raise ValueError(f"Unknown metric: {metric}")