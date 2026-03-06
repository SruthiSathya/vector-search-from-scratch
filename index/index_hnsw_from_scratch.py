import math
import random

def l2_distance(a, b):
    return math.sqrt(sum((x - y)**2 for x, y in zip(a, b)))

class Node:
    def __init__(self, vector, idx):
        self.vector = vector
        self.idx = idx
        self.neighbors = []  # list of neighbor Node references

class HNSWIndexFromScratch:
    def __init__(self, M=5):
        """
        M: maximum number of neighbors per node
        """
        self.nodes = []
        self.M = M

    def add(self, vector):
        new_node = Node(vector, len(self.nodes))
        # If graph not empty connect to nearest M nodes
        if self.nodes:
            distances = [(l2_distance(vector, n.vector), n) for n in self.nodes]
            distances.sort(key=lambda x: x[0])
            for _, neighbor in distances[:self.M]:
                new_node.neighbors.append(neighbor)
                neighbor.neighbors.append(new_node)  # bidirectional
        self.nodes.append(new_node)

    def search(self, query_vector, top_k=3):
        """
        Greedy search starting from a random node
        """
        if not self.nodes:
            return []

        entry = random.choice(self.nodes)
        visited = set()
        candidate = entry
        improved = True

        while improved:
            improved = False
            visited.add(candidate.idx)
            for neighbor in candidate.neighbors:
                if neighbor.idx in visited:
                    continue
                if l2_distance(query_vector, neighbor.vector) < l2_distance(query_vector, candidate.vector):
                    candidate = neighbor
                    improved = True

        # After greedy walk, pick top_k nearest from candidate's neighbors + candidate
        all_candidates = candidate.neighbors + [candidate]
        all_candidates.sort(key=lambda n: l2_distance(query_vector, n.vector))
        return [(n.idx, n.vector) for n in all_candidates[:top_k]]
