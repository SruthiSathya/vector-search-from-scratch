import math
import random
import networkx as nx
import matplotlib.pyplot as plt

def l2_distance(a, b):
    return math.sqrt(sum((x - y)**2 for x, y in zip(a, b)))

class Node:
    def __init__(self, vector, idx):
        self.vector = vector
        self.idx = idx
        self.neighbors = []

class HNSWIndexFromScratch:
    def __init__(self, M=5):
        self.nodes = []
        self.M = M

    def add(self, vector):
        new_node = Node(vector, len(self.nodes))

        if self.nodes:
            distances = [(l2_distance(vector, n.vector), n) for n in self.nodes]
            distances.sort(key=lambda x: x[0])

            for _, neighbor in distances[:self.M]:
                new_node.neighbors.append(neighbor)
                neighbor.neighbors.append(new_node)

        self.nodes.append(new_node)

    def search(self, query_vector, top_k=3, return_path=False):

        if not self.nodes:
            return []

        entry = random.choice(self.nodes)

        visited = set()
        traversal_edges = []
        candidate = entry
        improved = True

        while improved:
            improved = False
            visited.add(candidate.idx)

            for neighbor in candidate.neighbors:

                traversal_edges.append((candidate.idx, neighbor.idx))

                if neighbor.idx in visited:
                    continue

                if l2_distance(query_vector, neighbor.vector) < l2_distance(query_vector, candidate.vector):
                    candidate = neighbor
                    improved = True

        all_candidates = candidate.neighbors + [candidate]

        all_candidates.sort(key=lambda n: l2_distance(query_vector, n.vector))

        results = [(n.idx, n.vector) for n in all_candidates[:top_k]]

        if return_path:
            return results, visited, traversal_edges

        return results


def visualize_hnsw(index, visited, traversal_edges):

    G = nx.Graph()

    for node in index.nodes:
        G.add_node(node.idx)

        for neighbor in node.neighbors:
            G.add_edge(node.idx, neighbor.idx)

    pos = nx.spring_layout(G)

    plt.figure(figsize=(8,8))

    nx.draw(
        G,
        pos,
        node_color=["red" if n in visited else "lightblue" for n in G.nodes()],
        with_labels=True,
        node_size=600
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=traversal_edges,
        edge_color="green",
        width=2
    )

    plt.title("HNSW Search Traversal")
    plt.show()