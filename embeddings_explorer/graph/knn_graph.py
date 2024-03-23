from .graph_constructor import GraphConstructor
from sklearn.neighbors import NearestNeighbors
import numpy as np
import networkx as nx


class KnnGraphConstructor(GraphConstructor):
    def __init__(self, k=5):
        self.k = k

    def construct_graph(self, embeddings):
        # Prepare embeddings for KNN
        labels, embeddings_matrix = zip(*embeddings.items())
        embeddings_matrix = np.array(embeddings_matrix)

        # Use KNN to find nearest neighbors
        knn_model = NearestNeighbors(
            n_neighbors=self.k + 1, algorithm='ball_tree').fit(embeddings_matrix)
        distances, indices = knn_model.kneighbors(embeddings_matrix)

        # Create a graph
        G = nx.Graph()

        for i, label in enumerate(labels):
            G.add_node(label, embedding=embeddings[label])

            # Add edges from this node to its k-nearest neighbors
            for j in range(1, self.k + 1):  # start from 1 to skip the node itself
                G.add_edge(label, labels[indices[i][j]],
                           weight=distances[i][j])

        return G
