from .graph_constructor import GraphConstructor
from sklearn.neighbors import NearestNeighbors
import numpy as np
import networkx as nx


class KnnGraphConstructor(GraphConstructor):
    def __init__(self, k=5, metric='cosine', weighted=True, normalize_euclidean=False):
        """
        Initializes the KnnGraphConstructor.

        Parameters:
        k (int): The number of nearest neighbors to connect each node to.
        metric (str): The distance metric to use ('cosine' or 'euclidean').
        weighted (bool): Whether to use distances as weights on the edges.
        normalize_euclidean (bool): Whether to normalize the Euclidean distances by the dimensionality of the embeddings.
        """
        self.k = k
        self.metric = metric
        self.weighted = weighted
        self.normalize_euclidean = normalize_euclidean

    def construct_graph(self, embeddings):
        labels, embeddings_matrix = zip(*embeddings.items())
        embeddings_matrix = np.array(embeddings_matrix)

        # Normalize embeddings for cosine metric
        if self.metric == 'cosine':
            embeddings_matrix = embeddings_matrix / \
                np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)

        knn_model = NearestNeighbors(
            n_neighbors=self.k, metric=self.metric).fit(embeddings_matrix)
        distances, indices = knn_model.kneighbors(embeddings_matrix)

        if self.metric == 'euclidean' and self.normalize_euclidean:
            # Normalize Euclidean distances by the square root of the number of dimensions
            dimensionality = embeddings_matrix.shape[1]
            distances = distances / np.sqrt(dimensionality)

        G = nx.Graph()

        for i, label in enumerate(labels):
            G.add_node(label, embedding=embeddings[label])

            for j in range(1, self.k):  # Skip the node itself
                neighbor_label = labels[indices[i][j]]
                if self.weighted:
                    distance = distances[i][j]
                    G.add_edge(label, neighbor_label, weight=distance)
                else:
                    G.add_edge(label, neighbor_label)

        return G
