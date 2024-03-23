from abc import ABC, abstractmethod


class GraphConstructor(ABC):
    @abstractmethod
    def construct_graph(self, embeddings):
        """
        Constructs a graph from embeddings.

        Parameters:
        embeddings (dict): A dictionary mapping items (e.g., words) to their embeddings.

        Returns:
        nx.Graph: A NetworkX graph constructed from the embeddings.
        """
        pass
