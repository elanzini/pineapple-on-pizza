from abc import ABC, abstractmethod


class Traverser(ABC):
    @abstractmethod
    def traverse(self, G, start_node, end_node=None):
        """
        Traverse the graph G starting from start_node, optionally aiming to reach end_node.

        Parameters:
        G (nx.Graph): The graph to be traversed.
        start_node (Any): The starting node for the traversal.
        end_node (Any, optional): The end node to reach. If not provided, the traversal might not aim for a specific end node.

        Returns:
        list: The path taken during the traversal, as a list of nodes.
        """
        pass
