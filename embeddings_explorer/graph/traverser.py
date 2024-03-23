from abc import ABC, abstractmethod


class Traverser(ABC):
    @abstractmethod
    def traverse(self, G, start_node, end_node):
        """
        Traverse the graph G starting from start_node, optionally aiming to reach end_node.

        Parameters:
        - G (nx.Graph): The graph to be traversed.
        - start_node (Any): The starting node for the traversal.
        - end_node (Any): The end node to reach.

        Returns:
        - tuple: A tuple containing:
            - list: The path taken during the traversal, as a list of nodes.
            - float: The distance of the path. For unweighted graphs or BFS, this could be the edge count.
        """
        pass
