import networkx as nx
from .traverser import Traverser


class BfsTraverser(Traverser):
    def traverse(self, G, start_node, end_node):
        """
        Traverse the graph G using BFS from start_node to end_node.

        Parameters:
        - G (nx.Graph): The graph to be traversed.
        - start_node (Any): The starting node for the traversal.
        - end_node (Any): The end node to reach.

        Returns:
        - tuple: A tuple containing:
            - list: The path taken during the traversal, as a list of nodes.
            - int: The distance of the path, measured as the number of edges.
        """
        try:
            # Find the shortest path using BFS (default for unweighted graphs)
            path = nx.shortest_path(G, start_node, end_node)
            # The distance is the number of edges, which is one less than the number of nodes in the path
            distance = len(path) - 1
            return path, distance
        except nx.NetworkXNoPath:
            print(f"No path found from {start_node} to {end_node}.")
            return [], 0
