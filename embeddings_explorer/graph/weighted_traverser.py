from .traverser import Traverser
import networkx as nx


class WeightedTraverser(Traverser):
    def traverse(self, G, start_node, end_node=None):
        # If end_node is not specified, raise an error as finding a 'general' traversal
        # without a target doesn't typically apply to weighted traversals focusing on shortest paths
        if end_node is None:
            raise ValueError(
                "End node must be specified for weighted traversal.")

        # Attempt to find the shortest path considering weights
        try:
            path = nx.dijkstra_path(G, start_node, end_node)
            return path
        except nx.NetworkXNoPath:
            print(f"No path found from {start_node} to {end_node}.")
            return []
