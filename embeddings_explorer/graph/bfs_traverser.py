from .traverser import Traverser
import networkx as nx


class BfsTraverser(Traverser):
    def traverse(self, G, start_node, end_node=None):
        # If end_node is not specified, perform BFS from start_node without a specific goal
        if end_node is None:
            # Generate a BFS tree from start_node and return the edges
            bfs_tree = nx.bfs_tree(G, start_node)
            return list(bfs_tree.edges())
        else:
            # Attempt to find a path to end_node
            # This implicitly uses BFS for unweighted graphs
            try:
                path = nx.shortest_path(G, start_node, end_node)
                return path
            except nx.NetworkXNoPath:
                print(f"No path found from {start_node} to {end_node}.")
                return []
