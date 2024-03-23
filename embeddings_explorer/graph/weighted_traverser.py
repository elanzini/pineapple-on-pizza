from .traverser import Traverser
import networkx as nx


class WeightedTraverser(Traverser):
    def traverse(self, G, start_node, end_node):
        # Attempt to find the shortest path considering weights
        try:
            path = nx.shortest_path(G, start_node, end_node, weight='weight')
            # Calculate total distance
            total_distance = sum(G[path[i]][path[i+1]]['weight']
                                 for i in range(len(path)-1))
            return path, total_distance
        except nx.NetworkXNoPath:
            print(f"No path found from {start_node} to {end_node}.")
            return []
