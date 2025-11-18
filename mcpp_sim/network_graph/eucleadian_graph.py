from .base import BaseNetworkGraph
import math

class EucleadianGraph(BaseNetworkGraph):
    def __init__(self):
        super().__init__()

    def _compute_weight(self, node1_id, node2_id):
        pos1 = self.graph.nodes[node1_id]['pos']
        pos2 = self.graph.nodes[node2_id]['pos']
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def compute_edges(self):
        for node in self.graph.nodes:
            for other_node in self.graph.nodes:
                if node != other_node:
                    self.add_edge(node, other_node, weight=self._compute_weight(node, other_node))
