import networkx as nx

class BaseNetworkGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def add_node(self, node_id, **attrs):
        self.graph.add_node(node_id, **attrs)

    def update_node(self, node_id, **attrs):
        self.graph.nodes[node_id].update(**attrs)

    def add_edge(self, node1_id, node2_id, **attrs):
        self.graph.add_edge(node1_id, node2_id, **attrs)

    def get_graph(self):
        return self.graph

    def compute_edges(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _compute_weight(self, node1_id, node2_id):
        raise NotImplementedError("This method should be implemented by subclasses.")

