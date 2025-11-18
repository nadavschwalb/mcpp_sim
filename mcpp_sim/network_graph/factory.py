from .base import BaseNetworkGraph
from .eucleadian_graph import EucleadianGraph

def create_network_graph(graph_type: str) -> BaseNetworkGraph:
    if graph_type == "eucleadian":
        return EucleadianGraph()
    if graph_type is None:
        return None
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")