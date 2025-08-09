import torch
from torch_geometric.nn import MessagePassing
from src.direction_mpnn import DirectionMPNN


class TestDirectionMPNN:
    def test_inheritance(self, direction_mpnn):
        assert issubclass(type(direction_mpnn), MessagePassing)

    def test_forward_shape(self, direction_mpnn, braess_graph):
        out = direction_mpnn(braess_graph.x, braess_graph.edge_index, braess_graph.edge_attr)
        assert out.shape == braess_graph.x.shape
