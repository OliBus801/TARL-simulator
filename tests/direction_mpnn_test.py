from src.direction_mpnn import DirectionMPNN
from torch_geometric.nn import MessagePassing
import torch

class TestDirectionMPNN:

    def test_direction_mpnn(self, direction_mpnn):
        """Test the DirectionMPNN class."""
        assert issubclass(type(direction_mpnn), MessagePassing)

    def test_message(self, direction_mpnn, braess_graph):
        """Test the propagate method of DirectionMPNN."""
        direction_mpnn.time = 2
        direction_mpnn(braess_graph.x, braess_graph.edge_index, braess_graph.edge_attr)

        assert 1 == 1