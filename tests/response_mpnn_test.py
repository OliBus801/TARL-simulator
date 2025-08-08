from torch_geometric.nn import MessagePassing

class TestResponseMPNN:

    def test_direction_mpnn(self, response_mpnn):
        """Test the DirectionMPNN class."""
        assert issubclass(type(response_mpnn), MessagePassing)

    def test_message(self, direction_mpnn, response_mpnn, braess_graph):
        """Test the propagate method of DirectionMPNN."""
        response_mpnn.time = 2
        direction_mpnn.time = 2
        direction_mpnn(braess_graph.x, braess_graph.edge_index, braess_graph.edge_attr)
        assert braess_graph.x[0,1] == 3.0

        response_mpnn(braess_graph.x, braess_graph.edge_index, braess_graph.edge_attr)
        assert braess_graph.x[2,301] == 1

        