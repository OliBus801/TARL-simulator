from torch_geometric.nn import MessagePassing


class TestResponseMPNN:
    def test_inheritance(self, response_mpnn):
        assert issubclass(type(response_mpnn), MessagePassing)

    def test_forward_and_history(self, direction_mpnn, response_mpnn, braess_graph):
        out = direction_mpnn(braess_graph.x, braess_graph.edge_index, braess_graph.edge_attr)
        assert len(response_mpnn.update_history) == 0
        out2 = response_mpnn(out, braess_graph.edge_index, braess_graph.edge_attr)
        assert out2.shape == out.shape
        assert len(response_mpnn.update_history) == 1
