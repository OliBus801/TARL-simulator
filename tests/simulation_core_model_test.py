import torch.nn as nn
import torch


class TestSimulationCoreModel:
    def test_initialization(self, core):
        assert issubclass(type(core), nn.Module)

    def test_forward(self, core, braess_graph):
        core(braess_graph)
        
        # Check the new queue
        assert torch.all(braess_graph.x[0, 0:2] == torch.tensor([1.0, 0.0]))
        assert torch.all(braess_graph.x[1, 0:2] == torch.tensor([2.0, 0.0]))
        assert torch.all(braess_graph.x[2, 0:2] == torch.tensor([3.0, 4.0]))

        core(braess_graph)

        # Check the new queue
        assert torch.all(braess_graph.x[0, 0:2] == torch.tensor([1.0, 0.0]))
        assert torch.all(braess_graph.x[1, 0:2] == torch.tensor([2.0, 0.0]))
        assert torch.all(braess_graph.x[2, 0:2] == torch.tensor([3.0, 4.0]))

        core(braess_graph)

        # Check the new queue
        assert torch.all(braess_graph.x[0, 0:2] == torch.tensor([1.0, 3.0]))
        assert torch.all(braess_graph.x[1, 0:2] == torch.tensor([2.0, 0.0]))
        assert torch.all(braess_graph.x[2, 0:2] == torch.tensor([4.0, 0.0]))