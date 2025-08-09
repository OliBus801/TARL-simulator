import torch
import torch.nn as nn


class TestSimulationCoreModel:
    def test_initialization(self, core):
        assert isinstance(core, nn.Module)

    def test_forward(self, core, braess_graph):
        out = core(braess_graph)
        assert torch.is_tensor(out.x)
        assert out.x.shape == braess_graph.x.shape
