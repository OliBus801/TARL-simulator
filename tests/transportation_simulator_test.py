from src.transportation_simulator import TransportationSimulator
from src.simulation_core_model import SimulationCoreModel
import torch

class TestTransportationsioux_falls:

    def test_config_network(self, sioux_falls: TransportationSimulator):
        assert sioux_falls.graph.x.size(0) == 334
        assert sioux_falls.graph.edge_index.size(0) == 2
        assert sioux_falls.graph.edge_index.size(1) == sioux_falls.graph.edge_attr.size(0)
        assert sioux_falls.graph.edge_attr.size(1) == 1
        assert torch.all(sioux_falls.graph.edge_index < 334)

    def test_config_core(self, sioux_falls: TransportationSimulator):
        assert isinstance(sioux_falls.model_core, SimulationCoreModel)

    def test_run(self, sioux_falls: TransportationSimulator):
        sioux_falls.run()


