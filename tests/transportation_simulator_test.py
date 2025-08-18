from src.transportation_simulator import TransportationSimulator
from src.simulation_core_model import SimulationCoreModel


class TestTransportationSimulator:
    def test_config_network(self, simulator: TransportationSimulator):
        assert simulator.graph.x.size(0) == 6
        assert simulator.graph.edge_index.size(1) == 6
        assert simulator.graph.edge_attr.size(0) == 6
        assert simulator.graph.edge_index_routes.size(1) == 2

    def test_config_core(self, simulator: TransportationSimulator):
        assert isinstance(simulator.model_core, SimulationCoreModel)

    def test_run(self, simulator: TransportationSimulator):
        start_time = simulator.time
        simulator.run()
        assert simulator.time == start_time + simulator.timestep
        assert simulator.agent.agent_features[0, simulator.agent.DONE] == 1
