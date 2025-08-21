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
        steps = 0
        while (
            simulator.agent.agent_features[1, simulator.agent.DONE] == 0
            and steps < 20
        ):
            simulator.run()
            steps += 1
        assert simulator.time == start_time + steps * simulator.timestep
        assert simulator.agent.agent_features[1, simulator.agent.DONE] == 1
        assert simulator.wardrop_gap_values
        fig = simulator.plot_wardrop_gap(output_dir=None)
        assert fig is not None
