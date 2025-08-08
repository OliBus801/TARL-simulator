import pytest
from src.agents.base import Agents
from src.direction_mpnn import DirectionMPNN
from src.response_mpnn import ResponseMPNN
from src.simulation_core_model import SimulationCoreModel
from src.transportation_simulator import TransportationSimulator
    
import torch
from torch_geometric.data import Data


@pytest.fixture(scope='class')
def agents():
    """Fixture for creating a DiscreteChoiceModel instance."""
    model = Agents()
    return model

@pytest.fixture
def direction_mpnn():
    """Fixture for creating a DirectionMPNN instance."""
    model = DirectionMPNN()
    return model

@pytest.fixture
def response_mpnn():
    """Fixture for creating a ResponseMPNN instance."""
    model = ResponseMPNN()
    return model

@pytest.fixture
def core():
    return SimulationCoreModel(100)

@pytest.fixture
def sioux_falls(simulator: TransportationSimulator, agents: Agents):
    simulator.config_network("data/Siouxfalls_network_PT.xml")
    agents.load("save/sioux_falls_agent_data.pt")
    simulator.agent = agents
    simulator.configure_core()
    return simulator


@pytest.fixture(scope="class")
def braess_graph():
    Nmax = 100
    feature_dim = 3 * Nmax + 7

    def build_node_feature(agent_pos, agent_t_arrival, agent_pos_at_arrival,
                           max_number_agent, number_agent, free_flow_time,
                           length_of_road, max_flow, selected_road, id_road):
        f = torch.zeros(feature_dim)
        f[0:Nmax] = agent_pos.clone().detach().float()
        f[Nmax:2*Nmax] = agent_t_arrival.clone().detach().float()
        f[2*Nmax:3*Nmax] = agent_pos_at_arrival.clone().detach().float()
        f[3*Nmax + 0] = max_number_agent
        f[3*Nmax + 1] = number_agent
        f[3*Nmax + 2] = free_flow_time
        f[3*Nmax + 3] = length_of_road
        f[3*Nmax + 4] = max_flow
        f[3*Nmax + 5] = selected_road
        f[3*Nmax + 6] = id_road
        return f

    # Nœuds : A=0, B=1, C=2, D=3
    x = torch.stack([
        build_node_feature(torch.zeros(100), torch.zeros(100), torch.zeros(100), 2, 1, 3.0, 100.0, 10.0, 1, 0),  # A
        build_node_feature(torch.zeros(100), torch.zeros(100), torch.zeros(100), 2, 1, 1.0, 100.0, 10.0, 2, 1),  # B
        build_node_feature(torch.zeros(100), torch.zeros(100), torch.zeros(100), 2, 2, 1.0, 100.0, 10.0, 0, 2),  # C
    ])

    # Put some agents on the road
    x[0, 0] = torch.tensor([1.0])
    x[1, 0] = torch.tensor([2.0])
    x[2, 0] = torch.tensor([3.0])
    x[2, 1] = torch.tensor([4.0])

    # Put the traffic 
    x[2, 2 * Nmax + 1] = torch.tensor([1.0])


    # Arêtes directionnelles du réseau de Braess
    edge_index = torch.tensor([
        [0, 1, 2],  # source
        [1, 2, 0],  # target
    ], dtype=torch.long)

    edge_attr = torch.rand(edge_index.size(1), 1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

@pytest.fixture
def simulator():
    simulator = TransportationSimulator()
    return simulator