import torch.nn as nn
from src.direction_mpnn import DirectionMPNN
from src.response_mpnn import ResponseMPNN
from torch_geometric.data import Data

from src.feature_helpers import FeatureHelpers # A Supprimer
import torch


class SimulationCoreModel(nn.Module):
    """
    Simulation Core that captures road dynamics but does not handle insertion or withdrawal agent.

    Parameters
    ----------
    Nmax : int
        The maximal number of agents in a queue.

    Attributes
    ----------
    time : int
        Time in seconds
    direction_mpnn : DirectionMPNN
        A MPNN layer that handles the direction communication to downstream nodes
    response_mpnn : ResponseMPNN
        A MPNN layer that handles the response communication to upstream nodes
    Nmax : int 
        The maximal number of agents in a queue.
    """

    def __init__(self, Nmax: int, device: str, time: int, torch_compile: bool = False):
        super(SimulationCoreModel, self).__init__()
        self.direction_mpnn = DirectionMPNN(Nmax=Nmax, time=time).to(device)
        self.response_mpnn = ResponseMPNN(Nmax=Nmax, time=time).to(device)
        if torch_compile:
            self.direction_mpnn = torch.compile(self.direction_mpnn)
            self.response_mpnn = torch.compile(self.response_mpnn)
        self.time = time
        self.Nmax = Nmax

    def forward(self, graph: Data):
        """
        Computes the next traffic network position.

        Parameters
        ----------
        graph : Data
            The graph representing the traffic network.
        """
        # Forward pass of the simulation core model.
        num_roads = graph.num_roads
        x_roads = graph.x[:num_roads]

        # Retrieve or compute pre-computed static factors
        if hasattr(graph, "critical_number") and hasattr(graph, "congestion_constant"):
            critical_number = graph.critical_number[:num_roads]
            congestion_constant = graph.congestion_constant[:num_roads]
        else:
            h = self.direction_mpnn
            critical_number = (
                graph.x[:num_roads, h.MAX_FLOW]
                * graph.x[:num_roads, h.FREE_FLOW_TIME_TRAVEL]
                / 3600
            )
            congestion_constant = graph.x[
                :num_roads, h.FREE_FLOW_TIME_TRAVEL
            ] * (graph.x[:num_roads, h.MAX_NUMBER_OF_AGENT] + 10 - critical_number)

        # Use only the road graph for message passing
        node_feature = self.direction_mpnn(
            x_roads,
            graph.edge_index_routes,
            graph.edge_attr_routes,
            critical_number=critical_number,
            congestion_constant=congestion_constant,
        )
        node_feature = self.response_mpnn(
            node_feature, graph.edge_index_routes, graph.edge_attr_routes
        )

        graph.x[:num_roads] = node_feature
        
        return graph
    
    def set_time(self, time):
        self.time = time
        self.direction_mpnn.set_time(time)
        self.response_mpnn.set_time(time)


