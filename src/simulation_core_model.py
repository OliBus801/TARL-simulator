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

    def __init__(self, Nmax: int, device: str, time: int):
        super(SimulationCoreModel, self).__init__()
        self.direction_mpnn = DirectionMPNN(Nmax=Nmax, time=time).to(device)
        self.response_mpnn = ResponseMPNN(Nmax=Nmax, time=time).to(device)
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
        n = torch.sum(graph.x[:, self.direction_mpnn.NUMBER_OF_AGENT])
        num_roads = graph.num_roads
        x_roads = graph.x[:num_roads]
        # Use only the road graph for message passing
        node_feature = self.direction_mpnn(x_roads, graph.edge_index_routes, graph.edge_attr_routes)
        node_feature = self.response_mpnn(node_feature, graph.edge_index_routes, graph.edge_attr_routes)

        x_full = graph.x.clone()
        x_full[:num_roads] = node_feature

        graph.x = x_full
        
        return graph
    
    def set_time(self, time):
        self.time = time
        self.direction_mpnn.set_time(time)
        self.response_mpnn.set_time(time)


