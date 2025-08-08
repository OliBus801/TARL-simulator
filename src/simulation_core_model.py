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

    def __init__(self, Nmax: int, device: str, time: int, compute_node_metrics: bool):
        super(SimulationCoreModel, self).__init__()
        self.direction_mpnn = DirectionMPNN(Nmax=Nmax, time=time).to(device)
        self.response_mpnn = ResponseMPNN(Nmax=Nmax, time=time, compute_node_metrics=compute_node_metrics).to(device)
        self.time = time
        self.Nmax = Nmax

    def forward(self, graph: Data):
        """
        Computes the next traffic network position.

        Parameters
        ----------
        graph : Data
            The graph representing the traffic network
        edge_index : torch.Tensor
            Edge index
        edge_attr : torch.Tensor
            Edge attribute
        """
        """
        Forward pass of the simulation core model.
        Args:
            graph: The input graph data.
        Returns:
            The output graph data after processing through the MPNN layers.
        """
        n = torch.sum(graph.x[:, self.direction_mpnn.NUMBER_OF_AGENT])
        # Process the graph through the DirectionMPNN layer
        node_feature = self.direction_mpnn(graph.x, graph.edge_index, graph.edge_attr)
        
        # Process the graph through the ResponseMPNN layer
        node_feature = self.response_mpnn(node_feature, graph.edge_index, graph.edge_attr)

        # Update the graph with the new features
        updated_graph = Data(x=node_feature, edge_index=graph.edge_index, edge_attr=graph.edge_attr)
        
        if n != torch.sum(updated_graph.x[:, self.direction_mpnn.NUMBER_OF_AGENT]):
            pass
        return updated_graph
    
    
    
    def set_time(self, time):
        self.time = time
        self.direction_mpnn.set_time(time)
        self.response_mpnn.set_time(time)


