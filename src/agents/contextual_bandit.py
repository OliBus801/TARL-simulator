from src.agents.base import Agents
from torch_geometric.data import Data
from src.feature_helpers import AgentFeatureHelpers, FeatureHelpers
import torch

class ContextualBandits(Agents):
    def __init__(self, device: str):
        super().__init__(device)

    def choice(self, graph: Data, h: FeatureHelpers):
        """
        Choose the next direction to take for each road using predicted contextual costs.

        Parameters
        ----------
        graph : Data
            Graph of the traffic network
        h : FeatureHelpers
            Helpers for selecting index
        """

        

        