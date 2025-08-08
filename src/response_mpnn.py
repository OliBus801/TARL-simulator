from torch_geometric.nn import MessagePassing
from src.feature_helpers import FeatureHelpers
import torch

class ResponseMPNN(MessagePassing):
    """
    MPNN that communicate which agent has been accepted previously.

    Parameters
    ----------
    Nmax : int
        The maximal number of agents in a queue.

    Attributes
    ----------
    time : int
        Time in seconds
    """
    def __init__(self, Nmax: int = 100, time: int = 0, compute_node_metrics: bool = False):
        MessagePassing.__init__(self, aggr='max', flow='target_to_source') # We reverse the flow and use a max-agregator
        self.NUMBER_OF_AGENT = 3 * Nmax + 1
        FeatureHelpers.__init__(self, Nmax=Nmax)
        self.time = time
        self.compute_node_metrics = compute_node_metrics

        if self.compute_node_metrics:
            self.update_history = []  # Initialize an empty list to store update history

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Notifies and updates the upstream nodes.

        Parameters
        ----------
        x : torch.Tensor
            Node features
        edge_index : torch.Tensor
            Edge index
        edge_attr : torch.Tensor
            Edge attribute
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """
        Create the message containing the agent selected.

        Parameters
        ----------
        x_i : torch.Tensor
            Upstream node features
        x_j : torch.Tensor
            Downstream node features

        Returns
        ----------
        message:
            The message that contains in the first column the index of the selected agent
        """
        # x_j is the source node and x_i is the target node
        
        # Look for the last agent in the queue
        end_queue = (x_j[:, self.NUMBER_OF_AGENT]).to(torch.int64) - 1
        is_there_agent = x_j[:, self.NUMBER_OF_AGENT] != 0
        last_agent = x_j[torch.arange(x_i.size(0)), end_queue]
        first_agent = x_i[:, self.HEAD_FIFO]
        
        # Message the agent to the next road
        message = (last_agent == first_agent) & is_there_agent
        if torch.any(message):
            pass
        message = message.float()
        return message.unsqueeze(1)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the updated node features

        Parameters
        ----------
        aggr_out : torch.Tensor
            Aggregated message containing ones if we need to update the node
        x : torch.Tensor
            Node features

        Returns
        ----------
        x_updated:
            The message that contains in the first column the index of the selected agent
        """
        # Clone the input to avoid in-place modification
        x_updated = x.clone()

        # Mask the agents that have been transferred
        mask_update = (aggr_out > 0).squeeze(1)


        # Compute the slicing
        AGENT_POSITION_UPSTREAM = slice(self.AGENT_POSITION.start, self.AGENT_POSITION.stop - 1)
        AGENT_TIME_ARRIVAL_UPSTREAM = slice(self.AGENT_TIME_ARRIVAL.start, self.AGENT_TIME_ARRIVAL.stop - 1)
        AGENT_POSITION_AT_ARRIVAL_UPSTREAM = slice(self.AGENT_POSITION_AT_ARRIVAL.start, self.AGENT_POSITION_AT_ARRIVAL.stop - 1)
        AGENT_POSITION_DOWNSTREAM = slice(self.AGENT_POSITION.start + 1, self.AGENT_POSITION.stop)
        AGENT_TIME_ARRIVAL_DOWNSTREAM = slice(self.AGENT_TIME_ARRIVAL.start + 1, self.AGENT_TIME_ARRIVAL.stop)
        AGENT_POSITION_AT_ARRIVAL_DOWNSTREAM = slice(self.AGENT_POSITION_AT_ARRIVAL.start + 1, self.AGENT_POSITION_AT_ARRIVAL.stop)

        # Update the position of the agent
        x_updated[mask_update, AGENT_POSITION_UPSTREAM] = x[mask_update, AGENT_POSITION_DOWNSTREAM]  # Update the position of the agent
        x_updated[mask_update, AGENT_TIME_ARRIVAL_UPSTREAM] = x[mask_update, AGENT_TIME_ARRIVAL_DOWNSTREAM]  # Update the time of arrival of the agent
        x_updated[mask_update, AGENT_POSITION_AT_ARRIVAL_UPSTREAM] = x[mask_update, AGENT_POSITION_AT_ARRIVAL_DOWNSTREAM]  # Update the position of the agent at arrival
        x_updated[mask_update, self.NUMBER_OF_AGENT] = x[mask_update, self.NUMBER_OF_AGENT] - 1  # Update the number of agents on the road

        # Store the update mask and current time in the history
        if self.compute_node_metrics:
            self.update_history.append((self.time, mask_update.clone()))

        return x_updated


    def set_time(self, time: int):
        """
        Set the time

        Parameters
        ----------
        time : int
            Time to set
        """
        self.time = time
