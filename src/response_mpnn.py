from torch_geometric.nn import MessagePassing
from src.feature_helpers import FeatureHelpers
import torch


@torch.compile
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
    def __init__(self, Nmax: int = 100, time: int = 0):
        MessagePassing.__init__(self, aggr='max', flow='target_to_source') # We reverse the flow and use a max-agregator
        self.NUMBER_OF_AGENT = 3 * Nmax + 1
        FeatureHelpers.__init__(self, Nmax=Nmax)
        self.time = time
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
        Compute a per-edge mask (E, 1) telling whether the downstream node (x_j)
        has accepted the agent that the upstream node (x_i) intended to send (its FIFO head).

        Parameters
        ----------
        x_i : torch.Tensor
            Upstream node features
        x_j : torch.Tensor
            Downstream node features

        Returns
        -------
        torch.Tensor
            A float tensor of shape [E, 1] with values in {0.0, 1.0}.
            1.0 means that, for edge (i -> j), the downstream node j has accepted
            the agent that upstream i intended to send (its FIFO head); 0.0 otherwise.
            Use as a gate to apply state updates (e.g., pop from i’s FIFO).
        """
        # x_i: upstream (source), x_j: downstream (target) for edge (i -> j)
        E, device = x_i.size(0), x_i.device

        # Agent counts (int64 for safe indexing)
        cnt_up = x_i[:, self.NUMBER_OF_AGENT].to(torch.int64)
        cnt_dn = x_j[:, self.NUMBER_OF_AGENT].to(torch.int64)
        has_up = cnt_up > 0        # upstream has something to send
        has_dn = cnt_dn > 0        # downstream has a valid tail

        # Upstream head ID and downstream tail ID
        head_id = x_i[:, self.HEAD_FIFO].to(torch.int64)
        tail_idx = torch.clamp(cnt_dn - 1, min=0)
        rows = torch.arange(E, device=device)
        tail_id_all = x_j[rows, tail_idx]
        sentinel = torch.full((E,), -1, dtype=torch.int64, device=device)
        tail_id = torch.where(has_dn, tail_id_all.to(torch.int64), sentinel)

        # Gate: downstream accepted the agent upstream intended to send
        message = (has_up & has_dn & (tail_id == head_id)).unsqueeze(1)

        # Float mask (E, 1)
        return message.to(dtype=x_i.dtype)


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
        # Mask the agents that have been transferred
        mask_update = (aggr_out > 0).squeeze(1)

        # If no agents needs to be updated, return the original features
        if not mask_update.any():
            return x

        # Compute the slicing
        AGENT_POSITION_UPSTREAM = slice(self.AGENT_POSITION.start, self.AGENT_POSITION.stop - 1)
        AGENT_TIME_ARRIVAL_UPSTREAM = slice(self.AGENT_TIME_ARRIVAL.start, self.AGENT_TIME_ARRIVAL.stop - 1)
        AGENT_TIME_DEPARTURE_UPSTREAM = slice(self.AGENT_TIME_DEPARTURE.start, self.AGENT_TIME_DEPARTURE.stop - 1)
        AGENT_POSITION_DOWNSTREAM = slice(self.AGENT_POSITION.start + 1, self.AGENT_POSITION.stop)
        AGENT_TIME_ARRIVAL_DOWNSTREAM = slice(self.AGENT_TIME_ARRIVAL.start + 1, self.AGENT_TIME_ARRIVAL.stop)
        AGENT_TIME_DEPARTURE_DOWNSTREAM = slice(self.AGENT_TIME_DEPARTURE.start + 1, self.AGENT_TIME_DEPARTURE.stop)

        # Update the position of the agent without cloning the full tensor
        with torch.no_grad():
            x[mask_update, AGENT_POSITION_UPSTREAM] = x[mask_update, AGENT_POSITION_DOWNSTREAM]
            x[mask_update, AGENT_TIME_ARRIVAL_UPSTREAM] = x[mask_update, AGENT_TIME_ARRIVAL_DOWNSTREAM]
            x[mask_update, AGENT_TIME_DEPARTURE_UPSTREAM] = x[mask_update, AGENT_TIME_DEPARTURE_DOWNSTREAM]
            x[mask_update, self.NUMBER_OF_AGENT] = x[mask_update, self.NUMBER_OF_AGENT] - 1

        # Store the update mask and current time in the history
        self.update_history.append((self.time, mask_update.clone()))

        return x


    def set_time(self, time: int):
        """
        Set the time

        Parameters
        ----------
        time : int
            Time to set
        """
        self.time = time
