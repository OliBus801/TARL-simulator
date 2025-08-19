import torch
from torch_geometric.nn import MessagePassing
from src.feature_helpers import FeatureHelpers


class DirectionMPNN(MessagePassing, FeatureHelpers):
    """
    MPNN that communicate the next direction of agents to downstream nodes
    and update these nodes according aggregate messages.

    Parameters
    ----------
    Nmax : int
        The maximal number of agents in a queue.

    Attributes
    ----------
    time : int
        Time in seconds
    """

    def __init__(self, Nmax = 100, time: int = 0):
        # Initialise the inherited classes 
        MessagePassing.__init__(self)
        FeatureHelpers.__init__(self, Nmax=Nmax)
        self.time = time
        self.Nmax = Nmax
        self.road_optimality_data = None # To store the delta_travel_time for each road


    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Compute the message to sent to the downstream nodes. It contains only the identifier of 
        the agent and the intersection coefficient.

        Parameters
        ----------
        x_i : torch.Tensor
            The downstream node features
        x_j : torch.Tensor
            The upstream node features
        edge_attr: torch.Tensor
            Intersection coefficients

        Returns
        ----------
        message:
            The message that contains in the first column the agent identifier and on the last 
            column the intersection coefficient
        """
        
        
        # Time departure pre-computed at agent insertion
        time_departure = x_j[:, self.HEAD_FIFO_DEPARTURE].unsqueeze(1)
        time_arrival = x_j[:, self.HEAD_FIFO_ARRIVAL]

        # Agent identifier
        agent_id = x_j[:, self.HEAD_FIFO].unsqueeze(1)

        # Compute the probability to take the road
        mask = torch.logical_and(time_departure < self.time,  # Check if the departure time is before the actual time
                                 (x_i[:, self.NUMBER_OF_AGENT] < x_i[:, self.MAX_NUMBER_OF_AGENT] - self.CONGESTION_FILE).unsqueeze(1)) # Check if the fifo is not full
        mask = torch.logical_and(mask, (x_j[:, self.SELECTED_ROAD] == x_i[:, self.ROAD_INDEX]).unsqueeze(1)) # Check if the direction is the road
        mask = torch.logical_and(mask, (x_j[:, self.NUMBER_OF_AGENT] > 0).unsqueeze(1))# Check if there is agent inside the queue

        # Case when there is congestion
        submask = torch.logical_and((time_departure - self.time < -10).flatten(), x_j[:, self.MAX_NUMBER_OF_AGENT] - self.CONGESTION_FILE <= x_j[:, self.NUMBER_OF_AGENT] )
        submask = torch.logical_and(submask, x_j[:, self.MAX_NUMBER_OF_AGENT] - x_j[:, self.NUMBER_OF_AGENT] <= x_i[:, self.MAX_NUMBER_OF_AGENT] - x_i[:, self.NUMBER_OF_AGENT])
        submask = torch.logical_and(submask, (x_j[:, self.SELECTED_ROAD] == x_i[:, self.ROAD_INDEX]))
        mask = torch.logical_or(mask, submask.unsqueeze(1))  # If there is some 
        prob = edge_attr* mask.float()

        # Compute the road optimality data
        travel_time = time_departure.squeeze(1) - time_arrival
        delta_travel_time = torch.clamp(travel_time - x_j[:, self.FREE_FLOW_TIME_TRAVEL], min=0)
        self.road_optimality_data = {"delta_travel_time": delta_travel_time}


        message = torch.cat((agent_id, prob), dim=1)
        return message


    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor, ptr=None, dim_size=None) -> torch.Tensor:
        """
        Aggregates all messages and extracts the agent to be transferred.

        Parameters
        ----------
        inputs : torch.Tensor
            The message extract from upstream nodes to downstream nodes with shape (num_edges, 2).
        index : torch.Tensor
            Index tensor specifying the downstream nodes index (num_edges,).
        ptr : torch.Tensor, optional
            Pointer tensor for variable-size inputs, with shape (num_nodes + 1,).
        dim_size : int, optional
            The size of the dimension to aggregate over, typically the number of nodes.

        Returns
        -------
        chosen_agent : torch.Tensor
            The chosen agent to include inside the downstream road with shape (num_nodes, 1).
        """

        # Define the name of the columns
        AGENT_ID = 0
        PROB = 1

        # Sort messages by index
        sorted_index, sorted_idx = torch.sort(index)
        inputs_sorted = inputs[sorted_idx]

        # We compute the cumulative sum of all probabilities
        cum_sum = torch.cumsum(inputs_sorted[:, PROB], dim=0)

        # Find the first index of each group
        is_first = torch.cat([
            torch.tensor([True], device=sorted_index.device),  # premier élément est début de groupe
            sorted_index[1:] != sorted_index[:-1]
        ])
        first_indices = torch.nonzero(is_first).squeeze(1)

        # Find the last index of each group
        if_last = torch.cat([
            sorted_index[:-1] != sorted_index[1:],  # dernier élément est fin de groupe
            torch.tensor([True], device=sorted_index.device)
        ])
        last_indices = torch.nonzero(if_last).squeeze(1)

        # Compute the sum
        buffer = cum_sum[last_indices]
        last_sum = torch.zeros_like(buffer)
        last_sum[1:] = buffer[:-1]
        cum_sum = cum_sum - last_sum[sorted_index]  

        # Normalize the sum
        sum_over_group = cum_sum[last_indices]
        mask = ~torch.isclose(sum_over_group, torch.tensor(0.0, device=sum_over_group.device), atol=1e-5)
        mask = mask[sorted_index]
        cum_sum  = torch.where(mask, cum_sum / sum_over_group[sorted_index], cum_sum)

        # Randomly select an element in the group
        r = torch.rand_like(sum_over_group)
        r = r[sorted_index]
        r = torch.where(cum_sum > r, 1, 0)

        r = torch.cumsum(r, dim=0)
        r_sum = torch.zeros_like(sum_over_group)
        r_sum[1:] = r[last_indices[:-1]]
        r = r - r_sum[sorted_index]
        selected_indices = torch.where(r == 1, 1, 0)

        # Compute the selected messages
        chosen_agent = torch.zeros_like(first_indices, dtype=torch.float)
        selected_indices = torch.nonzero(r == 1, as_tuple=False).squeeze()
        chosen_agent[sorted_index[selected_indices]] = inputs_sorted[selected_indices, AGENT_ID]

        return chosen_agent

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Updates the node features thanks to the selected agents.

        Parameters
        ----------
        aggr_out : torch.Tensor
            Chosen agents selected to be include the queue road with shape (num_nodes, 1).
        x : torch.Tensor
            Node features with shape (num_nodes, num_features).

        Returns
        ----------
        x_updated : torch.Tensor
            The updated node features
        """
        x_updated = x.clone()

        # Compute the number of agents
        end_queue = x_updated[:, self.NUMBER_OF_AGENT].to(torch.int64)
        start_counts = x_updated[:, self.NUMBER_OF_AGENT]
        idx = torch.arange(x.size(0), device=x.device)
        x_updated[idx, end_queue] = aggr_out
        x_updated[idx, self.Nmax + end_queue] = self.time

        # Compute and store departure time for the inserted agent
        critical_number = x_updated[idx, self.MAX_FLOW] * x_updated[idx, self.FREE_FLOW_TIME_TRAVEL] / 3600
        time_congestion = x_updated[idx, self.FREE_FLOW_TIME_TRAVEL] * (
            x_updated[idx, self.MAX_NUMBER_OF_AGENT] + 10 - critical_number
        ) / (x_updated[idx, self.MAX_NUMBER_OF_AGENT] + 10 - start_counts)
        travel_time = torch.max(
            torch.stack((x_updated[idx, self.FREE_FLOW_TIME_TRAVEL], time_congestion)), dim=0
        ).values
        x_updated[idx, self.AGENT_TIME_DEPARTURE.start + end_queue] = self.time + travel_time

        # Update the number of agents on the road
        is_agent = aggr_out != 0  # So agent who gets ID : 0 are not agent and can break the simulation
        x_updated[is_agent, self.NUMBER_OF_AGENT] = start_counts[is_agent] + 1
        return x_updated

    def set_time(self, time):
        """
        Set time.

        Parameters
        ----------
        time : int
            Time to set
        """
        self.time = time


    def forward(self, x, edge_index, edge_attr):
        """
        Passes first agents on upstream road to their downstream.

        Parameters
        ----------
        x : torch.Tensor
            Node features
        edge_index : torch.Tensor
            Edge index
        edge_attr : torch.Tensor
            Edge attribute
        """
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

