import torch
from torch_geometric.nn import MessagePassing
from src.feature_helpers import FeatureHelpers
from torch_scatter import scatter_add, scatter_max


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
        time_departure = x_j[:, self.HEAD_FIFO_DEPARTURE_TIME].unsqueeze(1)
        time_arrival = x_j[:, self.HEAD_FIFO_ARRIVAL_TIME]

        # Agent identifier
        agent_id = x_j[:, self.HEAD_FIFO].unsqueeze(1)

        # Compute mask : Can we send the agent on the downstream node ?
        mask = torch.logical_and(time_departure <= self.time,  # Check if we reached the departure time
                                 (x_i[:, self.NUMBER_OF_AGENT] < x_i[:, self.MAX_NUMBER_OF_AGENT] - self.CONGESTION_FILE).unsqueeze(1)) # Check if the fifo is not full
        mask = torch.logical_and(mask, (x_j[:, self.SELECTED_ROAD] == x_i[:, self.ROAD_INDEX]).unsqueeze(1)) # Check if the direction is the road
        mask = torch.logical_and(mask, (x_j[:, self.NUMBER_OF_AGENT] > 0).unsqueeze(1))# Check if there is agent inside the queue

        # Case when there is congestion
        submask = torch.logical_and((time_departure - self.time < -10).flatten(), x_j[:, self.MAX_NUMBER_OF_AGENT] - self.CONGESTION_FILE <= x_j[:, self.NUMBER_OF_AGENT] )
        submask = torch.logical_and(submask, x_j[:, self.MAX_NUMBER_OF_AGENT] - x_j[:, self.NUMBER_OF_AGENT] <= x_i[:, self.MAX_NUMBER_OF_AGENT] - x_i[:, self.NUMBER_OF_AGENT])
        submask = torch.logical_and(submask, (x_j[:, self.SELECTED_ROAD] == x_i[:, self.ROAD_INDEX]))
        mask = torch.logical_or(mask, submask.unsqueeze(1))  # If there is some 
        prob = edge_attr * mask.float()

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

        # Ensure we know the number of nodes to aggregate over
        if dim_size is None:
            dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

        # Compute the total probability per downstream node using scatter_add
        prob_per_node = scatter_add(inputs[:, PROB], index, dim=0, dim_size=dim_size)

        # Sample one agent per node using the Gumbel-max trick
        eps = 1e-12
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(inputs[:, PROB] + eps)))
        scores = torch.log(inputs[:, PROB] + eps) + gumbel_noise
        _, argmax = scatter_max(scores, index, dim=0, dim_size=dim_size)

        # Gather the chosen agent id for nodes that received messages
        chosen_agent = torch.zeros(dim_size, device=inputs.device)
        mask = prob_per_node > 0
        chosen_agent[mask] = inputs[argmax[mask], AGENT_ID]

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
        # Perform in-place updates without tracking gradients
        with torch.no_grad():
            end_queue = x[:, self.NUMBER_OF_AGENT].to(torch.int64)
            start_counts = x[:, self.NUMBER_OF_AGENT]
            idx = torch.arange(x.size(0), device=x.device)
            x[idx, end_queue] = aggr_out
            x[idx, self.Nmax + end_queue] = self.time

            # Compute and store departure time for the inserted agent
            critical_number = x[idx, self.MAX_FLOW] * x[idx, self.FREE_FLOW_TIME_TRAVEL] / 3600
            time_congestion = x[idx, self.FREE_FLOW_TIME_TRAVEL] * (
                x[idx, self.MAX_NUMBER_OF_AGENT] + 10 - critical_number
            ) / (x[idx, self.MAX_NUMBER_OF_AGENT] + 10 - start_counts)
            travel_time = torch.max(
                torch.stack((x[idx, self.FREE_FLOW_TIME_TRAVEL], time_congestion)), dim=0
            ).values
            x[idx, self.AGENT_TIME_DEPARTURE.start + end_queue] = self.time + travel_time

            # Update the number of agents on the road
            is_agent = aggr_out != 0  # So agent who gets ID : 0 are not agent and can break the simulation
            x[is_agent, self.NUMBER_OF_AGENT] = start_counts[is_agent] + 1
        return x

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

