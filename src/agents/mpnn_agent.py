from src.agents.base import Agents
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_scipy_sparse_matrix, degree, to_networkx
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
import torch
import torch.nn as nn
from torch._C import functorch as fc
from src.feature_helpers import ObservationFeatureHelpers
from torch_geometric.data import Data
import networkx as nx
import numpy as np



class MPNNPolicyNet(MessagePassing, Agents):

    h = ObservationFeatureHelpers()

    def __init__(self, edge_index, num_nodes, free_flow_time_travel, device):
        Agents.__init__(self, device = device)
        MessagePassing.__init__(self, aggr='mean', flow='target_to_source')
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.num_edges = edge_index.size(1)
        self.dim_node_features = 16
        self.dim_edge_features = 1
        self.refresh_dijkstra(edge_index, free_flow_time_travel)
        self.nodes_embedding = nn.Embedding(num_nodes, 1)
        self.edge_mlp_test = nn.Sequential(
            nn.Linear(self.dim_node_features + self.dim_node_features, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * self.dim_node_features + self.dim_edge_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        for module in self.edge_mlp:
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, -0.1, 0.1)
                nn.init.constant_(module.bias, 0) 

        for module in self.edge_mlp_test:
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, -0.1, 0.1)
                nn.init.constant_(module.bias, 0) 
        

    def refresh_dijkstra(self, edge_index: torch.Tensor, free_flow_travel: torch.Tensor):
        """
        Refresh the Dijkstra agent with new edge index and node features.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge index tensor representing the graph structure.

        nodes_features : torch.Tensor
            Node features tensor.
        """
        assert free_flow_travel.size(0) == edge_index.size(1), "Free flow travel time must match the number of edges."
        assert edge_index.size(0) == 2, "Edge index must be a 2D tensor with shape [2, num_edges]."

        # Create a nx graph from the edge index
        nx_graph = Data(edge_index=edge_index)
        nx_graph.edge_attr = free_flow_travel
        nx_graph = to_networkx(nx_graph, edge_attrs=["edge_attr"], to_undirected=False)

        # Compute the Dijkstra matrix
        lengths = dict(nx.shortest_path_length(nx_graph, source=None, target=None, weight="edge_attr"))
    
        # Convert into a matrix
        dist_matrix = np.full((self.num_nodes, self.num_nodes), np.inf)
        np.fill_diagonal(dist_matrix, 0)
        for source, target_lengths in lengths.items():
            for target, length in target_lengths.items():
                dist_matrix[source, target] = length
        self.dist_matrix = torch.tensor(dist_matrix, dtype=torch.float32, device=self.device)

    def compute_dijkstra_logits(self,
                                agent_destination: torch.Tensor,
                                time_travel: torch.Tensor) -> torch.Tensor:
        """
        Compute the logits for the Dijkstra agent based on the node features, edge features, and agent index.
        Parameters
        ----------  
        node_features : torch.Tensor
            Input tensor containing the embeddings of traffic graph.
        edge_features : torch.Tensor
            Edge features tensor.   
        agent_index : torch.Tensor
            Index of the agent in the graph.    
        Returns
        -------
        torch.Tensor
            Logits tensor reflecting the probabilities of activation for each edge.
        """
        assert self.dist_matrix is not None, "The Dijkstra matrix has not been computed yet."
        source_index = self.edge_index[1] # Source nodes of the edges
        repeat_size = agent_destination.size(0) // source_index.size(0)
        assert agent_destination.size(0) % agent_destination.size(0) == 0, "Error with the batching."
        source_index = source_index.repeat(repeat_size)
        dist = self.dist_matrix[source_index, agent_destination]
        logits = - dist - time_travel # Negative distance for logits
        if repeat_size > 1:
            logits = logits.view(repeat_size, -1)
        else:
            logits = logits.view(-1)
        return logits

    

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor, agent_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MPNNPolicyNet.

        Parameters
        ----------
        node_features : torch.Tensor
            Input tensor containing the embeddings of traffic graph.

        edge_features : torch.Tensor
            Edge features tensor.

        agent_index : torch.Tensor
            Index of the agent in the graph.

        Returns
        -------
        torch.Tensor
            Logits tensor reflecting the probabilities of activation for each edge.
        """
        isBatch = False
        if node_features.dim() == 3:
            isBatch = True
            batch_size = node_features.size(0)
            num_node = node_features.size(1)
            num_edge = self.edge_index.size(1)

            # Removes the Batched Tensor 
            try:
                node_features = fc.get_unwrapped(node_features)[0]
                edge_features = fc.get_unwrapped(edge_features)[0]
                agent_index = fc.get_unwrapped(agent_index)[0]
            except:
                pass

            # Batch the node features and the edge features
            node_features = node_features.reshape(-1, node_features.size(-1))
            edge_features = edge_features.reshape(-1, edge_features.size(-1))

            # Fetches the features of the agent 
            agent_feature = self.agent_features[agent_index]
            agent_feature = agent_feature.reshape(-1, agent_feature.size(-1))
            x = torch.cat((node_features, agent_feature), dim=-1)

            # Batch the graph edge
            increment = torch.arange(batch_size, device=self.device).repeat_interleave(num_edge)
            increment = (increment*num_node).repeat(2,1)
            edge_index = self.edge_index.repeat(1, batch_size)
            edge_index = edge_index + increment
            
        else:
            edge_index = self.edge_index
            node_features = node_features
            agent_feature = self.agent_features[agent_index]
            x = torch.cat((node_features, agent_feature), dim=-1)
        
        # Update the edges of the graph using message passing
        logits = self.update_edges(x, edge_index, edge_attr=edge_features)
        if isBatch:
            logits = logits.view(batch_size, self.num_edges)
        else:
            logits = logits.view(self.num_edges)

        # Compute the logits for the Dijkstra agent
        x_j = node_features[edge_index[1]]
        critical_number = x_j[:,  self.h.MAX_FLOW] * x_j[:,  self.h.FREE_FLOW_TIME_TRAVEL] /3600
        time_congestion = x_j[:,  self.h.FREE_FLOW_TIME_TRAVEL] * (x_j[:,  self.h.MAX_NUMBER_OF_AGENT]+10 - critical_number) /(x_j[:,  self.h.MAX_NUMBER_OF_AGENT]+10 - x_j[:,  self.h.NUMBER_OF_AGENT])
        time_travel = torch.max(torch.stack((x_j[:,  self.h.FREE_FLOW_TIME_TRAVEL], time_congestion)), dim=0).values

        agent_destination = x[:, self.h.DESTINATION].to(torch.long)
        agent_destination = agent_destination[edge_index[0]]
        logits_dijkstra = torch.zeros_like(logits)   #self.compute_dijkstra_logits(agent_destination, time_travel)
        logits_dijkstra[0] = 0
        norm = torch.norm(logits + logits_dijkstra, p=2, dim=-1, keepdim=True)
        #return torch.tanh(logits)
        return logits  # Normalize the logits to avoid division by zero

    
    def update_edges(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        """
        Update the edges of the graph using message passing.

        Parameters²
        ----------
        x : torch.Tensor
            Input tensor containing the embeddings of traffic graph.

        edge_index : torch.Tensor
            Edge index tensor representing the graph structure.

        edge_attr : torch.Tensor, optional
            Edge attributes tensor (default is None).

        Returns
        -------
        torch.Tensor
            Updated edge attributes after message passing.
        """
        road_index = x[:, ObservationFeatureHelpers().ROAD_INDEX].to(torch.long)
        embeddings = self.nodes_embedding(road_index)
        return embeddings[edge_index[1]]


        # road_index = x[:, ObservationFeatureHelpers().ROAD_INDEX].to(torch.long)
        # embeddings = self.nodes_embedding(road_index)
        # x_i = embeddings[edge_index[0]]
        # x_j = embeddings[edge_index[1]]
        # e_ij = torch.cat([x_i, x_j], dim=1)
        # return self.edge_mlp_test(e_ij)

        # row, col = edge_index
        # x_i = x[edge_index[0]]
        # x_j = x[edge_index[1]]
        # e_ij = torch.cat([x_i, x_j, edge_attr], dim=1)
        # return self.edge_mlp(e_ij)

    def compute_encodings(self, edge_index: torch.Tensor, num_nodes: int, positional_dim: int) -> torch.Tensor:
        """
        Compute the positional embedding for the input tensor and store it in the class.

        Parameters
        ----------
        edge_index : Data
            A traffic network graph

        positional_size : int
            The number of eigenvalues to compute for the positional encoding.
        """
        A = to_scipy_sparse_matrix(edge_index, edge_attr=None, num_nodes=num_nodes)
        A = (A+A.T)/2
        L = csgraph.laplacian(A, normed=True)
        eigvals, eigvecs = eigsh(L, k=positional_dim + 5, which='SM')

        # Remove the trivial eigenvalue and eigenvector
        tol = 1e-5 
        nontrivial_mask = eigvals > tol
        eigvals = eigvals[nontrivial_mask]
        eigvecs = eigvecs[:, nontrivial_mask]
        eigvals = eigvals[:positional_dim]
        eigvecs = eigvecs[:, :positional_dim]

        # Normalize the eigenvectors
        eigvecs = torch.Tensor(eigvecs)
        eigvecs = eigvecs / torch.norm(eigvecs, dim=0, keepdim=True)

        # Save the positional embedding
        self.positional_embedding = eigvecs
        self.structural_embedding = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float32).unsqueeze(1)


class MPNNValueNet(MessagePassing, Agents):

    def __init__(self, edge_index, num_nodes, device):
        Agents.__init__(self, device = device)
        MessagePassing.__init__(self, aggr='mean', flow='target_to_source', )
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.num_edges = edge_index.size(1)
        self.dim_nodes_features = 16
        self.dim_edges_features = 1
        self.message_mlp = nn.Sequential(
            nn.Dropout(0.05),
            nn.Linear(self.dim_nodes_features + self.dim_edges_features, 1),
            nn.Tanh(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(1, 1),
            nn.Tanh()
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(self.num_nodes+1, 1)
        )

        self.time_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Dropout(0.05),    
            nn.ReLU(),           
            nn.Linear(32, 32),
            nn.Dropout(0.05),
            nn.ReLU(),
            nn.Linear(32, 1)  
        )

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor, agent_index: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MPNNValueNet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing the embeddings of traffic graph.

        edge_index : torch.Tensor
            Edge index tensor representing the graph structure.

        edge_attr : torch.Tensor, optional
            Edge attributes tensor (default is None).

        Returns
        -------
        torch.Tensor
            Value tensor reflecting the estimated value of the state.
        """
        isBatch = False
        if node_features.dim() == 3:
            isBatch = True
            batch_size = node_features.size(0)
            num_node = node_features.size(1)
            num_edge = self.edge_index.size(1)

            # Batch the node features and the edge features
            node_features = node_features.reshape(-1, node_features.size(-1))
            edge_features = edge_features.reshape(-1, edge_features.size(-1))

            # Fetches the features of the agent 
            agent_feature = self.agent_features[agent_index]
            agent_feature = agent_feature.reshape(-1, agent_feature.size(-1))
            x = torch.cat((node_features, agent_feature), dim=-1)

            # Batch the graph edge
            increment = torch.arange(batch_size, device=self.device).repeat_interleave(num_edge)
            increment = (increment*num_node).repeat(2,1)
            edge_index = self.edge_index.repeat(1, batch_size)
            edge_index = edge_index + increment

            # Additional stuff
            agent_feature = agent_feature.reshape(-1, agent_feature.size(-1))
            
        else:
            edge_index = self.edge_index
            node_features = node_features
            agent_feature = self.agent_features[agent_index]
            x = torch.cat((node_features, agent_feature), dim=-1)
        
        # Update the edges of the graph using message passing
        v = self.propagate(edge_index, x=x, edge_attr=edge_features)
        if isBatch:
            v = v.view(batch_size, self.num_nodes)
        else:
            v = v.view(self.num_nodes)

        time_emb = self.time_net(time)
        v = torch.cat((v, time_emb), dim=-1)
        # Use all information to compute the final value
        return self.final_mlp(v)

    
    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:

        """
        Update the edges of the graph using message passing.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing the embeddings of traffic graph.

        edge_index : torch.Tensor
            Edge index tensor representing the graph structure.

        edge_attr : torch.Tensor, optional
            Edge attributes tensor (default is None).

        Returns
        -------
        torch.Tensor
            Updated edge attributes after message passing.
        """
        message = torch.cat([x_j, edge_attr], dim=1)
        return self.message_mlp(message)
    
    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """
        Update the node features after message passing.

        Parameters
        ----------
        aggr_out : torch.Tensor
            Aggregated output from the message passing.

        Returns
        -------
        torch.Tensor
            Updated node features.
        """
        return self.node_mlp(aggr_out)
    



class MPNNValueNetSimple(MessagePassing, Agents):
    """
    A simple MPNN agent that uses message passing to compute the value of the state.
    """

    def __init__(self, edge_index, num_nodes, device):
        Agents.__init__(self, device = device)
        MessagePassing.__init__(self, aggr='mean', flow='target_to_source')
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.num_edges = edge_index.size(1)
        self.dim_nodes_features = 16
        self.dim_edges_features = 1
        self.final_mlp = nn.Sequential(
            nn.Linear(self.num_nodes+1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor, agent_index: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MPNNAgentSimple.

        Parameters
        ----------
        node_features : torch.Tensor
            Input tensor containing the embeddings of traffic graph.

        edge_features : torch.Tensor
            Edge features tensor.

        agent_index : torch.Tensor
            Index of the agent in the graph.

        Returns
        -------
        torch.Tensor
            Value tensor reflecting the estimated value of the state.
        """
        x_1 = node_features[..., 1]
        x = torch.cat((x_1, time), dim=-1)
        return self.final_mlp(x)
