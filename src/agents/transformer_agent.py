import torch
import torch.nn as nn
from typing import Optional
from torch._C import _functorch as fc
from torch_geometric.utils import to_scipy_sparse_matrix, degree
from torch_geometric.data import Data
from src.transformer.model import GraphTransformerNet
from src.transformer.embedding import EmbeddingMixer
from src.feature_helpers import FeatureHelpers

from src.agents.base import Agents
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh





class MLAgents(Agents, nn.Module):
    """
    A class that record agents, and lot of features mandatory for the traffic simulation.
    This class is used to predict the next road to take for each agent using a machine learning model.
    """

    def __init__(self, edge_index, num_nodes, device, edge_index_routes=None, num_roads=None):
        Agents.__init__(self, device = device)
        nn.Module.__init__(self)
        
        self.mixer = None
        self.transformer = GraphTransformerNet(
            node_dim_in=15,
            edge_dim_in=1,
            pe_in_dim=16,
            hidden_dim=16,
            gate=True,
            num_gt_layers=2,
            num_heads=4,
            dropout=0.1
        )
        self.value_head = torch.nn.Linear(16, 1)
        self.positional_embedding = None
        self.edge_index = edge_index

        # By default, compute positional encodings on the full graph.
        if edge_index_routes is None:
            edge_index_routes = edge_index
        if num_roads is None:
            num_roads = num_nodes

        # Compute encodings only on road nodes and pad zeros for SRC/DEST nodes
        self.compute_encodings(
            edge_index=edge_index_routes,
            num_nodes=num_roads,
            positional_dim=16,
            total_num_nodes=num_nodes,
        )

    def forward(self, node_features, edge_features, agent_index):
        """
        Forward pass of the model. It applies the transformer layer to the graph and returns the predicted next road.

        Parameters
        ----------
        graph : Data
            The input graph containing node features and edge indices.

        Returns
        -------
        h : torch.Tensor
            Output of the value fonction
        e : torch.Tensor
            Probabilities that an agent will take the road j when he is on the road i. The policy recap. 
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

            # Batch the graph nodes
            batch_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_node)

            # Batch the graph edge
            increment = torch.arange(batch_size, device=self.device).repeat_interleave(num_edge)
            increment = (increment*num_node).repeat(2,1)
            edge_index = self.edge_index.repeat(1, batch_size)
            edge_index = edge_index + increment

            # Additional stuff
            positional_embedding = self.positional_embedding.repeat(batch_size, 1)
            agent_feature = agent_feature.reshape(-1, agent_feature.size(-1))
            
        else:
            edge_index = self.edge_index
            node_features = node_features
            agent_feature = self.agent_features[agent_index]
            batch_idx = None
            positional_embedding = self.positional_embedding
            x = torch.cat((node_features, agent_feature), dim=-1)
        _, e = self.transformer(
            x=x,
            edge_index= edge_index,
            edge_attr = edge_features,
            pe = positional_embedding,
            batch = batch_idx
        )
        if isBatch:
            e = e.view(batch_size, num_edge)
        return e

    def save_network(self, file_path: str) -> None:
        """
        Saves the agents features into the file path.

        Parameters
        ----------
        file_path : str
            Path for saving the agent features
        """
        torch.save(self.state_dict(), file_path)

    def load_network(self, file_path: str) -> None:
        """
        Loads the agents features from the file path.

        Parameters
        ----------
        file_path : str
            Path for loading the agent features
        """        
        self.load_state_dict(torch.load(file_path, map_location=self.device))
        self.to(self.device)
        self.eval()

    def compute_encodings(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        positional_dim: int,
        total_num_nodes: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute the positional embedding for the input tensor and store it in the class.

        Parameters
        ----------
        x : Data
            A traffic network graph

        positional_size : int
            The number of eigenvalues to compute for the positional encoding.
        """
        A = to_scipy_sparse_matrix(edge_index, edge_attr=None, num_nodes=num_nodes)
        A = (A + A.T) / 2
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

        rwse = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float32).unsqueeze(1)

        # Pad embeddings with zeros for non-road nodes if needed
        if total_num_nodes is not None and total_num_nodes > num_nodes:
            pe = torch.zeros((total_num_nodes, positional_dim), dtype=eigvecs.dtype)
            pe[:num_nodes] = eigvecs
            rw = torch.zeros((total_num_nodes, 1), dtype=rwse.dtype)
            rw[:num_nodes] = rwse
            self.positional_embedding = pe
            self.structural_embedding = rw
        else:
            self.positional_embedding = eigvecs
            self.structural_embedding = rwse

    def choice(self, graph: Data, h: FeatureHelpers):
        """
        Chose the next direction to take for each agent using the transformer model.

        Parameters
        ----------
        graph : Data
            Graph of the trafic network
        h : FeatureHelpers
            Helpers for selecting index

        Returns
        ----------
        updated_graph : Data
            The updated graph with the selected roads for each agent.
        """
        # Get the output of the transformer model
        h_out, e_out = self(graph)

        # Sort the output
        group, indices = torch.sort(graph.edge_index[0], dim=1)
        e_out_sorted = e_out[indices]
        edge_index_sorted = graph.edge_index[:, indices]

        # Compute the cumulative probabilities
        cumsum = torch.cumsum(e_out_sorted, dim=1)
        first_indices = torch.zeros_like(group, dtype=torch.bool)
        first_indices[1:] = group[1:] != group[:-1]
        first_indices[0] = True
        last_indices = torch.zeros_like(group, dtype=torch.bool)
        last_indices[:-1] = group[1:] != group[:-1]
        last_indices[-1] = True

        # Make it cumulative probabilities
        sum_over_group = torch.zeros_like(group, dtype=torch.float32)
        sum_over_group[1:] = cumsum[last_indices][group[1:]]
        cumsum = cumsum - sum_over_group[group]

        
        # Randomly select an element in the group
        r = torch.rand_like(sum_over_group)
        r = r[group]
        r = cumsum > r

        # Update the graph with the selected roads
        selected_road = edge_index_sorted[1, r]
        road =  edge_index_sorted[0, r].to(torch.int64)
        
        x_updated = graph.x.clone()
        x_updated[road, h.SELECTED_ROAD] = selected_road # Reset the selected road
        updated_graph = Data(x=x_updated, edge_index=graph.edge_index, edge_attr=graph.edge_attr)
        return updated_graph



class ValueNet(MLAgents):
    def forward(self, node_features, edge_features, agent_index):
        """
        Forward pass of the model. It applies the transformer layer to the graph and returns the predicted next road.

        Parameters
        ----------
        graph : Data
            The input graph containing node features and edge indices.

        Returns
        -------
        h : torch.Tensor
            Output of the value fonction
        e : torch.Tensor
            Probabilities that an agent will take the road j when he is on the road i. The policy recap. 
        """
        
        if node_features.dim() == 3:
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

            # Batch the graph nodes
            batch_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_node)

            # Batch the graph edge
            increment = torch.arange(batch_size, device=self.device).repeat_interleave(num_edge)
            increment = (increment*num_node).repeat(2,1)
            edge_index = self.edge_index.repeat(1, batch_size)
            edge_index = edge_index + increment

            # Additional stuff
            positional_embedding = self.positional_embedding.repeat(batch_size, 1)
            agent_feature = agent_feature.reshape(-1, agent_feature.size(-1))
            
        else:
            node_features = node_features
            agent_feature = self.agent_features[agent_index]
            batch_idx = None
            x = torch.cat((node_features, agent_feature), dim=-1)
        h, _ = self.transformer(
            x=x,
            edge_index= edge_index,
            edge_attr = edge_features,
            pe = positional_embedding,
            batch = batch_idx
        )
        return h
    
