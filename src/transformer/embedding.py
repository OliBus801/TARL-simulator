import torch
import torch.nn as nn
from torch_geometric.utils import to_scipy_sparse_matrix, degree
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh


class EmbeddingMixer(nn.Module):
    """ 
    A class that takes the embedding of the agents, the positional encodings and the structural encodinds
    and mix them together to create a new embedding.
    """
    
    def __init__(self, edge_index: torch.Tensor, num_nodes: int, out_dim: int, nb_embeddings: int = 15):
        super().__init__()
        self.out_dim = out_dim
        self.nb_embeddings = nb_embeddings
        self.linear = nn.Linear(nb_embeddings, self.out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.compute_encodings(edge_index, num_nodes, 12)

        
        
    def compute_encodings(self, edge_index: torch.Tensor,num_nodes: int, positional_dim: int) -> torch.Tensor:
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

        self.linear_positional = nn.Linear(positional_dim, self.out_dim)
        self.linear_structural = nn.Linear(1, self.out_dim)
        nn.init.xavier_uniform_(self.linear_positional.weight)
        nn.init.zeros_(self.linear_positional.bias)
        nn.init.xavier_uniform_(self.linear_structural.weight)
        nn.init.zeros_(self.linear_structural.bias)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the embedding mixer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing the embeddings of traffic graph.

        Returns
        -------
        torch.Tensor
            Output tensor after mixing the embeddings.
        """
        x_emb = self.linear(x)
        x_pos = self.linear_positional(self.positional_embedding)
        x_struct = self.linear_structural(self.structural_embedding)
        return x_emb + x_pos + x_struct 