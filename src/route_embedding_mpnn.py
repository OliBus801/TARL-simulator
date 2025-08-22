import torch
from torch import nn


class RouteEmbeddingMPNN(nn.Module):
    """Simple message-passing neural network for route embeddings.

    This module performs mean aggregation of neighboring node features over a
    fixed number of hops. It can be used to embed routes based on the
    surrounding network structure.

    Parameters
    ----------
    in_channels : int
        Dimensionality of input node features.
    out_channels : int
        Dimensionality of the returned embeddings.
    num_hops : int
        Number of message-passing hops to perform.
    """

    def __init__(self, in_channels: int, out_channels: int, num_hops: int):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.num_hops = num_hops

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute route embeddings via mean-aggregation message passing.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape (num_nodes, in_channels).
        edge_index : torch.Tensor
            Edge indices of shape (2, num_edges) following the PyG convention.

        Returns
        -------
        torch.Tensor
            Embeddings of shape (num_nodes, out_channels).
        """
        h = self.lin(x)
        for _ in range(self.num_hops):
            h = self._mean_aggregate(h, edge_index)
        return h

    def _mean_aggregate(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Perform one hop of mean aggregation message passing."""
        row, col = edge_index
        out = torch.zeros_like(x)
        out.index_add_(0, row, x[col])
        deg = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        deg.index_add_(0, row, torch.ones_like(row, dtype=x.dtype))
        deg = deg.clamp(min=1).unsqueeze(-1)
        return out / deg
