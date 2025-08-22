import torch
import torch.nn as nn
from torch.distributions import Categorical


class ContextualCostHead(nn.Module):
    """MLP head estimating contextual costs for outgoing edges.

    Parameters
    ----------
    input_dim : int
        Dimension of concatenated features :math:`[h_v \oplus x_a \oplus x_{agent}]`.
    hidden_dim : int, optional
        Hidden dimension of intermediate layers.
    num_layers : int, optional
        Number of linear layers (2 or 3).
    dropout : float, optional
        Dropout probability applied after hidden layers.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        assert num_layers in (2, 3), "num_layers must be 2 or 3"
        layers = []
        last_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute cost predictions for each outgoing edge.

        Parameters
        ----------
        features : torch.Tensor
            Concatenated edge, action and agent features.

        Returns
        -------
        torch.Tensor
            Predicted cost :math:`\hat c` for each edge.
        """
        return self.mlp(features).squeeze(-1)

    @torch.no_grad()
    def sample_action(self, costs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Sample an action using softmax over negative costs.

        Parameters
        ----------
        costs : torch.Tensor
            Cost predictions for outgoing edges.
        mask : torch.Tensor
            Boolean mask indicating valid edges (True means valid).

        Returns
        -------
        torch.Tensor
            Index of the sampled edge.
        """
        logits = -costs.clone()
        logits[~mask] = float('-inf')
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        return dist.sample()
