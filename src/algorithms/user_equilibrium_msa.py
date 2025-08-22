"""Compute traffic assignment with the Method of Successive Averages (MSA).

This module exposes :func:`run_msa` which post–processes a simulation
snapshot to obtain a deterministic user–equilibrium of flows using the
MSA averaging rule.  The algorithm operates on the road network encoded
as a :class:`torch_geometric.data.Data` object and on the trip
information stored inside :class:`src.agents.base.Agents`.

The high level steps are:

1. Build the origin–destination (OD) demand matrix from the agents.
2. Iteratively assign the demand to shortest paths while updating
   link costs using the MSA averaging rule until convergence.
3. Return the converged per–road hourly flows in a Python ``dict``.

The implementation favours clarity over ultimate performance as it is
primarily intended for analysis and unit test sized networks.
"""

from __future__ import annotations

from typing import Dict

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from src.agents.base import Agents
from src.feature_helpers import FeatureHelpers


def _build_demand_matrix(agent: Agents, num_nodes: int) -> np.ndarray:
    """Build an OD demand matrix from agent features.

    Parameters
    ----------
    agent:
        Agents container holding the ``agent_features`` tensor.  Each
        row represents a trip.  Row ``0`` is a dummy entry used by the
        simulator and is ignored here.
    num_nodes:
        Total number of nodes in the network graph.

    Returns
    -------
    np.ndarray
        A ``num_nodes × num_nodes`` matrix where ``M[i, j]`` represents
        the number of trips originating from node ``i`` and terminating
        at node ``j``.
    """

    feats = agent.agent_features
    if feats is None or feats.size(0) <= 1:
        return np.zeros((num_nodes, num_nodes), dtype=float)

    origins = feats[1:, agent.ORIGIN].to(torch.int64)
    dests = feats[1:, agent.DESTINATION].to(torch.int64)
    flat = origins * num_nodes + dests
    counts = torch.bincount(flat, minlength=num_nodes * num_nodes)
    od = counts.view(num_nodes, num_nodes).to(torch.float64).cpu().numpy()
    return od


def run_msa(graph: Data, agents: Agents, tol: float = 1e-5, max_iter: int = 1000) -> Dict[int, float]:
    """Run the Method of Successive Averages on the given graph.

    Parameters
    ----------
    graph: Data
        Road network represented as a ``torch_geometric`` ``Data``
        object.  The node features must contain free–flow travel times
        and capacities as created by
        :func:`src.transportation_simulator.TransportationSimulator.config_network`.
    agents: Agents
        Agents container holding the list of trips from which the OD
        matrix is built.
    tol: float, optional
        Convergence tolerance on the L1 norm of successive flow
        estimates.  The default is ``1e-5``.
    max_iter: int, optional
        Maximum number of MSA iterations, by default ``1000``.

    Returns
    -------
    Dict[int, float]
        Dictionary mapping each road node index to its converged hourly
        flow.
    """

    # ------------------------------------------------------------------
    # Pre-processing and helpers
    # ------------------------------------------------------------------
    num_nodes = int(graph.x.size(0))
    num_features = int(graph.x.size(1))
    Nmax = (num_features - 7) // 3
    h = FeatureHelpers(Nmax=Nmax)

    num_roads = int(getattr(graph, "num_roads", num_nodes))

    free_flow = graph.x[:, h.FREE_FLOW_TIME_TRAVEL].detach().cpu().numpy()
    capacity = graph.x[:, h.MAX_FLOW].detach().cpu().numpy()

    demand = _build_demand_matrix(agents, num_nodes)

    # Boolean mask selecting the actual roads in the graph; SRC/DEST
    # nodes have an index of ``-1`` in the ``ROAD_INDEX`` feature.
    is_road = graph.x[:, h.ROAD_INDEX].detach().cpu().numpy() >= 0

    # Initialise flow and cost arrays
    flow = np.zeros(num_nodes, dtype=float)
    cost = np.zeros(num_nodes, dtype=float)
    cost[is_road] = free_flow[is_road]

    # Build the directed graph used for shortest path calculations
    edge_index = graph.edge_index.detach().cpu().numpy()
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    for u, v in edge_index.T:
        G.add_edge(int(u), int(v), weight=float(cost[v]))

    alpha = 0.15  # BPR parameters
    beta = 4.0

    for it in range(1, max_iter + 1):
        prev_flow = flow.copy()
        aux_flow = np.zeros(num_nodes, dtype=float)

        # ------------------------------------------------------------------
        # All-or-nothing assignment on current costs
        # ------------------------------------------------------------------
        od_pairs = np.argwhere(demand > 0)
        for o, d in od_pairs:
            vol = demand[o, d]
            if vol <= 0:
                continue
            try:
                path = nx.shortest_path(G, source=int(o), target=int(d), weight="weight")
            except nx.NetworkXNoPath:
                continue
            # Skip the origin node itself; add flow only on road nodes
            for node in path[1:]:
                if is_road[node]:
                    aux_flow[node] += vol

        # ------------------------------------------------------------------
        # MSA averaging of flows and update of costs
        # ------------------------------------------------------------------
        step = 1.0 / it
        flow += step * (aux_flow - flow)
        cost[is_road] = free_flow[is_road] * (
            1.0 + alpha * (flow[is_road] / np.maximum(capacity[is_road], 1e-8)) ** beta
        )

        # Update edge weights with the new node costs (cost of entering v)
        for u, v in G.edges():
            G[u][v]["weight"] = float(cost[v])

        # Convergence check
        gap = np.linalg.norm(flow - prev_flow, ord=1)
        if gap < tol:
            break

    # Prepare output dictionary – only return counts for actual roads
    return {int(i): float(flow[i]) for i in range(num_roads)}
