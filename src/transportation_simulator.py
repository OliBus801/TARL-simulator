from lxml import etree
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from src.feature_helpers import FeatureHelpers
from src.simulation_core_model import SimulationCoreModel
from src.agents.base import Agents
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend for saving plots
import matplotlib.pyplot as plt
import time
import tqdm
from collections import defaultdict
import numpy as np
import os
import networkx as nx

class TransportationSimulator:
    """
    Manage the traffic network with some toolkit functions

    Parameters
    ----------

    Attributes
    ----------
    graph : Data
        Graph representing the traffic network with pytorch geometric format
    time : int
        Time in second.
    timestep : int
        Timestep for the simulation
    """

    def __init__(self, device: str, torch_compile: bool = False):
        self.model_core = None
        self.agent = Agents(device)
        self.device = device
        self.torch_compile = torch_compile

        # The feature are the graph transportation network
        self.graph = None
        self.time = 0
        
        # Performance
        self.inserting_time = 0
        self.core_time = 0 
        self.withdraw_time = 0
        self.choice_time = 0

        # Configuration
        self.timestep = 1 #  Seconds
        self.node_metrics = False # Useful to compute node metrics

        # Record
        self.leg_histogram_values = []
        self.road_optimality_values = []
        self.wardrop_gap_values = []
        self.on_way_before = 0
        self.done_before = 0


    def config_network(self, file_path: str) -> None:
        """
        Configure the network from a configuration file and create the graph representing the
        network

        Parameters
        ----------
        file_path : str
            Relative or absolute path which contains the MATSim configuration
        ----------
        """
        # Check start time for time measurement
        start_time = time.time()

        # Determine the actual file to use (.xml.gz or .xml)
        gz_path = file_path + ".xml.gz"
        xml_path = file_path + ".xml"
        if os.path.exists(gz_path):
            actual_path = gz_path
        elif os.path.exists(xml_path):
            actual_path = xml_path
        else:
            raise FileNotFoundError(f"Neither {gz_path} nor {xml_path} exists.")

        # Print the file path
        print(f"Configuring network from file: {actual_path}")

        # Try to open and parse the file 
        try:
            tree = etree.parse(actual_path)
            root = tree.getroot()
        except OSError as e:
            print(f"Error reading the file: {e}")
            return

        # Extract the main information
        links = root.find("links")
        try:
            effective_cell_size = float(links.get("effectivecellsize"))
        except:
            effective_cell_size = 7.5
        h = FeatureHelpers(Nmax=0)

        # Pre-processing the links to create node features and gather intersections
        node_features = []
        intersections = set()
        outgoing = defaultdict(list)
        incoming = defaultdict(list)
        for i, link in enumerate(tqdm.tqdm(links, desc="Processing links")):
            attrs = link.attrib
            from_id = attrs["from"]
            to_id = attrs["to"]
            intersections.update([from_id, to_id])
            outgoing[from_id].append(i)
            incoming[to_id].append(i)

            feature = torch.zeros(h.ROAD_INDEX - h.MAX_NUMBER_OF_AGENT + 1, dtype=torch.float32)
            feature[h.ROAD_INDEX] = i
            feature[h.LENGHT_OF_ROAD] = float(attrs["length"])
            feature[h.MAX_FLOW] = float(attrs["capacity"])
            feature[h.FREE_FLOW_TIME_TRAVEL] = feature[h.LENGHT_OF_ROAD] / float(attrs["freespeed"])
            feature[h.MAX_NUMBER_OF_AGENT] = int(
                feature[h.LENGHT_OF_ROAD] * float(attrs["permlanes"]) / effective_cell_size
            ) + 1
            node_features.append(feature)

        # Update Nmax based on the maximum number of agents in the node features
        self.Nmax = int(max(f[h.MAX_NUMBER_OF_AGENT] for f in node_features).item() + 1)
        h = FeatureHelpers(Nmax=self.Nmax)
        self.h = h

        # Create the node features tensor including SRC/DEST nodes
        num_roads = len(node_features)
        num_inters = len(intersections)
        x = torch.zeros((num_roads + 2 * num_inters, int(3*self.Nmax + 7)), dtype=torch.float32)
        for i in tqdm.tqdm(range(num_roads), desc="Building node features"):
            x[i, h.MAX_NUMBER_OF_AGENT : h.ROAD_INDEX + 1] = node_features[i]

        neutral_feature = torch.zeros(int(3*self.Nmax + 7), dtype=torch.float32)
        neutral_feature[h.ROAD_INDEX] = -1
        intersection_indices = {}
        for idx, inter in enumerate(sorted(intersections)):
            src_idx = num_roads + 2 * idx
            dest_idx = src_idx + 1
            x[src_idx] = neutral_feature
            x[dest_idx] = neutral_feature
            intersection_indices[inter] = (src_idx, dest_idx)

        # Create the route->route edges
        route_from_list = []
        route_to_list = []
        route_edge_attr = []
        for j, link in enumerate(tqdm.tqdm(links, desc="Building route→route edges")):
            attrs = link.attrib
            upstream = j
            to_node = attrs["to"]
            total_flow = 0.0
            edge_indices_start = len(route_edge_attr)

            for downstream in outgoing[to_node]:
                route_from_list.append(upstream)
                route_to_list.append(downstream)
                cap = float(attrs["capacity"])
                route_edge_attr.append(cap)
                total_flow += cap

            for i in range(edge_indices_start, len(route_edge_attr)):
                route_edge_attr[i] /= total_flow if total_flow > 0 else 1.0

        edge_index_routes = torch.tensor([route_from_list, route_to_list], dtype=torch.long)
        edge_attr_routes = torch.tensor(route_edge_attr, dtype=torch.float32).view(-1, 1)

        # Build complete edge list including SRC/DEST nodes
        from_list = route_from_list.copy()
        to_list = route_to_list.copy()
        edge_attr = route_edge_attr.copy()

        # SRC(i) -> route edges
        for inter, (src_idx, _) in intersection_indices.items():
            for road in outgoing.get(inter, []):
                from_list.append(src_idx)
                to_list.append(road)
                edge_attr.append(0.0)

        # route -> DEST(j) edges
        for inter, (_, dest_idx) in intersection_indices.items():
            for road in incoming.get(inter, []):
                from_list.append(road)
                to_list.append(dest_idx)
                edge_attr.append(0.0)

        edge_index = torch.tensor([from_list, to_list], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32).view(-1, 1)

        # Compute the adjacency matrix
        num_nodes = x.size(0)
        adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
        adjacency_matrix[edge_index[0], edge_index[1]] = 1

        # Pre-compute normalized adjacency for SRC nodes -> roads
        src_rows_idx = torch.arange(num_roads, num_nodes, 2, dtype=torch.long)
        src_adj = adjacency_matrix[src_rows_idx, :num_roads].to(torch.float32)
        src_deg = src_adj.sum(dim=1, keepdim=True)
        src_adj = torch.where(src_deg > 0, src_adj / src_deg, torch.zeros_like(src_adj))

        # Pre-compute static congestion factors
        critical_number = x[:, h.MAX_FLOW] * x[:, h.FREE_FLOW_TIME_TRAVEL] / 3600
        congestion_constant = x[:, h.FREE_FLOW_TIME_TRAVEL] * (
            x[:, h.MAX_NUMBER_OF_AGENT] + 10 - critical_number
        )

        # Creating the PyG Graph object
        self.graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_index_routes=edge_index_routes,
            edge_attr_routes=edge_attr_routes,
            num_roads=num_roads,
            adj_matrix=adjacency_matrix,
            src_adj=src_adj,
            critical_number=critical_number,
            congestion_constant=congestion_constant,
        ).to(self.device)

        # Print the execution time
        end_time = time.time()
        print(f"🕒 | Network configured in {end_time - start_time:.2f} seconds")
    
    def save_network(self, file_path: str) -> None:
        """
        Save the current network graph to a pickle (pt) file.

        Parameters
        ----------
        file_path : str
            Path to save the graph data.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save({
            "graph": self.graph,
            "Nmax": self.Nmax,
        }, file_path)
        print(f"💾 | Network saved to {file_path}")

    def load_network(self, scenario: str) -> None:
        """
        Load the network graph from a picke (pt) file.

        Parameters
        ----------
        file_path : str
            Path to load the graph data from.
        """
        try:
            file_path = os.path.join("save", scenario, "network.pt")
            d = torch.load(file_path, weights_only=False)
            self.graph = d["graph"]
            self.Nmax = d["Nmax"]
            print(f"🚦 | Network loaded from {file_path}")
        except FileNotFoundError:
            print(f"📁 | Network save file {file_path} not found. Trying to load from XML...")
            file_path = os.path.join("data", scenario, "network")
            self.config_network(file_path)
            self.save_network(os.path.join("save", scenario, "network.pt"))

        self.h = FeatureHelpers(Nmax=self.Nmax)
        
    def configure_core(self):
        """
        Configure the simulation core for handle road simulation thanks to the


        Parameters
        ----------
        file_path : str
            Relative or absolute path which contains the MATSim configuration
        ----------
        """
        self.model_core = SimulationCoreModel(
            self.Nmax, self.device, self.time, torch_compile=self.torch_compile
        )
    
    def set_time(self, time):
        """
        Set the simulation time for the environment, agent, and model core.
        Args:
            time (float): The current simulation time.
        """
        self.time = time
        self.agent.set_time(time)
        self.model_core.set_time(time)

    def run(self):
        h = FeatureHelpers(Nmax=self.Nmax)

        
        # Insert agent in the network
        b = time.time()
        self.graph.x = self.agent.insert_agent_into_network(self.graph, h)
        e = time.time()
        self.inserting_time += e-b

        # Withdraw agents
        b = e
        self.graph.x = self.agent.withdraw_agent_from_network(
            self.graph, h
        )
        e = time.time()
        self.withdraw_time += e-b

        
        # Run the simulation
        #old_positions = self.graph.x[:, h.NUMBER_OF_AGENT]
        b = e
        self.graph = self.agent.choice(self.graph, h)
        e = time.time()
        self.choice_time += e-b

        # Run the core model
        b = e
        self.graph = self.model_core(self.graph)
        e = time.time()
        self.core_time += e-b
        """
        # This is broken and maybe not even useful.
        i = 0
        while torch.any(old_positions != self.graph.x[:, h.NUMBER_OF_AGENT]) and i < 5:
            # Compute the direction of the agents
            b = e
            self.graph = self.agent.choice(self.graph, h)
            e = time.time()
            self.choice_time += e-b

            # Run the core model
            b = e
            self.graph = self.model_core(self.graph)
            e = time.time()
            self.core_time += e-b
            i+=1 
        """
        self.set_time(self.time + self.timestep)

        value_on_way = torch.sum(self.agent.agent_features[:, self.agent.ON_WAY])
        value_done = torch.sum(self.agent.agent_features[:, self.agent.DONE])
        self.leg_histogram_values.append([value_on_way - self.on_way_before + value_done - self.done_before,
                                            value_done - self.done_before, value_on_way, self.time])
        self.on_way_before = value_on_way
        self.done_before = value_done

        self.road_optimality_values.append((self.time, self.model_core.direction_mpnn.road_optimality_data["delta_travel_time"].cpu()))

        gap = self.compute_wardrop_gap()
        self.wardrop_gap_values.append((self.time, gap))

    def compute_wardrop_gap(self) -> float:
        h = self.h if hasattr(self, 'h') else FeatureHelpers(Nmax=self.Nmax)
        if self.graph is None or self.agent is None:
            return 0.0

        agent_feats = getattr(self.agent, 'agent_features', None)
        if agent_feats is None:
            return 0.0

        on_way = agent_feats[:, self.agent.ON_WAY] > 0
        agent_ids = torch.nonzero(on_way, as_tuple=False).squeeze(1)
        if agent_ids.numel() == 0:
            return 0.0

        x = self.graph.x
        edge_index = self.graph.edge_index

        x_j = x[edge_index[0]]
        time_congestion = self.graph.congestion_constant[edge_index[1]] / (
            x_j[:, h.MAX_NUMBER_OF_AGENT] + 10 - x_j[:, h.NUMBER_OF_AGENT]
        )
        time_flow = torch.max(
            torch.stack((x_j[:, h.FREE_FLOW_TIME_TRAVEL], time_congestion)), dim=0
        ).values

        # Build edge cost matrix
        num_nodes = x.size(0)
        edge_cost = torch.full((num_nodes, num_nodes), float('inf'), device=x.device)
        edge_cost[edge_index[0], edge_index[1]] = time_flow

        # Build NetworkX graph for shortest paths
        nx_graph = self.graph.clone()
        nx_graph.edge_attr = time_flow
        nx_graph = to_networkx(nx_graph, edge_attrs=["edge_attr"], to_undirected=False)
        lengths = dict(nx.all_pairs_dijkstra_path_length(nx_graph, weight="edge_attr"))
        paths = dict(nx.all_pairs_dijkstra_path(nx_graph, weight="edge_attr"))

        length_tensor = torch.full((num_nodes, num_nodes), float('inf'), device=x.device)
        next_hop = torch.full((num_nodes, num_nodes), -1, dtype=torch.long, device=x.device)
        for src, dst_dict in lengths.items():
            for dst, cost in dst_dict.items():
                length_tensor[src, dst] = cost
        for src, dst_dict in paths.items():
            for dst, path in dst_dict.items():
                next_hop[src, dst] = path[1] if len(path) >= 2 else src

        # Determine current road for each agent
        num_roads = int(self.graph.num_roads)
        agent_road = torch.full((agent_feats.size(0),), -1, dtype=torch.long, device=x.device)
        road_agents = x[:num_roads, h.AGENT_POSITION]
        road_counts = x[:num_roads, h.NUMBER_OF_AGENT].to(torch.int64)
        for r in range(num_roads):
            n = int(road_counts[r])
            if n > 0:
                ids = road_agents[r, :n].to(torch.long)
                agent_road[ids] = r

        roads = agent_road[agent_ids]
        dests = agent_feats[agent_ids, self.agent.DESTINATION].to(torch.long)
        selected = x[roads, h.SELECTED_ROAD].to(torch.long)

        valid = (roads >= 0) & (selected >= 0)
        if not torch.any(valid):
            return 0.0

        roads = roads[valid]
        dests = dests[valid]
        selected = selected[valid]

        c_min = length_tensor[roads, dests]
        c_sel = edge_cost[roads, selected] + length_tensor[selected, dests]

        mask = torch.isfinite(c_min) & torch.isfinite(c_sel)
        if not torch.any(mask):
            return 0.0

        gap = torch.sum(c_sel[mask] - c_min[mask]).item()
        return gap

    def reset(self):
        h = FeatureHelpers(Nmax=self.Nmax)
        torch.zero_(self.graph.x[:, h.AGENT_POSITION])
        torch.zero_(self.graph.x[:, h.AGENT_TIME_DEPARTURE])
        torch.zero_(self.graph.x[:, h.AGENT_TIME_ARRIVAL])
        torch.zero_(self.graph.x[:, h.NUMBER_OF_AGENT])
        
    def state(self):
        h = FeatureHelpers(Nmax=self.Nmax)
        x = self.graph.x[:, h.MAX_NUMBER_OF_AGENT:]
        edge_index = self.graph.edge_index
        edge_attr = self.graph.edge_attr
        agent_index = (self.graph.x[:, h.HEAD_FIFO]).to(torch.int64)
        return x, edge_attr, edge_index, agent_index

    def config_parameters(self, timestep_size: float = 1, start_time: int = 0):
        """
        Configure the meta parameters of the simulation.  
        
        Parameters
        ----------
        timestep_size : int
            Time between two updates
        start_time : int
            Time at which the simulation starts (in seconds)
        ----------
        """
        self.timestep = timestep_size
        self.time = start_time

        # After configuring the parameters, we need to reconfigure the core model
        self.configure_core()


    def plot_leg_histogram(self, output_dir: str | None = "data/outputs"):

        if not self.leg_histogram_values:
            print("No data available for plotting.")
            return None
        
        # Make sure the values are on CPU and convert to numpy for plotting
        values = []
        for v in self.leg_histogram_values:
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            values.append([x.item() if isinstance(x, torch.Tensor) else x for x in v])

        on_way = []
        departure = []
        arrival = []
        time = []

        on, dep, arr, t = 0, 0, 0, values[0][3]
        n = 18 // self.timestep  # Number of timesteps to average over (30 minutes)

        for i in range(len(values)):
            if i % n == 0:
                on_way.append(on)
                departure.append(dep)
                arrival.append(arr)
                time.append(t // 60)  # minutes
                dep, arr = 0, 0
            dep += values[i][0]
            arr += values[i][1]
            t = values[i][3]
            on = values[i][2]
                

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Create a secondary axis for "Departure" and "Arrival"
        ax1.step(time, on_way, label='On Way', color='green')
        ax1.step(time, departure, label='Departure', color='red', linestyle='--', where="post")
        ax1.step(time, arrival, label='Arrival', color='blue', linestyle='-.', where="post")
        ax1.set_ylabel("Number of Agents", color='black')
        ax1.tick_params(axis='y', labelcolor='black')

        # Fix x-axis to have hours instead of minutes
        min_hour = min(time) // 60
        max_hour = max(time) // 60
        print(f"Min hour: {min_hour}, Max hour: {max_hour}")
        ax1.set_xticks([i * 60 for i in range(min_hour, max_hour + 1)])
        ax1.set_xticklabels([str(i) for i in range(min_hour, max_hour + 1)])
        ax1.set_xlabel("Hour of Day")

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        ax1.legend(lines1, labels1, loc='upper left')

        ax1.set_title("Leg Histogram Over Time")
        fig.tight_layout()

        if output_dir is not None:
            filename = "leg_histogram.png"
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, filename))
            print("Leg histogram saved as ", filename)

        return fig
        
    def plot_road_optimality(self, output_dir: str | None = "data/outputs", road_ids: list = []):
        """
        Plot the optimality of roads based on the difference between free-flow travel time
        and actual travel time.

        In the dual graph, nodes = roads and edges = turns. Raw values are computed per edge.
        Here, we aggregate edge values by their source node index (edge_index[0]) to measure
        how costly it is to leave a given road toward all its outgoing connections.

        Parameters
        ----------
        save_plot : bool
            If True, save the plot to a file instead of displaying it.
        road_ids : list
            List of road IDs to plot the optimality for.
        output_dir : str
            Directory to save the plot if save_plot is True.
        ----------
        """
        if not self.road_optimality_values:
            print("No road optimality data available for plotting.")
            return None
        
        # Number of nodes
        num_nodes = self.graph.num_roads
        origin_idx = self.graph.edge_index_routes[0]  # [E]

        # --- Vectorisation temporelle ---
        # Timestamps et matrice [T, E] des valeurs edge-wise
        times = torch.tensor([t for t, _ in self.road_optimality_values], dtype=torch.float32)
        v_mat = torch.stack([v for _, v in self.road_optimality_values], dim=0)  # [T, E]
        times = (times / 3600.0).to(v_mat.device)

        # Agrégation par nœud source en une seule passe: [T, N]
        agg = torch.zeros(v_mat.size(0), num_nodes, device=v_mat.device, dtype=v_mat.dtype)
        agg.scatter_add_(1, origin_idx.unsqueeze(0).expand(v_mat.size(0), -1), v_mat)

        # --- Plot (CPU) ---
        fig, ax = plt.subplots(figsize=(12, 6))
        t_np = times.detach().cpu().numpy()
        agg_np = agg.detach().cpu().numpy()

        if road_ids:
            for road_id in road_ids:
                ax.plot(t_np, agg_np[:, road_id], label=f"Node {road_id}")
        else:
            for road_id in range(agg_np.shape[1]):
                ax.plot(t_np, agg_np[:, road_id], label=f"Node {road_id}")

        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Delta Travel Time (s) — sum over outgoing edges")
        ax.set_title("Road Optimality (Aggregated by Source Node) Over Time")
        ax.legend()
        fig.tight_layout()

        if output_dir is not None:
            filename = "road_optimality.png"
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, filename))
            print("Road optimality plot saved as", filename)

        return fig

    def plot_wardrop_gap(self, output_dir: str | None = "data/outputs"):
        """
        Plot the Wardrop gap over time.

        Parameters
        ----------
        output_dir : str
            Directory to save the plot if not None.
        ----------
        """

        if not self.wardrop_gap_values:
            print("No Wardrop gap data available for plotting.")
            return None

        times = torch.tensor([t for t, _ in self.wardrop_gap_values], dtype=torch.float32)
        gaps = torch.tensor([g for _, g in self.wardrop_gap_values], dtype=torch.float32)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(times.div(3600).detach().cpu().numpy(), gaps.detach().cpu().numpy())
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Wardrop Gap")
        ax.set_title("Wardrop Gap Over Time")
        fig.tight_layout()

        if output_dir is not None:
            filename = "wardrop_gap.png"
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, filename))
            print("Wardrop gap plot saved as", filename)

        return fig

    def plot_computation_time(self, output_dir: str = "data/outputs"):
        """
        Plot the computation time for different phases of the simulation in a pie chart.

        Parameters
        ----------
        save_plot : bool
            If True, save the plot to a file instead of displaying it.
        output_dir : str
            Directory to save the output files. Default is 'data/outputs'.
        ----------
        """

        times = [self.inserting_time, self.choice_time, self.core_time, self.withdraw_time]
        # Replace NaN values with -1
        times = [t if not np.isnan(t) else -1 for t in times]
        labels = ['Inserting', 'Choice', 'Core', 'Withdraw']
        total = sum(times)

        if total == 0:
            print("No computation time data available for plotting.")
            return None

        # Custom formatter to include bold percentage and italic value in seconds
        def format_label(pct, allvals):
            absolute = pct / 100 * sum(allvals)
            return r"$\bf{{{:.1f}\%}}$" "\n" r"$\it{{{:.2f}\ s}}$".format(pct, absolute)

        plt.figure(figsize=(8, 8))
        wedges, texts, autotexts = plt.pie(
            times,
            labels=labels,
            autopct=lambda pct: format_label(pct, times),
            startangle=90,
            textprops=dict(color="black", fontsize=12)
        )
        plt.title(
            "Computation Time Distribution\nTotal Execution Time: {:.2f} s".format(total),
            fontsize=14
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        filename = "computation_time.png"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename))
        print("Computation time plot saved as", filename)
    
    def compute_node_metrics(self, output_dir: str | None = "data/outputs"):
        """
        Compute metrics for each node in the graph, such as average V/C ratio, 
        standard deviation of V/C ratio, and hourly traffic count.

        Writes a CSV file 'node_metrics.csv' with columns:
        node_id, avg_vc, std_vc, count_0h, count_1h, ..., count_Nh

        Returns
        -------
        node_metrics : dict
            A dictionary containing the computed metrics for each node.
            Keys are node IDs; values are dicts:
              - 'avg_vc': float
              - 'std_vc': float
              - 'hourly_counts': list of int (len = number_of_hours)
        """
        import pandas as pd
        # --- 1) Hourly traffic counts ---
        
        # Retrieve the collected data from the model core and agent withdrawals 
        update_history = getattr(self.model_core.response_mpnn, 'update_history', [])
        withdraw_history = getattr(self.agent, 'withdraw_history', [])
        combined_history = update_history + withdraw_history
        if not combined_history:
            print("No update history available for computing node metrics.")
            return {}
        
    
        
        # Retrieve all times and all masks into tensors
        times = torch.tensor([t for t, _ in combined_history], dtype=torch.long)      # (T,)
        mask_matrix = torch.stack([mask for _, mask in combined_history], dim=0)      # (T, N), dtype=torch.bool

        # Compute hours and hourly dimension
        hours = (times // 3600).clamp(min=0)    # (T,)
        max_hour = int(hours.max().item())
        num_hours = max_hour + 1

        # One-hot encoding of hours
        # -> shape (T, num_hours), dtype=torch.long
        hour_onehot = F.one_hot(hours, num_classes=num_hours).to(dtype=torch.long)

        # Convert mask to int for summation
        mask_int = mask_matrix.to(dtype=torch.long)                                  # (T, N)

        # Vectorized counting: (num_hours, T) @ (T, N) -> (num_hours, N)
        counts_per_hour = hour_onehot.T @ mask_int                                  # (H, N)

        # Transpose if you prefer (N, H)
        counts_per_node = counts_per_hour.T                                        # (N, H)
        num_nodes = counts_per_node.size(0)

        # --- 2) V/C ratio ---
        # Retrieve capacities (shape N) and convert to float

        h = FeatureHelpers(Nmax=self.Nmax)

        cap = self.graph.x[:num_nodes, h.MAX_FLOW]        # (N,)
        # Avoid division by zero by replacing 0 → NaN while keeping shape
        cap_safe = cap.clone()
        cap_safe[cap_safe == 0] = float('nan')

        # V/C ratio: broadcast cap_safe.unsqueeze(1) → (N, H)
        vc = counts_per_node.float() / cap_safe.unsqueeze(1)           # (N, H)

        # --- 3) Mean and standard deviation in one step ---
        # PyTorch handles NaN correctly with nanmean/nanstd
        avg_vc = torch.nanmean(vc, dim=1)  # (N,)
        std_vc = torch.std(vc, dim=1, unbiased=False)   # (N,)

        # --- 4) Export CSV & construction du dict ---
        # Conversion en numpy pour pandas
        counts_np = counts_per_node.cpu().numpy()
        avg_np    = avg_vc.cpu().numpy()
        std_np    = std_vc.cpu().numpy()

        # DataFrame : d’abord la matrice de counts, puis on ajoute node_id, avg_vc, std_vc
        df_counts = pd.DataFrame(
            counts_np,
            columns=[f'count_{h}h' for h in range(num_hours)]
        )
        df = df_counts.copy()
        df['node_id'] = range(num_nodes)
        df['avg_vc']  = avg_np
        df['std_vc']  = std_np

        # Réordonnage des colonnes
        cols = ['node_id', 'avg_vc', 'std_vc'] + [f'count_{h}h' for h in range(num_hours)]
        df = df[cols]
        # Sauvegarde du DataFrame en CSV
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            df.to_csv(os.path.join(output_dir, 'node_metrics.csv'), index=False)
            print(f"Wrote {os.path.join(output_dir, 'node_metrics.csv')}")

        # Reconstruction du dict de sortie
        node_metrics = {
            n: {
                'avg_vc': float(avg_np[n]),
                'std_vc': float(std_np[n]),
                'hourly_counts': counts_np[n].tolist(),
            }
            for n in range(num_nodes)
        }

        return node_metrics


    def get_info(self, road_id, h: FeatureHelpers):

        road = self.graph.x[road_id]
        time_arrival = road[h.HEAD_FIFO_ARRIVAL_TIME]
        time_departure = road[h.HEAD_FIFO_DEPARTURE_TIME]

        return (
            f"Route {road_id} : {road[h.NUMBER_OF_AGENT]} / {road[h.MAX_NUMBER_OF_AGENT]} \n"
            f"Queue: {road[h.AGENT_POSITION][:15]}\n"
            f"Prochain depart dans {time_departure - self.time} vers la route {road[h.SELECTED_ROAD]}\n"
            f"Heure actuel : {self.time}"
        )
    

    def save(self, path):
        torch.save(self.graph, "graph.pth")