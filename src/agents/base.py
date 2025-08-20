import numpy as np
from lxml import etree
import torch
import os
from torch_geometric.utils import to_dense_adj, to_networkx
from torch_geometric.data import Data
from datetime import datetime
import networkx as nx
from sklearn.neighbors import KDTree
from src.feature_helpers import AgentFeatureHelpers, FeatureHelpers


class Agents(AgentFeatureHelpers):
    """
    A class that record agents, and lot of features mandatory for the traffic simulation.

    Parameters
    ----------

    Attributes
    ----------
    agent_features : torch.Tensor
        Agent features.
    time : int
        Time in second.
    """

    def __init__(self, device):
        super().__init__()
        self.agent_features = None
        self.time = 0
        self.device = device
        self.withdraw_history: list[tuple[int, torch.BoolTensor]] = []



    def config_agents_from_xml(self, scenario: str, *, verbose: bool = True) -> None:
        """Parse MATSim population and network files to configure agents."""
        
        def extract_activities(plan_elem):
            acts = plan_elem.findall("act")
            if not acts:
                acts = plan_elem.findall("activity")
            return acts
        
        def extract_departure_time(act_elem):
            time_str = act_elem.get("end_time")
            if not time_str:
                return 0
            for fmt in ("%H:%M:%S", "%H:%M"):
                try:
                    t = datetime.strptime(time_str, fmt)
                    return t.hour * 3600 + t.minute * 60 + t.second
                except ValueError:
                    continue
            return 0
        
        def parse_person_attributes(person_elem) -> dict:
            attrs = dict(person_elem.attrib)
            attributes_elem = person_elem.find("attributes")
            if attributes_elem is not None:
                for attr in attributes_elem.findall("attribute"):
                    name = attr.get("name")
                    value = attr.text
                    if name and value:
                        attrs[name] = value
            attrs.setdefault("car_avail", attrs.get("carAvail", "always"))
            attrs.setdefault("sex", "m")
            attrs.setdefault("employed", "no")
            attrs.setdefault("age", "20")
            return attrs
        
        def get_actual_path(file_path: str) -> str:
            gz_path = file_path + ".xml.gz"
            xml_path = file_path + ".xml"
            if os.path.exists(gz_path):
                return gz_path
            if os.path.exists(xml_path):
                return xml_path
            raise FileNotFoundError(f"Neither {gz_path} nor {xml_path} exists.")
        
        agent_file_path = get_actual_path(os.path.join("data", scenario, "population"))
        network_file_path = get_actual_path(os.path.join("data", scenario, "network"))
        
        try:
            population = etree.parse(agent_file_path).getroot()
            network = etree.parse(network_file_path).getroot()
        except OSError as e:
            raise RuntimeError(f"Error reading the file: {e}")
        
        if population is None:
            raise ValueError("The XML file does not contain a 'population' element.")
        nodes = network.find("nodes")
        if nodes is None:
            raise ValueError("The XML file does not contain a 'nodes' element.")
        links = network.find("links")
        if links is None:
            raise ValueError("The XML file does not contain a 'links' element.")
        
        node_positions = {node.get("id"): (float(node.get("x")), float(node.get("y"))) for node in nodes}
        num_links = len(links)
        link_positions = np.zeros((num_links, 2))
        link_from_id = {}
        link_to_id = {}
        link_from_list = []
        link_to_list = []
        intersections = set()
        for i, link in enumerate(links):
            a, b = link.get("from"), link.get("to")
            link_positions[i, 0] = (node_positions[a][0] + node_positions[b][0]) / 2
            link_positions[i, 1] = (node_positions[a][1] + node_positions[b][1]) / 2
            lid = link.get("id")
            link_from_id[lid] = a
            link_to_id[lid] = b
            link_from_list.append(a)
            link_to_list.append(b)
            intersections.update([a, b])

        # Map intersections -> (SRC, DEST) indices as created in config_network
        intersection_indices = {}
        sorted_intersections = sorted(intersections)
        for idx, inter in enumerate(sorted_intersections):
            src_idx = num_links + 2 * idx
            intersection_indices[inter] = (src_idx, src_idx + 1)

        # Build a KDTree over intersection node coordinates for legacy scenarios
        intersection_coords = np.array([node_positions[inter] for inter in sorted_intersections])
        kdtree = KDTree(intersection_coords)

        # Creating dummy agent
        dummy_row = [0.0, 0.0, 25 * 3600, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0]
        rows = [dummy_row]
        trips_per_agent = []
        exclude = {"car_avail_not_always":0, "no_plan":0, "too_few_activities":0, "no_valid_trip":0}
        invalid_trip_coords = 0
        total_agents = 0
        selected_agents = 0
        
        for person in population:
            if isinstance(person, etree._Comment):
                continue
            total_agents += 1
            attrs = parse_person_attributes(person)
            car_avail = attrs.get("car_avail", attrs.get("carAvail", "")).lower()
            if car_avail != "always":
                exclude["car_avail_not_always"] += 1
                continue
            plan = person.find("plan")
            if plan is None:
                exclude["no_plan"] += 1
                continue
            acts = extract_activities(plan)
            if len(acts) < 2:
                exclude["too_few_activities"] += 1
                continue
            sex = 1 if attrs.get("sex", "m").lower() == "f" else 0
            employed = 1 if attrs.get("employed", "no").lower() == "yes" else 0
            age = float(attrs.get("age", 0))
            valid_trips = 0
            for i in range(len(acts) - 1):
                origin_node = acts[i].get("link")
                dest_node = acts[i + 1].get("link")
                
                # Legacy support: infer nearest intersection from coordinates if link is invalid
                if origin_node not in intersection_indices:
                    ox, oy = acts[i].get("x"), acts[i].get("y")
                    if ox is not None and oy is not None:
                        try:
                            idx = kdtree.query([[float(ox), float(oy)]], return_distance=False)[0][0]
                            origin_node = sorted_intersections[idx]
                        except Exception:
                            pass
                if dest_node not in intersection_indices:
                    dx, dy = acts[i + 1].get("x"), acts[i + 1].get("y")
                    if dx is not None and dy is not None:
                        try:
                            idx = kdtree.query([[float(dx), float(dy)]], return_distance=False)[0][0]
                            dest_node = sorted_intersections[idx]
                        except Exception:
                            pass
                try:
                    if origin_node in intersection_indices and dest_node in intersection_indices:
                        src_idx = intersection_indices[origin_node][0]
                        dest_idx = intersection_indices[dest_node][1]
                    else:
                        print(f"Could not create plan for person {person.get('id')}: Invalid trip : {origin_node} -> {dest_node}")
                        continue
                except (TypeError, ValueError, KeyError):
                    invalid_trip_coords += 1
                    continue

                dep = extract_departure_time(acts[i])
                rows.append([
                    float(src_idx),
                    float(dest_idx),
                    float(dep),
                    0.0,
                    age,
                    float(sex),
                    float(employed),
                    0.0,
                    0.0,
                ])
                valid_trips += 1
            if valid_trips>0:
                selected_agents +=1
                trips_per_agent.append(valid_trips)
            else:
                exclude["no_valid_trip"] +=1
        
        self.agent_features = torch.tensor(rows, dtype=torch.float32, device=self.device)
        total_trips = len(rows) - 1

        # Logging
        print("\n" + "="*10 + " 👥 Population Created" + "="*10)
        info = "ℹ️  | "
        print(f"{info} {selected_agents}/{total_agents} agents selected ({(100*selected_agents/total_agents if total_agents else 0):.2f}%)")
        print(f"{info} Total trips: {total_trips}")
        print(f"{info} Final agent_features shape: {tuple(self.agent_features.shape)}")
        if verbose:
            if trips_per_agent:
                trips_per_agent = np.array(trips_per_agent)
                print(f"{info} Trips per agent - min:{trips_per_agent.min()} max:{trips_per_agent.max()} mean:{trips_per_agent.mean():.2f} median:{np.median(trips_per_agent):.2f}")

            print(f"{info} Exclusion reasons: {exclude}, invalid_trip_coords={invalid_trip_coords}")
            dep_times = self.agent_features[1:, self.DEPARTURE_TIME].cpu().numpy()
            dep_times = dep_times[dep_times>0]
            if dep_times.size:
                # Conversion secondes -> heures (0-23)
                dep_hours = (dep_times // 3600).astype(int)

                # Comptage par heure
                counts = np.bincount(dep_hours, minlength=24)

                # Impression formatée
                print("📊 | Departure histogram (bins = 1h) (null counts ignored):")
                for h in range(24):
                    if counts[h] >= 1:
                        print(f"{h:02d}h : {counts[h]}")
            print(f"{info} First rows (origin, destination, dep_time):\n{self.agent_features[:3,[self.ORIGIN,self.DESTINATION,self.DEPARTURE_TIME]]}")
            print(f"{info} Network: {len(nodes)} nodes, {len(links)} links")

    def insert_agent_into_network(self, graph: Data, h: FeatureHelpers) -> torch.Tensor:
        """Insert all agents that are ready to depart into the traffic network."""

        x = graph.x
        ready = (
            (self.agent_features[:, self.DEPARTURE_TIME] <= self.time)
            & (self.agent_features[:, self.ON_WAY] == 0)
            & (self.agent_features[:, self.DONE] == 0)
        )
        if not torch.any(ready):
            return x

        graph = self.choice(graph, h)
        x = graph.x

        origins = self.agent_features[ready, self.ORIGIN].to(torch.long)
        road_idx = x[origins, h.SELECTED_ROAD].to(torch.long)

        # Compute remaining capacity on each selected road
        remaining_cap = (
            x[road_idx, h.MAX_NUMBER_OF_AGENT]
            - h.CONGESTION_FILE
            - x[road_idx, h.NUMBER_OF_AGENT]
        ).to(torch.long)
        capacity_ok = remaining_cap > 0
        if not torch.any(capacity_ok):
            return x

        agent_idx = torch.nonzero(ready, as_tuple=False).squeeze(1)[capacity_ok]
        road_idx = road_idx[capacity_ok]
        remaining_cap = remaining_cap[capacity_ok]

        order = torch.argsort(road_idx)
        road_sorted = road_idx[order]
        agent_sorted = agent_idx[order]
        cap_sorted = remaining_cap[order]

        unique_roads, counts = torch.unique_consecutive(road_sorted, return_counts=True)
        start_counts = x[road_sorted, h.NUMBER_OF_AGENT].to(torch.long)
        # Capacity available per unique road (same value repeated within group)
        cap_per_road = cap_sorted[torch.cumsum(counts, 0) - counts]
        allowed_counts = torch.minimum(counts, cap_per_road)

        # Mask to keep only the agents that can actually enter
        mask = torch.zeros_like(road_sorted, dtype=torch.bool)
        idx = 0
        for c, a in zip(counts.tolist(), allowed_counts.tolist()):
            mask[idx : idx + a] = True
            idx += c

        road_sorted = road_sorted[mask]
        agent_sorted = agent_sorted[mask]
        start_counts = start_counts[mask]

        valid = allowed_counts > 0
        unique_roads = unique_roads[valid]
        counts = allowed_counts[valid]

        if agent_sorted.numel() == 0:
            return x

        offsets = torch.arange(road_sorted.size(0), device=x.device) - torch.repeat_interleave(
            torch.cat([torch.tensor([0], device=x.device), counts.cumsum(0)[:-1]], dim=0),
            counts,
        )
        positions = start_counts + offsets

        x[road_sorted, h.AGENT_POSITION.start + positions] = agent_sorted.to(torch.float)
        x[road_sorted, h.AGENT_TIME_ARRIVAL.start + positions] = float(self.time)

        # Compute and store the time of departure for each inserted agent
        critical_number = x[road_sorted, h.MAX_FLOW] * x[road_sorted, h.FREE_FLOW_TIME_TRAVEL] / 3600
        time_congestion = x[road_sorted, h.FREE_FLOW_TIME_TRAVEL] * (
            x[road_sorted, h.MAX_NUMBER_OF_AGENT] + 10 - critical_number
        ) / (x[road_sorted, h.MAX_NUMBER_OF_AGENT] + 10 - start_counts.to(x.dtype))
        travel_time = torch.max(
            torch.stack((x[road_sorted, h.FREE_FLOW_TIME_TRAVEL], time_congestion)), dim=0
        ).values
        time_departure = float(self.time) + travel_time
        x[road_sorted, h.AGENT_TIME_DEPARTURE.start + positions] = time_departure

        x[unique_roads, h.NUMBER_OF_AGENT] += counts.to(x.dtype)
        self.agent_features[agent_sorted, self.ON_WAY] = 1.0

        graph.x = x
        return x


    def withdraw_agent_from_network(
        self, x: torch.Tensor, edge_index: torch.Tensor, h: FeatureHelpers
    ) -> torch.Tensor:
        """
        Withdraw all agents at the head of the queue that have reached their destination.

        Parameters
        ----------
        x : torch.Tensor
            Node feature tensor of the road network. This tensor is updated and
            returned.
        edge_index : torch.Tensor
            Graph connectivity of the network in COO format.
        h : FeatureHelpers
            Index helpers to understand how columns work.
        """

        x_update = x.clone()
        # Pre-compute adjacency matrix to check connectivity to destinations
        adj = to_dense_adj(edge_index, max_num_nodes=x.size(0)).squeeze(0).to(x.device)

        # Only keep a withdrawal mask for the road nodes; intersection are ignored
        num_roads = int((x[:, h.ROAD_INDEX] >= 0).sum().item())
        withdrawn_mask = torch.zeros(num_roads, dtype=torch.bool, device=x.device)


        while True:
            candidate_agent = x_update[:, h.HEAD_FIFO].to(torch.long)
            roads = x_update[:, h.ROAD_INDEX].to(torch.long)
            dest = self.agent_features[candidate_agent, self.DESTINATION].to(torch.long)

            # Check if an edge exists from each road to the candidate's destination
            connectivity = adj[roads, dest] > 0
            mask = (connectivity 
                    & (x_update[:, h.NUMBER_OF_AGENT] != 0)
                    & (x_update[:, h.HEAD_FIFO_DEPARTURE_TIME] <= self.time))

            if not torch.any(mask[:num_roads]):
                break

            withdrawn_mask |= mask[:num_roads]  # Update the mask for roads only
            agents_to_withdraw = candidate_agent[mask]

            # Withdraw these agents from the network
            x_update[mask, : h.MAX_NUMBER_OF_AGENT - 1] = x_update[mask, 1 : h.MAX_NUMBER_OF_AGENT]
            x_update[mask, h.NUMBER_OF_AGENT] -= 1

            # Update agent features for withdrawn agents
            self.agent_features[agents_to_withdraw, self.DONE] = 1
            self.agent_features[agents_to_withdraw, self.ON_WAY] = 0
            self.agent_features[agents_to_withdraw, self.ARRIVAL_TIME] = self.time

        self.withdraw_history.append((self.time, withdrawn_mask.clone()))
        return x_update


    def save(self, file_path: str) -> None:
        """
        Saves the agents features into the file path.

        Parameters
        ----------
        file_path : str
            Path for saving the agent features
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(self.agent_features, file_path)
        print(f"💾 | Population saved to {file_path}")
    
    def load(self, scenario: str) -> None:
        """
        Loads the agents features from the file path.

        Parameters
        ----------
        file_path : str
            Path for loading the agent features
        """
        try:
            file_path = os.path.join("save", scenario, "population.pt")
            obj = torch.load(file_path, weights_only=True, map_location=self.device)
            if isinstance(obj, torch.Tensor):
                self.agent_features = obj
            else:
                raise TypeError(f"Expected a Tensor for 'agent_features', got {type(obj)}. " + "Regenerate the file by saving only tensors.")
            print(f"👥 | Population loaded from {file_path}")

        except FileNotFoundError:
            print(f"📁 | Population save file {file_path} not found. Trying to load from XML...")
            self.config_agents_from_xml(scenario)
            self.save(os.path.join("save", scenario, "population.pt"))

        # We make sure that the agent ID: 0 will never join the network
        self.agent_features[0, self.DEPARTURE_TIME] = 48*3600

    def choice(self, graph: Data, h: FeatureHelpers):
        """
        Chose the next direction to take for each agent.

        This samples a downstream road for every node that has outgoing
        connections. Roads without outgoing connections (e.g. those leading
        directly to a destination) and DEST nodes are ignored.

        Parameters
        ----------
        """
        node_feature = graph.x.clone()
        total_nodes = node_feature.size(0)
        num_roads = graph.num_roads

        # --- ROUTES -> ROUTES ---
        # Adjacency only between roads
        road_adj = to_dense_adj(
            graph.edge_index_routes, max_num_nodes=num_roads
        ).squeeze(0)
        road_mask = road_adj.sum(dim=-1) > 0
        road_probs = road_adj[road_mask]
        road_probs = road_probs / road_probs.sum(dim=-1, keepdim=True)

        # ---- SRC nodes ----
        edge_index = graph.edge_index
        src_edge_mask = edge_index[0] >= num_roads
        src_edges = edge_index[:, src_edge_mask]

        num_src = (total_nodes - num_roads + 1) // 2
        src_adj = torch.zeros((num_src, num_roads), device=node_feature.device)
        src_rows = (src_edges[0] - num_roads) // 2
        src_cols = src_edges[1]
        src_adj[src_rows, src_cols] = 1
        src_mask = src_adj.sum(dim=-1) > 0
        src_probs = src_adj[src_mask]
        src_probs = src_probs / src_probs.sum(dim=-1, keepdim=True)


        # Sample next roads
        probs = torch.cat([road_probs, src_probs], dim=0)
        if probs.numel() > 0:
            sampled = torch.multinomial(probs, num_samples=1).squeeze(1)
            num_roads_sel = road_probs.size(0)
            road_ids = torch.arange(num_roads, device=node_feature.device)[road_mask]
            node_feature[road_ids, h.SELECTED_ROAD] = sampled[:num_roads_sel].to(
                torch.float
            )
            src_ids = torch.arange(num_src, device=node_feature.device)[src_mask]
            node_feature[num_roads + 2 * src_ids, h.SELECTED_ROAD] = sampled[
                num_roads_sel:
            ].to(torch.float)

        updated_graph = Data(
            x=node_feature,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            edge_index_routes=graph.edge_index_routes,
            edge_attr_routes=graph.edge_attr_routes,
            num_roads=graph.num_roads,
        )
        return updated_graph
    
    def reset(self):
        """
        Reset the agents to their initial state.
        """
        self.agent_features[:, self.ON_WAY] = 0.0
        self.agent_features[:, self.DONE] = 0.0
        self.withdraw_history = []

        

    def set_time(self, time):
        """
        Set the time for the agents.

        Parameters
        ----------
        time : int
            Time value to update
        """
        self.time = time
        

class DijkstraAgents(Agents):

    def __init__(self, device):
        super().__init__(device)
        self.count = 0
        self.refresh_rate = 10 # Number of iterations before refreshing the Dijkstra computing


    def choice(self, graph: Data, h: FeatureHelpers):
        """
        Choose the next direction to take for each agent using Dijkstra.

        Parameters
        ----------
        graph : Data
            Graph of the traffic network
        h : FeatureHelpers
            Helpers for selecting index
        """

        if self.count % self.refresh_rate == 0:
            # Compute travel time on the routes
            x_j = graph.x[graph.edge_index[0]]  # Source node
            critical_number = x_j[:, h.MAX_FLOW] * x_j[:, h.FREE_FLOW_TIME_TRAVEL] / 3600
            time_congestion = x_j[:, h.FREE_FLOW_TIME_TRAVEL] * (
                x_j[:, h.MAX_NUMBER_OF_AGENT] + 10 - critical_number
            ) / (x_j[:, h.MAX_NUMBER_OF_AGENT] + 10 - x_j[:, h.NUMBER_OF_AGENT])

            # Travel time = max(free-flow, congested)
            time_flow = torch.max(
                torch.stack((x_j[:, h.FREE_FLOW_TIME_TRAVEL], time_congestion)), dim=0
            ).values

            # Conversion en networkx
            nx_graph = graph.clone()
            nx_graph.edge_attr = time_flow
            nx_graph = to_networkx(nx_graph, edge_attrs=["edge_attr"], to_undirected=False)

            # Calcul de tous les chemins minimaux
            paths = dict(nx.all_pairs_dijkstra_path(nx_graph, weight="edge_attr"))

            # Préparation d'un tableau tensorisé des prochaines étapes
            num_nodes = graph.x.size(0)
            next_hop_tensor = torch.full((num_nodes, num_nodes), -1, dtype=torch.int64)  # -1 = chemin non défini

            for src, dst_dict in paths.items():
                for dst, path in dst_dict.items():
                    if len(path) >= 2:
                        next_hop_tensor[src, dst] = path[1]
                    elif len(path) == 1:
                        next_hop_tensor[src, dst] = src  # déjà sur place

            self.next_hop_tensor = next_hop_tensor.to(graph.x.device)

        # Get agents origin and destination
        agents = graph.x[:, h.HEAD_FIFO].to(torch.int64)
        destinations = self.agent_features[agents, self.DESTINATION].to(torch.int64)

        # Met à jour les routes sélectionnées
        x_update = graph.x.clone()
        idx = torch.arange(agents.size(0), device=graph.x.device)
        x_update[:, h.SELECTED_ROAD] = self.next_hop_tensor[idx, destinations]

        # Mise à jour du graphe
        updated_graph = Data(
            x=x_update,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            edge_index_routes=graph.edge_index_routes,
            edge_attr_routes=graph.edge_attr_routes,
            num_roads=graph.num_roads
        )
        self.count += 1
        return updated_graph

