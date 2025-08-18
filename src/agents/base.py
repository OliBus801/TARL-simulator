import numpy as np
from lxml import etree
import torch
import os
from torch_geometric.utils import to_dense_adj, to_networkx
from torch_geometric.data import Data
from datetime import datetime
import networkx as nx
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
        for idx, inter in enumerate(sorted(intersections)):
            src_idx = num_links + 2 * idx
            intersection_indices[inter] = (src_idx, src_idx + 1)

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

        capacity_ok = (
            x[road_idx, h.NUMBER_OF_AGENT]
            < x[road_idx, h.MAX_NUMBER_OF_AGENT] - h.CONGESTION_FILE
        )
        if not torch.any(capacity_ok):
            return x

        agent_idx = torch.nonzero(ready, as_tuple=False).squeeze(1)[capacity_ok]
        road_idx = road_idx[capacity_ok]

        order = torch.argsort(road_idx)
        road_sorted = road_idx[order]
        agent_sorted = agent_idx[order]
        start_counts = x[road_sorted, h.NUMBER_OF_AGENT].to(torch.long)
        unique_roads, counts = torch.unique_consecutive(road_sorted, return_counts=True)
        offsets = torch.arange(road_sorted.size(0), device=x.device) - torch.repeat_interleave(
            torch.cat([torch.tensor([0], device=x.device), counts.cumsum(0)[:-1]], dim=0),
            counts,
        )
        positions = start_counts + offsets

        x[road_sorted, h.AGENT_POSITION.start + positions] = agent_sorted.to(torch.float)
        x[road_sorted, h.AGENT_TIME_ARRIVAL.start + positions] = float(self.time)
        x[road_sorted, h.AGENT_POSITION_AT_ARRIVAL.start + positions] = start_counts.to(x.dtype)

        x[unique_roads, h.NUMBER_OF_AGENT] += counts.to(x.dtype)
        self.agent_features[agent_sorted, self.ON_WAY] = 1.0

        graph.x = x
        return x


    def withdraw_agent_from_network(self, x: torch.Tensor, h: FeatureHelpers) -> None:
        """
        Withdraws all agents in the front of queue of every road in the traffic network. 

        Parameters
        ----------
        x : torch.Tensor
            A tensor which contains the node features of the network, and which is update 
            at the end of function.
        h : FeatureHelpers
            Index helpers to understand how columns works
        """
        
        # Mask agent that have to get out
        x_update = x.clone()
        candidate_agent = x_update[:, h.HEAD_FIFO].to(torch.int64)
        mask = ((self.agent_features[candidate_agent, self.DESTINATION] == x_update[:, h.ROAD_INDEX]) & 
                (x_update[:, h.NUMBER_OF_AGENT] != 0))
        # Withdraw these agents from the network
        x_update[mask, : h.MAX_NUMBER_OF_AGENT - 1] = x_update[mask, 1 : h.MAX_NUMBER_OF_AGENT]
        x_update[mask, h.NUMBER_OF_AGENT] -= 1
        self.agent_features[candidate_agent[mask], self.DONE] = 1
        self.agent_features[candidate_agent[mask], self.ON_WAY] = 0
        self.agent_features[candidate_agent[mask], self.ARRIVAL_TIME] = self.time
        mask = ((self.agent_features[candidate_agent, self.DESTINATION] == x_update[:, h.ROAD_INDEX]) &
                (x_update[:, h.NUMBER_OF_AGENT] != 0))
        if torch.any(mask):
            return self.withdraw_agent_from_network(x_update, h)
        else:
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
        num_roads = graph.num_roads

        # --- Sample for road nodes -------------------------------------------------
        adj_routes = to_dense_adj(
            graph.edge_index_routes, max_num_nodes=num_roads
        ).squeeze(0)
        out_routes = adj_routes.sum(dim=-1)
        mask_routes = out_routes > 0
        if mask_routes.any():
            adj_norm = adj_routes[mask_routes] / out_routes[mask_routes].unsqueeze(-1)
            index = torch.multinomial(adj_norm, num_samples=1).squeeze(1)
            road_idx = torch.arange(num_roads, device=node_feature.device)[mask_routes]
            node_feature[road_idx, h.SELECTED_ROAD] = index.to(torch.float)

        # --- Sample for SRC nodes --------------------------------------------------
        total_nodes = node_feature.size(0)
        adj_full = to_dense_adj(
            graph.edge_index, max_num_nodes=total_nodes
        ).squeeze(0)
        out_full = adj_full.sum(dim=-1)
        src_mask = (torch.arange(total_nodes, device=node_feature.device) >= num_roads) & (
            out_full > 0
        )
        if src_mask.any():
            adj_src = adj_full[src_mask]
            adj_src = adj_src / adj_src.sum(dim=-1, keepdim=True)
            index = torch.multinomial(adj_src, num_samples=1).squeeze(1)
            node_feature[src_mask, h.SELECTED_ROAD] = index.to(torch.float)

        updated_graph = Data(
            x=node_feature,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            edge_index_routes=graph.edge_index_routes,
            edge_attr_routes=graph.edge_attr_routes,
            num_roads=num_roads,
        )
        return updated_graph
    
    def reset(self):
        """
        Reset the agents to their initial state.
        """
        self.agent_features[:, self.ON_WAY] = 0.0
        self.agent_features[:, self.DONE] = 0.0
        

    def set_time(self, time):
        """
        Set the time 
        

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
        self.refresh_rate = 1000000 # Number of iterations before refreshing the Dijkstra computing


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
            # Compute travel time
            x_j = graph.x[graph.edge_index[0]]  # Source node
            critical_number = x_j[:, h.MAX_FLOW] * x_j[:, h.FREE_FLOW_TIME_TRAVEL] / 3600
            time_congestion = x_j[:, h.FREE_FLOW_TIME_TRAVEL] * (x_j[:, h.MAX_NUMBER_OF_AGENT] + 10 - critical_number) / (
                x_j[:, h.MAX_NUMBER_OF_AGENT] + 10 - x_j[:, h.HEAD_FIFO_CONG]
            )

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
        updated_graph = Data(x=x_update, edge_index=graph.edge_index, edge_attr=graph.edge_attr)
        self.count += 1
        return updated_graph
    





