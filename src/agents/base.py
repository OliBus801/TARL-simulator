import numpy as np
from lxml import etree
import torch
import os
from torch_geometric.utils import to_dense_adj, to_networkx
from torch_geometric.data import Data
from sklearn.neighbors import KDTree
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



    def config_agents_from_xml(self, scenario: str) -> None:
        """
        Parse and configurate agents thanks to a MATSim XML file.

        Parameters
        ----------
        agent_file_path : str
            A string that contains the path for the agent XML description file from MATSim
        node_path : str
            A string that contains the path for the traffic network XML description file from
            MATSim
        """

        def extract_activities(plan_elem):
            # The name of the 'Activity' element is not the same for all scenarios
            acts = plan_elem.findall("act")
            if not acts:
                acts = plan_elem.findall("activity")
            return acts

        def extract_departure_time(act_elem):
            # Extract the departure time from end_time if it exists, else returns None
            time_str = act_elem.get("end_time")
            if not time_str:
                return None
            for fmt in ("%H:%M:%S", "%H:%M"):
                try:
                    t = datetime.strptime(time_str, fmt)
                    return t.hour * 3600 + t.minute * 60 + t.second
                except ValueError:
                    continue
            return None
        
        def parse_person_attributes(person_elem) -> dict:
            """
            Parse a 'person' XML element to retrieve their attributes.
            Should be robust to different types of XML structure of MATSim

            Parameters
            ----------
            person_elem : xml.etree.ElementTree.Element
                The XML element of the person to parse
            
            Returns
            -------
            dict
                A dictionnary [str, str] containing all attributes of the agents
            """
            attrs = dict(person_elem.attrib)  # direct attributes
            attributes_elem = person_elem.find("attributes")
            if attributes_elem is not None:
                for attr in attributes_elem.findall("attribute"):
                    name = attr.get("name")
                    value = attr.text
                    if name and value:
                        attrs[name] = value
            if "carAvail" not in attrs: attrs["car_avail"] = "always"  # Default value if not present
            if "sex" not in attrs: attrs["sex"] = "m"  # Default value if not present
            if "employed" not in attrs: attrs["employed"] = "no"  # Default value if not present
            if "age" not in attrs: attrs["age"] = "20"  # Default value if not present
            return attrs

        def get_actual_path(file_path: str) -> str:
            gz_path = file_path + ".xml.gz"
            xml_path = file_path + ".xml"
            if os.path.exists(gz_path):
                return gz_path
            elif os.path.exists(xml_path):
                return xml_path
            else:
                raise FileNotFoundError(f"Neither {gz_path} nor {xml_path} exists.")

        agent_file_path = get_actual_path(os.path.join("data", scenario, "population"))
        network_file_path = get_actual_path(os.path.join("data", scenario, "network"))

        # Try to open the two files and see if these contains information
        try:
            tree = etree.parse(agent_file_path)
            population = tree.getroot()
            tree = etree.parse(network_file_path)
            network = tree.getroot()
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
        
        # Build node positions
        node_positions = {
            node.get("id"): (float(node.get("x")), float(node.get("y")))
            for node in nodes
        }
        
        # Build link positions
        link_positions = np.zeros((len(links), 2))
        for i, link in enumerate(links):
            node_A = link.get("from")
            node_B = link.get("to")
            link_positions[i, 0] = (node_positions[node_A][0] + node_positions[node_B][0]) / 2
            link_positions[i, 1] = (node_positions[node_A][1] + node_positions[node_B][1]) / 2
        
        # Parse agents
        x_pos, y_pos = [], []
        valid_agents = []
        total_agents = 0
        selected_agents = 0

        for person in population:
            if isinstance(person, etree._Comment): continue

            total_agents += 1

            # Retrieve agent attributes and put some default values
            attrs = parse_person_attributes(person)
            car_avail = attrs.get("car_avail", attrs.get("carAvail", "")).lower() # Attribute has different name in different scenario

            if car_avail != "always": continue

            plan = person.find("plan")
            if plan is None: continue

            # Retrieve agent activities
            acts = extract_activities(plan)

            # We only keep exactly two activities for each agent
            # TODO : Make the number of activities dynamic for each agent
            if len(acts) < 2: continue

            try:
                x0 = float(acts[0].get("x"))
                y0 = float(acts[0].get("y"))
                x1 = float(acts[1].get("x"))
                y1 = float(acts[1].get("y"))
            except (TypeError, ValueError):
                continue # Skip malformed coordinates
                
            x_pos.extend([x0, x1])
            y_pos.extend([y0, y1])
            valid_agents.append((person, attrs, acts))
            selected_agents += 1

        
        print(f"{selected_agents}/{total_agents} agents selected (car_avail='always' and ≥2 activities) ({100 * selected_agents / total_agents:.2f}%)")


        # Fuse the x and y position
        position = np.stack((x_pos, y_pos), axis=1)

        # Use the KDBall from sklearn
        tree = KDTree(link_positions)
        _, index = tree.query(position, k=1)

        # Fetch agent features
        self.agent_features = torch.zeros((len(position), len(self)), dtype=torch.float32).to(self.device)

        for i, (person, attrs, acts) in enumerate(valid_agents):
            # We assume a default value for sex, employed and age if attribute is not present
            # TODO: Find a fix for this, IPF or something that makes sense
            sex = 1 if attrs.get("sex", "m").lower() == "f" else 0
            employed = 1 if attrs.get("employed", "no").lower() == "yes" else 0
            age = float(attrs.get("age", 0))

            # Row 1 (origin)
            self.agent_features[2*i, self.SEX] = sex
            self.agent_features[2*i, self.EMPLOYMENT_STATUS] = employed
            self.agent_features[2*i, self.AGE] = age
            self.agent_features[2*i, self.ORIGIN] = index[2*i].item()
            self.agent_features[2*i, self.DESTINATION] = index[2*i+1].item()
            dep_time = extract_departure_time(acts[0])
            if dep_time is not None:
                self.agent_features[2*i, self.DEPARTURE_TIME] = dep_time

            # Row 2 (destination)
            self.agent_features[2*i+1, self.SEX] = sex
            self.agent_features[2*i+1, self.EMPLOYMENT_STATUS] = employed
            self.agent_features[2*i+1, self.AGE] = age
            self.agent_features[2*i+1, self.ORIGIN] = index[2*i+1].item()
            self.agent_features[2*i+1, self.DESTINATION] = index[2*i].item()
            dep_time_2 = extract_departure_time(acts[1])
            if dep_time_2 is not None:
                self.agent_features[2*i+1, self.DEPARTURE_TIME] = dep_time_2

        # Ensure that the agent ID: 0 will be not add in the list of agents
        self.agent_features[0, self.DEPARTURE_TIME] = 25*3600
        print(f"Final agent_features shape: {self.agent_features.shape}")


    def insert_agent_into_network(self, x: torch.Tensor, h: FeatureHelpers) -> None:
        """
        Insert agents as much as possible in the traffic network. The origin road is mentionned in agent feature. 

        Parameters
        ----------
        x : torch.Tensor
            A tensor which contains the node features of the network, and which is update 
            at the end of function.
        h : FeatureHelpers
            Index helpers to understand how columns works
        """

        if torch.any(x[:, h.NUMBER_OF_AGENT] >= 198):
            pass

        # Create a mask in order to select agent which can insert the network
        mask = ((self.agent_features[:, self.DEPARTURE_TIME] < self.time) &       # Ensure that the time is good
                (self.agent_features[:, self.ON_WAY] == 0) &                      # Ensure that the agent is not already on the road
                (self.agent_features[:, self.DONE] == 0))                         # Ensure that the agent is not already arrived
        road_index = self.agent_features[:, self.ORIGIN].to(torch.int64)
        mask = mask & (x[road_index, h.NUMBER_OF_AGENT] < x[road_index, h.MAX_NUMBER_OF_AGENT] - h.CONGESTION_FILE)
            

        # Need to select only one person to insert per road
        agent = self.agent_features[mask]                     # agent stores the tensor of agent features to insert
        if not torch.any(mask):  # No agent to insert
            return x
        agent_index = torch.nonzero(mask).squeeze(0)
        road = agent[:, self.ORIGIN]
        nb_waiting_agent = agent_index.size(0)
        road_sorted, index_road_sorted = torch.sort(road)
        mask = torch.zeros_like(road, dtype=torch.bool)
        mask[0] = True
        mask[1:] = road_sorted[:-1] !=  road_sorted[1:]
        selected_index = index_road_sorted[mask]
        road = road[selected_index]
        agent = agent[selected_index]
        agent_index = agent_index[selected_index].flatten()
        nb_waiting_agent -= selected_index.size(0)

        # Insert all these agents
        road_select = agent[:, self.ORIGIN].to(torch.int64)
        end_queue = (x[road_select, h.NUMBER_OF_AGENT]).to(torch.int64) 
        x[road_select, end_queue] = agent_index.to(torch.float)                   # Queue for agent index
        x[road_select, h.Nmax + end_queue] = self.time                            # Queue for time arrival
        x[road_select, 2*h.Nmax + end_queue] = x[road_select, h.NUMBER_OF_AGENT]  # Queue for number of agent in queue
        x[road_select, h.NUMBER_OF_AGENT] += 1
        # Update the agent features
        self.agent_features[agent_index, self.ON_WAY] = 1.0

        # Insert the others agent
        if nb_waiting_agent > 0:
            return self.insert_agent_into_network(x, h)
        else:
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
        Chose the next direction to take for each agent

        Parameters
        ----------
        """
        node_feature = graph.x.clone()                                # Clone the node_feature for mo
        adj = to_dense_adj(graph.edge_index)                          # Compute the adjency matrix
        adj = (adj / adj.sum(dim = 1, keepdim=True)).squeeze(0)       # Normalise the adjency matrix
        index = torch.multinomial(adj, num_samples=1).squeeze(1)      # Draw the next moves
        node_feature[:, h.SELECTED_ROAD] = index.to(torch.float)      # Update the choice of user
        updated_graph = Data(x=node_feature, edge_index=graph.edge_index, edge_attr=graph.edge_attr)
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
    





