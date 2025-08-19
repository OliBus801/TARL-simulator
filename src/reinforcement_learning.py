from torchrl.envs import EnvBase
from torchrl.data import TensorSpec, BoundedTensorSpec, UnboundedContinuousTensorSpec, CompositeSpec, UnboundedDiscreteTensorSpec
from torch.distributions import Distribution

from tensordict import TensorDict, TensorDictBase


import torch
from torch_scatter import scatter_softmax, scatter_max
import time

from src.transportation_simulator import TransportationSimulator


class GraphDistribution(Distribution):

    def __init__(self, logits: torch.Tensor, edge_index: torch.Tensor, temperature: float = 1.0):
        super().__init__()
        assert not torch.isnan(logits).any()
        self.edge_index = edge_index
        self.groups, self.index = torch.sort(edge_index[0])
        self.inv_index = torch.argsort(self.index)
        self.nodes = torch.unique(self.groups)  
        self.nb_nodes = self.nodes.size(0)
        self.proba = scatter_softmax(logits/temperature, self.edge_index[0])
        self.proba_sort = self.proba[..., self.index]
        self.log_proba_sort = torch.log(self.proba_sort+1e-8)

        # First and last indices
        self.first_indices = torch.zeros_like(self.groups, dtype=torch.bool)
        self.first_indices[1:] = (self.groups[1:] != self.groups[:-1])
        self.first_indices[0] = True
        self.last_indices = torch.zeros_like(self.groups, dtype=torch.bool)
        self.last_indices[:-1] = (self.groups[1:] != self.groups[:-1])
        self.last_indices[-1] = True

        # Compute the cummulative sum
        self.cumsum = torch.cumsum(self.proba_sort, dim=-1)
        node_shape = logits.shape[:-1] + (self.nb_nodes,)
        bsum = torch.zeros(node_shape)
        bsum[..., 1:] = self.cumsum[..., self.last_indices][..., :-1]
        self.cumsum = self.cumsum - bsum[..., self.groups]

        # Compute the deterministic sample
        _, index = scatter_max(self.proba, self.edge_index[0])
        self.deterministic_sample = torch.zeros_like(self.proba)
        if self.proba.ndim == 1:
            self.deterministic_sample[index] = 1
        elif self.proba.ndim == 2:
            batch_size = self.proba.size(0)
            n = torch.arange(batch_size).repeat(self.nb_nodes)
            n = n.view(batch_size, self.nb_nodes)
            self.deterministic_sample[n, index] = 1
        else:
            raise NotImplemented

    @property
    def mode(self):
        return self.deterministic_sample


    def sample(self, sample_shape=torch.Size()):

        # Sample (batch_size, num_nodes) values

        sample = torch.rand(sample_shape + torch.Size([self.nb_nodes]))

        # Compute the cummulative sum for sample
        sample = sample[..., self.groups]
        r = torch.where(sample < self.cumsum, 1, 0)
        r = torch.cumsum(r, dim=-1)
        r_bsum = torch.zeros_like(self.nodes)
        r_bsum[1:] = r[..., self.last_indices][:-1]
        r = r - r_bsum[self.groups]
        
        # Compute actions
        hot_ones = torch.where(r==1, 1, 0)
        hot_ones = hot_ones[..., self.inv_index] 

        return hot_ones
    
    def log_prob(self, action: torch.Tensor):

        # Check first if the action is possible
        action_sort = action[..., self.index]
        cumsum = torch.cumsum(action_sort, dim=-1) # Sort and compute cummulative sum
        possible = torch.all(cumsum[..., self.last_indices] == torch.arange(1, self.nb_nodes+1), dim=-1)

        # Compute the log
        log = torch.sum(action_sort * self.log_proba_sort, dim=-1)
        log[~possible] = -torch.inf

        return log
    
    def entropy(self):
        return -torch.sum(self.proba_sort * self.log_proba_sort, dim=-1).flatten()
    




class SimulatorEnv(EnvBase):
    """
    A custom environment for reinforcement learning based on the simpson simulator.
    """

    def __init__(self, device: str = 'cpu', timestep_size: int = 1, start_time: int = 0, scenario: str = "Easy"):
        super().__init__(device=device)
        self.simulator = TransportationSimulator(device=device)
        self.simulator.load_network(scenario=scenario)
        self.simulator.config_parameters(timestep_size=timestep_size, start_time=start_time)
        self.to(device)
        
        # Attribute for Env
        self.num_edge = self.simulator.graph.edge_index.size(1)
        self.num_node = self.simulator.graph.x.size(0)
        self.num_obs = 7

        self.reward_spec = BoundedTensorSpec(
            shape=torch.Size([1]),  # reward scalaire
            dtype=torch.float32,
            low=-1e6,
            high=1e6
        )
        self.action_spec = BoundedTensorSpec(
            shape=torch.Size([self.num_edge]),
            dtype=torch.bool,
            low=torch.zeros(self.num_edge, dtype=torch.bool),
            high=torch.ones(self.num_edge, dtype=torch.bool),
            )


        self.observation_spec = CompositeSpec(
            node_features=UnboundedContinuousTensorSpec(
                shape=(self.num_node, self.num_obs),
                dtype=torch.float32,
                device=self.device
            ),
            edge_features=UnboundedContinuousTensorSpec(
                shape=(self.num_edge, 1),
                dtype=torch.float32,
                device=self.device
            ),
            # edge_index=UnboundedDiscreteTensorSpec(
            #     shape=(2, self.num_edge),
            #     dtype=torch.int64,
            #     device=self.device
            # ),
            agent_index=UnboundedDiscreteTensorSpec(
                shape=(self.num_node,),
                dtype=torch.int64, 
                device=device
            ),
            time=UnboundedContinuousTensorSpec(
                shape=(1,),
                dtype=torch.float32,
                device=device
            ),
        )

        self.terminated_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.bool,
            low=torch.tensor(0, dtype=torch.bool, device=device),
            high=torch.tensor(1, dtype=torch.bool, device=device),
            device=device
        )

        self.state = self.simulator.state()

        # Intern attribute
        self.old_state = self.simulator.graph.x[:, self.simulator.h.NUMBER_OF_AGENT]

    def _set_seed(self, seed):
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(seed)
        return seed

    def _reset(self, tensordict: TensorDictBase = None) -> TensorDictBase:
        """
        Reset the environment to its initial state.
        """
        self.simulator.reset()
        # reset logs and timers so that evaluation metrics only reflect the new episode
        self.simulator.inserting_time = 0
        self.simulator.choice_time = 0
        self.simulator.core_time = 0
        self.simulator.withdraw_time = 0
        self.simulator.leg_histogram_values = []
        self.simulator.road_optimality_values = []
        self.simulator.on_way_before = 0
        self.simulator.done_before = 0
        if hasattr(self.simulator.model_core.response_mpnn, "update_history"):
            self.simulator.model_core.response_mpnn.update_history = []

        self.simulator.set_time(3600 * 6 - 60)
        x, edge_attr, edge_index, agent_index = self.simulator.state()
        self.simulator.agent.reset()
        reward = torch.tensor([0], dtype=torch.float, device=self.device)
        done = torch.tensor([False], dtype=torch.bool)
        terminated = torch.tensor([False], dtype=torch.bool)
        return TensorDict({
            "node_features": x,
            "edge_features": edge_attr,
            # "edge_index": edge_index,
            "agent_index": agent_index,
            "time": torch.tensor([self.simulator.time], dtype=torch.float32, device=self.device),
            #"reward": reward,
            "terminated": terminated,
            "done": done,

        }, batch_size=[])
    

    def _step(self, tensordict: TensorDictBase):
        action = tensordict["action"]
        mask = action.to(torch.bool)
        node_origin = self.simulator.graph.edge_index[0][mask]
        node_destination = self.simulator.graph.edge_index[1][mask]
        h = self.simulator.h

        # Apply action (choice phase)
        b = time.time()
        self.simulator.graph.x[node_origin, h.SELECTED_ROAD] = node_destination.to(torch.float)
        e = time.time()
        self.simulator.choice_time += e - b

        # Core model
        b = e
        self.simulator.graph = self.simulator.model_core(self.simulator.graph)
        e = time.time()
        self.simulator.core_time += e - b

        # Withdraw agents
        b = e
        last_people = self.simulator.graph.x[:, h.HEAD_FIFO].to(torch.long)
        self.simulator.graph.x = self.simulator.agent.withdraw_agent_from_network(
            self.simulator.graph.x, self.simulator.graph.edge_index, h
        )
        e = time.time()
        self.simulator.withdraw_time += e - b

        # Insert agents
        b = e
        self.simulator.graph.x = self.simulator.agent.insert_agent_into_network(self.simulator.graph, h)
        e = time.time()
        self.simulator.inserting_time += e - b

        new_state = self.simulator.graph.x[:, h.NUMBER_OF_AGENT]
        reward = torch.zeros(1, dtype=torch.float32, device=self.device)
        # Compute reward
        arrived = self.simulator.agent.agent_features[last_people, self.simulator.agent.DONE].to(torch.bool)
        rewardable_people = last_people[arrived]
        time_travel = (
            self.simulator.agent.agent_features[rewardable_people, self.simulator.agent.ARRIVAL_TIME]
            - self.simulator.agent.agent_features[rewardable_people, self.simulator.agent.DEPARTURE_TIME]
        )
        individual_reward = 0 + torch.sum(100 * 600 / time_travel)
        reward = - torch.sum(self.simulator.graph.x[:, h.NUMBER_OF_AGENT])
        reward = torch.sum(reward).flatten()

        if torch.all(self.old_state == self.simulator.graph.x[:, h.NUMBER_OF_AGENT]):
            self.simulator.set_time(self.simulator.time + self.simulator.timestep)

        self.old_state = new_state
        if self.simulator.time > 7 * 3600:
            done = torch.tensor(True)
        else:
            done = torch.tensor(False)

        # Log histogram and optimality metrics
        value_on_way = torch.sum(self.simulator.agent.agent_features[:, self.simulator.agent.ON_WAY])
        value_done = torch.sum(self.simulator.agent.agent_features[:, self.simulator.agent.DONE])
        self.simulator.leg_histogram_values.append([
            value_on_way - self.simulator.on_way_before + value_done - self.simulator.done_before,
            value_done - self.simulator.done_before,
            value_on_way,
            self.simulator.time,
        ])
        self.simulator.on_way_before = value_on_way
        self.simulator.done_before = value_done
        self.simulator.road_optimality_values.append(
            (
                self.simulator.time,
                self.simulator.model_core.direction_mpnn.road_optimality_data["delta_travel_time"].cpu(),
            )
        )

        # Compute the next state
        node_features, edge_features, edge_index, agent_index = self.simulator.state()
        terminated = done

        return TensorDict({
            "node_features": node_features,
            "edge_features": edge_features,
            # "edge_index": edge_index,
            "agent_index": agent_index,
            "time": torch.tensor([self.simulator.time], dtype=torch.float32, device=self.device),
            "reward": reward,
            "terminated": terminated,
            "done": done,
        }, batch_size=[])




        
        

