import torch, matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

# ────────────────────────────────── TorchRL / TensorDict ──────────────────────────────────
from torch.utils.tensorboard import SummaryWriter
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import StepCounter
from torchrl.envs.utils import ExplorationType, set_exploration_type, check_env_specs
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from tensordict.nn import TensorDictModule

# ──────────────────────────────── code spécifique au projet ───────────────────────────────
from src.agents.mpnn_agent import MPNNPolicyNet, MPNNValueNet, MPNNValueNetSimple
from src.reinforcement_learning import SimulatorEnv, GraphDistribution

# ────────────────────────────────────── Hyper‑paramètres ──────────────────────────────────
sub_batch_size   = 100
num_epochs       = 20
clip_epsilon     = 0.2
gamma, lmbda     = 0.99, 0.95
entropy_eps      = 1e-4
frames_per_batch = 1100
total_frames     = 200_000
device           = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
lr_actor, lr_critic, max_grad_norm = 5e-3, 5e-4, 0.1
coeff_critic, coeff_exploration = 2e-4, 1000
temperature = 0.1


# ──────────────────────────────── Tracking ────────────────────────────────────────────────
log_dir = "runs/easy_ppo_" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir=log_dir)


# ──────────────────────────────── Environnement + Transforms ──────────────────────────────
base_env = SimulatorEnv()
#base_env.simulator.config_parameters(timestep=6, leg_histogram=True)
env = TransformedEnv(base_env, StepCounter())     # ← ajoute ("collector","step_count")


# agents & networks
free_flow_time_travel = base_env.simulator.graph.x[:, base_env.simulator.h.FREE_FLOW_TIME_TRAVEL]
edge_index = base_env.simulator.graph.edge_index
free_flow_time_travel = free_flow_time_travel[edge_index[1]].to(device)

agent      = MPNNPolicyNet(edge_index,
                      base_env.simulator.graph.x.size(0), 
                      free_flow_time_travel, 
                      device='cpu')

agent.load("save/Easy_population_PT.pt")
value_net  = MPNNValueNetSimple(base_env.simulator.graph.edge_index,
                      base_env.simulator.graph.x.size(0), device='cpu')
value_net.load("save/Easy_population_PT.pt")
env.base_env.simulator.agent = agent

check_env_specs(env)

# ───────────────────────────────────── Modules RL ─────────────────────────────────────────
policy_tdmodule = TensorDictModule(
    agent,
    in_keys=["node_features", "edge_features", "agent_index"],
    out_keys=["logits"]
)
policy_module = ProbabilisticActor(
    module=policy_tdmodule,
    spec=env.action_spec,
    distribution_class=GraphDistribution,
    in_keys=["logits"],
    distribution_kwargs={"edge_index": env.base_env.simulator.graph.edge_index,
                         "temperature": temperature},
    return_log_prob=True,
)

value_tdmodule = TensorDictModule(
    value_net,
    in_keys=["node_features", "edge_features", "agent_index", "time"],
    out_keys=["value"],
)
value_module = ValueOperator(
    module=value_tdmodule,
    in_keys=["node_features", "edge_features", "agent_index", "time"],
)

# ───────────────────────────────────── Collector ──────────────────────────────────────────
collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
    reset_at_each_iter=True,
    exploration_type=ExplorationType.RANDOM,
)

# ───────────────────────────────── Replay buffer ──────────────────────────────────────────
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)

# ──────────────────────────────────── Losses ──────────────────────────────────────────────
advantage_module = GAE(
    gamma=gamma, lmbda=lmbda,
    value_network=value_module,
    average_gae=True, device=device
)
loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam([
    {"params": loss_module.actor_network.parameters(), "lr": lr_actor},
    {"params": loss_module.critic_network.parameters(), "lr": lr_critic},
])
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)

# ──────────────────────────────────── Boucle PPO ──────────────────────────────────────────
logs, pbar = defaultdict(list), tqdm(total=total_frames)
eval_str = ""

for i, tensordict_data in enumerate(collector):
    for _ in range(num_epochs):
        advantage_module(tensordict_data)
        replay_buffer.extend(tensordict_data.reshape(-1).cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata     = replay_buffer.sample(sub_batch_size).to(device)
            loss_vals   = loss_module(subdata)
            loss_value  = (loss_vals["loss_objective"] +
                           loss_vals["loss_critic"] * coeff_critic +
                           loss_vals["loss_entropy"] * coeff_exploration)
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)

            optim.step(); optim.zero_grad()

    # ─────────── Logs entraînement ───────────
    step = i * frames_per_batch
    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    logs["lr"].append(optim.param_groups[0]["lr"])
    writer.add_scalar("train/avg_reward", logs["reward"][-1], step)
    writer.add_scalar("train/max_step", logs["step_count"][-1], step)
    writer.add_scalar("train/lr", logs["lr"][-1], step)
    writer.add_scalar("train/value_loss", loss_vals["loss_critic"].mean().item(), step)
    writer.add_scalar("train/loss_objective", loss_vals["loss_objective"].mean().item(), step)
    writer.add_scalar("train/loss_entropy", loss_vals["loss_entropy"].mean().item(), step)

    cum_reward_str = f"avg reward={logs['reward'][-1]:.4f}"
    stepcount_str  = f"max step={logs['step_count'][-1]}"
    lr_str         = f"lr={logs['lr'][-1]:.4f}"

    # ─────────── Évaluation périodique ───────────
    if i % 3 == 0:
        with set_exploration_type(ExplorationType.RANDOM), torch.no_grad():
            eval_rollout = env.rollout(5000, policy_module, break_when_any_done=True)
            logs["eval reward (sum)"].append(eval_rollout["next", "reward"].sum().item())
            logs["eval step_count"].append(eval_rollout["step_count"].max().item())
            eval_str = (f"eval return={logs['eval reward (sum)'][-1]:.4f}, "
                        f"eval max step={logs['eval step_count'][-1]}")
            writer.add_scalar("eval/return_sum", logs["eval reward (sum)"][-1], step)
            writer.add_scalar("eval/max_step", logs["eval step_count"][-1], step)

    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))
    pbar.update(tensordict_data.numel())
    scheduler.step()

# ─────────────────────────────── Visualisation ────────────────────────────────
plt.figure(figsize=(10,10))
plt.subplot(2,2,1); plt.plot(logs["reward"]);              plt.title("training rewards (avg)")
plt.subplot(2,2,2); plt.plot(logs["step_count"]);          plt.title("max step (train)")
plt.subplot(2,2,3); plt.plot(logs["eval reward (sum)"]);   plt.title("return (test)")
plt.subplot(2,2,4); plt.plot(logs["eval step_count"]);     plt.title("max step (test)")
plt.tight_layout(); plt.show()

writer.close()
