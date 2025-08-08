import torch
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs.utils import ExplorationType

def ppo_train(env, policy_module, value_module, *, total_frames=128, frames_per_batch=32, num_epochs=1,
              sub_batch_size=32, device=torch.device("cpu"), checkpoint_path=None):
    """A lightweight PPO training loop used for smoke tests.

    The implementation is intentionally compact and only performs a
    single optimisation round.  It is sufficient for unit tests and
    toy examples but not meant for large scale experiments.
    """
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

    advantage_module = GAE(gamma=0.99, lmbda=0.95, value_network=value_module, average_gae=True, device=device)
    loss_module = ClipPPOLoss(actor_network=policy_module, critic_network=value_module, clip_epsilon=0.2)
    optim = torch.optim.Adam(loss_module.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=frames_per_batch),
                                 sampler=SamplerWithoutReplacement())

    for tensordict_data in collector:
        # Only collect a single batch in this minimal implementation
        for _ in range(num_epochs):
            advantage_module(tensordict_data)
            replay_buffer.extend(tensordict_data.reshape(-1).to("cpu"))
            batch = replay_buffer.sample(min(sub_batch_size, replay_buffer._storage._storage.shape[0])).to(device)
            loss_vals = loss_module(batch)
            loss = loss_vals["loss_objective"] + loss_vals["loss_critic"]
            loss.backward()
            optim.step()
            optim.zero_grad()
        break

    if checkpoint_path is not None:
        try:
            torch.save(policy_module.state_dict(), checkpoint_path)
        except Exception:
            pass
