import time
import torch
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.envs.utils import set_exploration_type, ExplorationType
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

def ppo_train(env, policy_module, value_module, *, total_frames=128, frames_per_batch=32, num_epochs=1,
              sub_batch_size=32, device=torch.device("cpu"), checkpoint_path=None,
              log_dir=None, eval_env=None, eval_interval=0, log_interval=1, stochastic_eval=False):
    """A lightweight PPO training loop used for smoke tests.

    The implementation is intentionally compact but supports multiple
    optimisation epochs over each collected batch.  It is sufficient for
    unit tests and toy examples but not meant for large scale experiments.
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

    writer = SummaryWriter(log_dir) if log_dir is not None else None
    global_step = 0
    pbar = tqdm(total=total_frames, desc="PPO training", disable=total_frames is None)


    advantage_module = GAE(
        gamma=0.99,
        lmbda=0.95,
        value_network=value_module,
        average_gae=True,
        device=device,
    )
    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=0.2,
    )
    optim = torch.optim.Adam(loss_module.parameters(), lr=1e-3)

    def _log_training(it, batch_td, loss_vals, loss, grad_norm):
        if writer is None or it % log_interval:
            return

        # Average episode return from collected batch
        returns = []
        cumulative = 0.0
        dones = batch_td["next", "done"].view(-1)
        rewards = batch_td["next", "reward"].view(-1)
        for r, d in zip(rewards, dones):
            cumulative += r.item()
            if d:
                returns.append(cumulative)
                cumulative = 0.0
        if returns:
            avg_return = sum(returns) / len(returns)
        else:
            avg_return = cumulative

        writer.add_scalar("PPO/avg_episode_return", avg_return, global_step)
        writer.add_scalar("loss/objective", loss_vals.get("loss_objective", torch.tensor(float("nan"))).item(), global_step)
        writer.add_scalar("loss/value", loss_vals.get("loss_critic", torch.tensor(float("nan"))).item(), global_step)
        writer.add_scalar("loss/entropy", loss_vals.get("loss_entropy", torch.tensor(float("nan"))).item(), global_step)
        writer.add_scalar("loss/total", loss.item(), global_step)
        writer.add_scalar("approx_kl", loss_vals.get("approx_kl", torch.tensor(float("nan"))).item(), global_step)
        writer.add_scalar("clip_fraction", loss_vals.get("clip_fraction", torch.tensor(float("nan"))).item(), global_step)
        writer.add_scalar("grad_global_norm", grad_norm, global_step)
        writer.flush()

        # Transport metrics
        if hasattr(env, "simulator"):
            sim = env.simulator
            agent = sim.agent
            h = sim.h
            # avg travel time
            mask = agent.agent_features[:, agent.DONE] == 1
            if mask.any():
                avg_tt = torch.mean(
                    agent.agent_features[mask, agent.ARRIVAL_TIME] -
                    agent.agent_features[mask, agent.DEPARTURE_TIME]
                ).item()
                writer.add_scalar("transport/avg_travel_time", avg_tt, global_step)
            # vc ratio
            num = sim.graph.x[:, h.NUMBER_OF_AGENT]
            cap = sim.graph.x[:, h.MAX_NUMBER_OF_AGENT].clamp(min=1)
            vc = num / cap
            writer.add_scalar("transport/avg_vc_ratio", vc.mean().item(), global_step)
            writer.add_scalar("transport/std_vc_ratio", vc.std(unbiased=False).item(), global_step)

    def _evaluate(prefix, exploration_type):
        if writer is None or eval_env is None:
            return
        start = time.perf_counter()
        with torch.no_grad():
            with set_exploration_type(exploration_type):
                n_steps = (eval_env.end_time - eval_env.start_time) // eval_env.simulator.timestep
                td = eval_env.rollout(n_steps, policy_module, break_when_any_done=True)
        comp_ms = (time.perf_counter() - start) * 1000.0
        rewards = td["next", "reward"].view(-1)
        avg_return = rewards.sum().item()
        episode_len = rewards.shape[0]
        writer.add_scalar(f"{prefix}/avg_return", avg_return, global_step)
        writer.add_scalar(f"{prefix}/episode_len", episode_len, global_step)
        writer.add_scalar(f"{prefix}/computation_time_ms", comp_ms, global_step)

        sim = eval_env.simulator if hasattr(eval_env, "simulator") else None
        if sim is not None:
            # Figures
            fig = sim.plot_leg_histogram(output_dir=None)
            if fig is not None:
                writer.add_figure(f"{prefix}/leg_histogram", fig, global_step)
                import matplotlib.pyplot as plt
                plt.close(fig)
            fig = sim.plot_road_optimality(output_dir=None)
            if fig is not None:
                writer.add_figure(f"{prefix}/road_optimality_graph", fig, global_step)
                import matplotlib.pyplot as plt
                plt.close(fig)
            # Node metrics
            try:
                node_metrics = sim.compute_node_metrics(output_dir=None)
                if node_metrics:
                    import torch as _t
                    avg_vc = _t.tensor([m['avg_vc'] for m in node_metrics.values()])
                    std_vc = _t.tensor([m['std_vc'] for m in node_metrics.values()])
                    writer.add_histogram(f"{prefix}/nodes_metrics/avg_vc", avg_vc, global_step)
                    writer.add_histogram(f"{prefix}/nodes_metrics/std_vc", std_vc, global_step)
            except Exception:
                pass

        writer.flush()

    for i, tensordict_data in enumerate(collector):
        global_step += frames_per_batch

        # Compute advantages once for the whole batch
        advantage_module(tensordict_data)
        data = tensordict_data.reshape(-1).to(device)

        loss_vals_acc = {}
        loss_acc = 0.0
        grad_norm_acc = 0.0
        n_updates = 0

        for _ in range(num_epochs):
            perm = torch.randperm(data.shape[0])
            for idx in perm.split(sub_batch_size):
                batch = data[idx]
                loss_vals = loss_module(batch)
                loss = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals.get("loss_entropy", 0)
                )
                loss.backward()

                grads = [p.grad for p in loss_module.parameters() if p.grad is not None]
                grad_norm = 0.0
                if grads:
                    grad_norm = torch.norm(torch.stack([g.norm() for g in grads])).item()
                optim.step()
                optim.zero_grad()

                n_updates += 1
                for k, v in loss_vals.items():
                    loss_vals_acc[k] = loss_vals_acc.get(k, 0.0) + v.detach()
                loss_acc += loss.detach()
                grad_norm_acc += grad_norm

        if n_updates:
            avg_loss_vals = {k: v / n_updates for k, v in loss_vals_acc.items()}
            avg_loss = loss_acc / n_updates
            avg_grad_norm = grad_norm_acc / n_updates
        else:
            avg_loss_vals = loss_vals_acc
            avg_loss = torch.tensor(0.0)
            avg_grad_norm = 0.0

        _log_training(i, tensordict_data, avg_loss_vals, avg_loss, avg_grad_norm)
        pbar.update(frames_per_batch)
        if eval_interval and i % eval_interval == 0:
            _evaluate("eval", ExplorationType.MODE)
            if stochastic_eval:
                _evaluate("eval_stochastic", ExplorationType.RANDOM)

    if writer is not None:
        writer.close()
    pbar.close()

    if checkpoint_path is not None:
        try:
            torch.save(policy_module.state_dict(), checkpoint_path)
        except Exception:
            pass
