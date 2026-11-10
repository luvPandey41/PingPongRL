"""A small, educational PPO trainer written directly with PyTorch."""

import shutil
import time
from pathlib import Path

import mujoco.viewer
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from pingpong_env import PingPongEnv


# Main training settings. A new run adds this many transitions to an existing
# ppo.pt checkpoint, if one exists.
TRAINING_STEPS = 10_000_000
ROLLOUT_STEPS = 30_000
BATCH_SIZE = 250
PPO_EPOCHS = 10
LOG_EVERY_EPISODES = 1_000
VIEW_EVERY_EPISODES = 5_000
VIEW_SPEED = 0.1

LEARNING_RATE = 3e-4
GAMMA = 0.995
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
VALUE_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.01
MAX_GRAD_NORM = 0.5

MODEL_PATH = Path("models/ppo.pt")
PREVIOUS_MODEL_PATH = Path("models/ppo.previous.pt")
DEVICE = torch.device("cpu")


class ActorCritic(nn.Module):
    """The actor proposes actions; the critic estimates future reward."""

    def __init__(self, observation_size: int, action_size: int):
        super().__init__()

        # Each network is a small multilayer perceptron with two hidden layers.
        self.actor = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_size),
        )
        self.critic = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # The actor predicts a Gaussian mean; this learned parameter sets its
        # exploration spread for each of the six joint commands.
        self.log_standard_deviation = nn.Parameter(torch.zeros(action_size))

    def distribution_and_value(self, observation: torch.Tensor):
        mean = self.actor(observation)
        standard_deviation = self.log_standard_deviation.clamp(-5.0, 2.0).exp()
        distribution = Normal(mean, standard_deviation.expand_as(mean))
        value = self.critic(observation).squeeze(-1)
        return distribution, value


def squashed_log_probability(distribution: Normal, raw_action: torch.Tensor) -> torch.Tensor:
    """Log probability after tanh squashes commands into [-1, 1]."""
    action = torch.tanh(raw_action)
    correction = torch.log(1.0 - action.square() + 1e-6)
    return (distribution.log_prob(raw_action) - correction).sum(dim=-1)


def collect_rollout(model: ActorCritic, env: PingPongEnv, observation: np.ndarray, steps: int, stats: dict):
    """Run the current policy and store transitions for one PPO update."""
    observations = []
    raw_actions = []
    old_log_probabilities = []
    rewards = []
    values = []
    next_values = []
    terminated_flags = []
    episode_end_flags = []

    for _ in range(steps):
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            distribution, value = model.distribution_and_value(observation_tensor)
            raw_action = distribution.sample()
            action = torch.tanh(raw_action)
            log_probability = squashed_log_probability(distribution, raw_action)

        next_observation, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
        episode_ended = terminated or truncated

        next_observation_tensor = torch.as_tensor(next_observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            _, next_value = model.distribution_and_value(next_observation_tensor)

        observations.append(observation)
        raw_actions.append(raw_action.squeeze(0).cpu().numpy())
        old_log_probabilities.append(float(log_probability.item()))
        rewards.append(float(reward))
        values.append(float(value.item()))
        next_values.append(float(next_value.item()))
        terminated_flags.append(bool(terminated))
        episode_end_flags.append(bool(episode_ended))

        stats["episode_reward"] += float(reward)
        stats["episode_length"] += 1
        if episode_ended:
            stats["episode_count"] += 1
            stats["rewards"].append(stats["episode_reward"])
            stats["lengths"].append(stats["episode_length"])
            stats["touches"].append(int(info.get("touches", 0)))
            stats["episode_reward"] = 0.0
            stats["episode_length"] = 0

            if stats["episode_count"] % LOG_EVERY_EPISODES == 0:
                count = len(stats["rewards"])
                print(
                    f"[episodes {stats['episode_count']}] last {count}: "
                    f"mean_reward={np.mean(stats['rewards']):.4f}s, "
                    f"mean_steps={np.mean(stats['lengths']):.1f}, "
                    f"mean_paddle_touches={np.mean(stats['touches']):.2f}, "
                    f"best_touches={max(stats['touches'])}",
                    flush=True,
                )
                stats["rewards"].clear()
                stats["lengths"].clear()
                stats["touches"].clear()

            observation, _ = env.reset()
            if stats["episode_count"] % VIEW_EVERY_EPISODES == 0:
                show_episode(model, env, "periodic")
                observation, _ = env.reset()
        else:
            observation = next_observation

    rollout = {
        "observations": np.asarray(observations, dtype=np.float32),
        "raw_actions": np.asarray(raw_actions, dtype=np.float32),
        "old_log_probabilities": np.asarray(old_log_probabilities, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "values": np.asarray(values, dtype=np.float32),
        "next_values": np.asarray(next_values, dtype=np.float32),
        "terminated": np.asarray(terminated_flags, dtype=np.float32),
        "episode_ended": np.asarray(episode_end_flags, dtype=np.float32),
    }
    return rollout, observation


def calculate_advantages(rollout: dict) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE advantages and value targets from a collected rollout."""
    rewards = rollout["rewards"]
    values = rollout["values"]
    next_values = rollout["next_values"]
    terminated = rollout["terminated"]
    episode_ended = rollout["episode_ended"]

    advantages = np.zeros_like(rewards)
    next_advantage = 0.0
    for step in reversed(range(len(rewards))):
        # A time-limit truncation can bootstrap from its final observation;
        # a true terminal state cannot. Neither kind carries GAE across reset.
        bootstrap_mask = 1.0 - terminated[step]
        continuation_mask = 1.0 - episode_ended[step]
        temporal_difference = rewards[step] + GAMMA * bootstrap_mask * next_values[step] - values[step]
        next_advantage = temporal_difference + GAMMA * GAE_LAMBDA * continuation_mask * next_advantage
        advantages[step] = next_advantage

    returns = advantages + values
    return advantages, returns


def update_policy(model: ActorCritic, optimizer: torch.optim.Optimizer, rollout: dict) -> None:
    """Apply PPO's clipped policy update over shuffled minibatches."""
    advantages, returns = calculate_advantages(rollout)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    observations = torch.as_tensor(rollout["observations"], dtype=torch.float32, device=DEVICE)
    raw_actions = torch.as_tensor(rollout["raw_actions"], dtype=torch.float32, device=DEVICE)
    old_log_probabilities = torch.as_tensor(rollout["old_log_probabilities"], dtype=torch.float32, device=DEVICE)
    advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=DEVICE)
    returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=DEVICE)

    sample_count = len(observations)
    for _ in range(PPO_EPOCHS):
        indices = np.random.permutation(sample_count)
        for start in range(0, sample_count, BATCH_SIZE):
            batch_indices = indices[start:start + BATCH_SIZE]
            batch = torch.as_tensor(batch_indices, dtype=torch.long, device=DEVICE)

            distribution, predicted_values = model.distribution_and_value(observations[batch])
            new_log_probabilities = squashed_log_probability(distribution, raw_actions[batch])

            # Compare the new and behavior policies, then clip their ratio so
            # one gradient step cannot change the policy too aggressively.
            probability_ratio = torch.exp(new_log_probabilities - old_log_probabilities[batch])
            unclipped_objective = probability_ratio * advantages_tensor[batch]
            clipped_objective = torch.clamp(
                probability_ratio,
                1.0 - CLIP_EPSILON,
                1.0 + CLIP_EPSILON,
            ) * advantages_tensor[batch]
            policy_loss = -torch.minimum(unclipped_objective, clipped_objective).mean()

            value_loss = (predicted_values - returns_tensor[batch]).square().mean()
            # Gaussian entropy encourages exploration; tanh changes the exact
            # entropy, so this is the usual simple pre-squash approximation.
            entropy_bonus = distribution.entropy().sum(dim=-1).mean()
            loss = policy_loss + VALUE_COEFFICIENT * value_loss - ENTROPY_COEFFICIENT * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()


def save_checkpoint(model: ActorCritic, optimizer: torch.optim.Optimizer, total_steps: int) -> None:
    """Save weights, optimizer state, and step count for continued training."""
    temporary_path = MODEL_PATH.with_suffix(".tmp")
    torch.save(
        {
            "total_steps": total_steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        temporary_path,
    )
    temporary_path.replace(MODEL_PATH)


def show_episode(model: ActorCritic, env: PingPongEnv, label: str) -> None:
    """Show one deterministic episode at slow playback speed."""
    observation, _ = env.reset()
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            print(f"[viewer] Showing a {label} policy episode at 10% speed; close the window to continue.", flush=True)
            done = False
            while viewer.is_running():
                if not done:
                    observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    with torch.no_grad():
                        mean = model.actor(observation_tensor)
                        action = torch.tanh(mean).squeeze(0).cpu().numpy()
                    observation, _, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                viewer.sync()
                delay = 0.02 if done else env.model.opt.timestep * env.frame_skip / VIEW_SPEED
                time.sleep(delay)
    except RuntimeError as error:
        print(f"[viewer] Could not open viewer: {error}", flush=True)
        print("[viewer] On macOS, run with: mjpython ppo.py", flush=True)


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    env = PingPongEnv()
    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    total_steps = 0

    if MODEL_PATH.exists():
        shutil.copy2(MODEL_PATH, PREVIOUS_MODEL_PATH)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        total_steps = int(checkpoint["total_steps"])
        print(
            f"Resuming at {total_steps:,} steps; previous manual policy backed up to {PREVIOUS_MODEL_PATH}.",
            flush=True,
        )
    else:
        print("Starting a new PyTorch PPO policy.", flush=True)

    stats = {
        "episode_count": 0,
        "episode_reward": 0.0,
        "episode_length": 0,
        "rewards": [],
        "lengths": [],
        "touches": [],
    }
    observation, _ = env.reset()
    steps_left = TRAINING_STEPS

    while steps_left > 0:
        rollout_steps = min(ROLLOUT_STEPS, steps_left)
        rollout, observation = collect_rollout(model, env, observation, rollout_steps, stats)
        update_policy(model, optimizer, rollout)

        total_steps += rollout_steps
        steps_left -= rollout_steps
        save_checkpoint(model, optimizer, total_steps)

    print(f"Training complete at {total_steps:,} total steps. Saved {MODEL_PATH}.", flush=True)
    show_episode(model, env, "final")
    env.close()


if __name__ == "__main__":
    main()
