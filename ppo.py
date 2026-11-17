"""A small, educational PPO trainer written directly with PyTorch."""

import shutil
import time
from collections import Counter
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from pingpong_env import PingPongEnv


# Main training settings. Training resumes until the checkpoint reaches this
# total number of policy transitions.
TRAINING_STEPS = 10_000_000
ROLLOUT_STEPS = 30_000
BATCH_SIZE = 250
PPO_EPOCHS = 10
LOG_EVERY_EPISODES = 1000
VIEW_EVERY_EPISODES = 10_000_000
VIEW_SPEED = 1
GUIDANCE_START_STRENGTH = 0.90
GUIDANCE_DECAY_STEPS = 2_000_000

LEARNING_RATE = 3e-4
GAMMA = 0.9995
GAE_LAMBDA = 0.995
CLIP_EPSILON = 0.2
VALUE_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.01
MAX_GRAD_NORM = 0.5

MODEL_PATH = Path("models/ppo.pt")
PREVIOUS_MODEL_PATH = Path("models/ppo.previous.pt")
BEST_EPISODE_PATH = Path("models/best_episode.npz")
# Checkpoints must start fresh when the policy's inputs change.
OBSERVATION_VERSION = 3  # 18-feature relative-intercept observation layout
REWARD_VERSION = 4  # next-state intercept target and unit-scale distance reward
# Bump this whenever the environment's definition of a valid rally changes.
# Version 6 also treats the paddle handle as part of the allowed assembly.
BEST_EPISODE_VERSION = 7
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
        # exploration spread for each controllable joint command.
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


def capture_state(env: PingPongEnv) -> dict:
    """Start a replay trace with the current complete MuJoCo state."""
    return {
        "qpos": [env.data.qpos.copy()],
        "qvel": [env.data.qvel.copy()],
        "times": [float(env.data.time)],
    }


def save_best_episode(trace: dict, touches: int, episode_number: int, training_steps: int) -> None:
    """Persist the highest-touch training episode as replayable physics states."""
    np.savez_compressed(
        BEST_EPISODE_PATH,
        qpos=np.asarray(trace["qpos"], dtype=np.float64),
        qvel=np.asarray(trace["qvel"], dtype=np.float64),
        times=np.asarray(trace["times"], dtype=np.float64),
        touches=np.asarray(touches, dtype=np.int64),
        episode=np.asarray(episode_number, dtype=np.int64),
        training_steps=np.asarray(training_steps, dtype=np.int64),
        version=np.asarray(BEST_EPISODE_VERSION, dtype=np.int64),
    )


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

        stats["steps_this_run"] += 1
        next_observation, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
        episode_ended = terminated or truncated
        stats["total_steps"] += 1

        for qpos, qvel, sim_time in env.last_substep_states:
            stats["current_episode_trace"]["qpos"].append(qpos)
            stats["current_episode_trace"]["qvel"].append(qvel)
            stats["current_episode_trace"]["times"].append(sim_time)

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
        stats["episode_guidance_strength"] += float(info.get("guidance_strength", 0.0))
        stats["episode_predicted_target_reward"] += float(info.get("predicted_target_reward", 0.0))
        stats["episode_approach_reward"] += float(info.get("approach_reward", 0.0))
        stats["episode_ball_alignment_reward"] += float(info.get("ball_alignment_reward", 0.0))
        stats["episode_contact_reward"] += float(info.get("contact_reward", 0.0))
        stats["episode_contact_quality_reward"] += float(info.get("contact_quality_reward", 0.0))
        stats["episode_return_reward"] += float(info.get("return_reward", 0.0))
        stats["episode_milestone_reward"] += float(info.get("milestone_reward", 0.0))
        for contact in info.get("paddle_contact_events", []):
            if contact["rally_return_index"] == 1:
                stats["first_contact_vy"].append(float(contact["ball_vy"]))
                stats["first_contact_vz"].append(float(contact["ball_vz"]))
                stats["first_contact_valid"].append(float(contact["valid_return"]))
            else:
                stats["later_contact_vy"].append(float(contact["ball_vy"]))
                stats["later_contact_vz"].append(float(contact["ball_vz"]))
                stats["later_contact_valid"].append(float(contact["valid_return"]))
        if episode_ended:
            stats["episode_count"] += 1
            episode_touches = int(info.get("touches", 0))
            if episode_touches > stats["best_touches"]:
                stats["best_touches"] = episode_touches
                save_best_episode(
                    stats["current_episode_trace"],
                    touches=episode_touches,
                    episode_number=stats["episode_count"],
                    training_steps=stats["total_steps"],
                )
                print(
                    f"[best episode] {episode_touches} paddle touches at "
                    f"{stats['total_steps']:,} training steps; saved replay to {BEST_EPISODE_PATH}.",
                    flush=True,
                )

            stats["rewards"].append(stats["episode_reward"])
            stats["lengths"].append(stats["episode_length"])
            stats["termination_reasons"].append(info.get("termination_reason", "unknown"))
            stats["guidance_strengths"].append(
                stats["episode_guidance_strength"] / max(stats["episode_length"], 1)
            )
            stats["touches"].append(episode_touches)
            is_second_practice = bool(info.get("second_return_practice", False))
            stats["practice_flags"].append(is_second_practice)
            if is_second_practice:
                stats["practice_successes"].append(float(episode_touches >= 1))
            else:
                stats["regular_touches"].append(episode_touches)
                stats["recent_regular_touches"].append(episode_touches)
                stats["recent_regular_touches"] = stats["recent_regular_touches"][-100:]
            stats["approach_rewards"].append(stats["episode_approach_reward"])
            stats["predicted_target_rewards"].append(stats["episode_predicted_target_reward"])
            stats["ball_alignment_rewards"].append(stats["episode_ball_alignment_reward"])
            stats["contact_rewards"].append(stats["episode_contact_reward"])
            stats["contact_quality_rewards"].append(stats["episode_contact_quality_reward"])
            stats["return_rewards"].append(stats["episode_return_reward"])
            stats["milestone_rewards"].append(stats["episode_milestone_reward"])
            stats["recent_touches"].append(episode_touches)
            stats["recent_touches"] = stats["recent_touches"][-100:]
            stats["episode_reward"] = 0.0
            stats["episode_length"] = 0
            stats["episode_guidance_strength"] = 0.0
            stats["episode_approach_reward"] = 0.0
            stats["episode_predicted_target_reward"] = 0.0
            stats["episode_ball_alignment_reward"] = 0.0
            stats["episode_contact_reward"] = 0.0
            stats["episode_contact_quality_reward"] = 0.0
            stats["episode_return_reward"] = 0.0
            stats["episode_milestone_reward"] = 0.0

            recent_touches = stats["recent_regular_touches"]
            if len(recent_touches) >= 50 and stats["curriculum_stage"] == 0:
                first_return_rate = np.mean(np.asarray(recent_touches) >= 1)
                if first_return_rate >= 0.15:
                    stats["curriculum_stage"] = 1
                    env.curriculum_stage = 1
                    stats["recent_touches"] = []
                    stats["recent_regular_touches"] = []
                    print(
                        f"[curriculum] Stage 1: mild shot variation after {first_return_rate:.0%} "
                        "of recent episodes reached one return.",
                        flush=True,
                    )
            elif len(recent_touches) >= 50 and stats["curriculum_stage"] == 1:
                two_return_rate = np.mean(np.asarray(recent_touches) >= 2)
                if two_return_rate >= 0.10:
                    stats["curriculum_stage"] = 2
                    env.curriculum_stage = 2
                    stats["recent_touches"] = []
                    stats["recent_regular_touches"] = []
                    print(
                        f"[curriculum] Stage 2: wider shot variation after {two_return_rate:.0%} "
                        "of recent episodes reached two returns.",
                        flush=True,
                    )

            if stats["episode_count"] % LOG_EVERY_EPISODES == 0:
                count = len(stats["rewards"])
                total_progress = min(stats["total_steps"] / TRAINING_STEPS, 1.0)
                regular_touches = stats["regular_touches"]
                window_best_touches = max(stats["touches"])
                print(
                    f"[episodes {stats['episode_count']} | "
                    f"session_steps={stats['steps_this_run']:,} | "
                    f"total_steps={stats['total_steps']:,}/{TRAINING_STEPS:,} "
                    f"({total_progress:.1%})] last {count}: "
                    f"mean_reward={np.mean(stats['rewards']):.4f}, "
                    f"mean_steps={np.mean(stats['lengths']):.1f}, "
                    f"mean_regular_touches={np.mean(regular_touches) if regular_touches else 0.0:.2f}, "
                    f"first_touch_rate={np.mean(np.asarray(regular_touches) >= 1) if regular_touches else 0.0:.0%}, "
                    f"second_touch_rate={np.mean(np.asarray(regular_touches) >= 2) if regular_touches else 0.0:.0%}, "
                    f"best_touches_window={window_best_touches}, "
                    f"best_touches_all_time={stats['best_touches']}, "
                    f"endings={dict(Counter(stats['termination_reasons']))}, "
                    f"curriculum_stage={stats['curriculum_stage']}",
                    flush=True,
                )
                stats["rewards"].clear()
                stats["lengths"].clear()
                stats["termination_reasons"].clear()
                stats["guidance_strengths"].clear()
                stats["touches"].clear()
                stats["practice_flags"].clear()
                stats["practice_successes"].clear()
                stats["regular_touches"].clear()
                stats["approach_rewards"].clear()
                stats["predicted_target_rewards"].clear()
                stats["ball_alignment_rewards"].clear()
                stats["contact_rewards"].clear()
                stats["contact_quality_rewards"].clear()
                stats["return_rewards"].clear()
                stats["milestone_rewards"].clear()
                stats["first_contact_vy"].clear()
                stats["first_contact_vz"].clear()
                stats["first_contact_valid"].clear()
                stats["later_contact_vy"].clear()
                stats["later_contact_vz"].clear()
                stats["later_contact_valid"].clear()

            observation, _ = env.reset()
            stats["current_episode_trace"] = capture_state(env)
            if stats["episode_count"] % VIEW_EVERY_EPISODES == 0:
                show_episode(model, env, "periodic")
                observation, _ = env.reset()
                stats["current_episode_trace"] = capture_state(env)
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
    # Normalize only the critic's regression error. The value predictions
    # remain in reward units for GAE, while critic gradients stay comparable
    # to the normalized policy-gradient scale.
    returns_scale = returns_tensor.std(unbiased=False).clamp_min(1.0)

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

            value_loss = (
                (predicted_values - returns_tensor[batch]) / returns_scale
            ).square().mean()
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
            "observation_version": OBSERVATION_VERSION,
            "reward_version": REWARD_VERSION,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        temporary_path,
    )
    temporary_path.replace(MODEL_PATH)


def show_episode(model: ActorCritic, env: PingPongEnv, label: str) -> None:
    """Show one deterministic episode at slow playback speed."""
    saved_guidance_strength = env.guidance_strength
    env.guidance_strength = 0.0
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

                update_intercept_marker(viewer, env)
                viewer.sync()
                delay = 0.02 if done else env.model.opt.timestep * env.frame_skip / VIEW_SPEED
                time.sleep(delay)
    except RuntimeError as error:
        print(f"[viewer] Could not open viewer: {error}", flush=True)
        print("[viewer] On macOS, run with: mjpython ppo.py", flush=True)
    finally:
        env.guidance_strength = saved_guidance_strength


def update_intercept_marker(viewer, env: PingPongEnv) -> None:
    """Draw the predicted contact point as a cyan sphere in the viewer."""
    scene = viewer.user_scn
    if scene is None:
        return
    scene.ngeom = 0
    target_x, target_z, _, valid = env.refresh_intercept_prediction()
    if not valid:
        return

    mujoco.mjv_initGeom(
        scene.geoms[0],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.025, 0.0, 0.0]),
        np.array([target_x, env._intercept_plane_y, target_z]),
        np.eye(3).ravel(),
        np.array([0.0, 0.9, 1.0, 0.85]),
    )
    scene.ngeom = 1


def show_best_episode(env: PingPongEnv) -> None:
    """Replay the saved best-touch trajectory from recorded MuJoCo states."""
    if not BEST_EPISODE_PATH.exists():
        print("[viewer] No best-episode replay has been recorded yet.", flush=True)
        return

    with np.load(BEST_EPISODE_PATH) as saved:
        if "version" not in saved.files or int(saved["version"]) != BEST_EPISODE_VERSION:
            print(
                "[viewer] Saved best episode predates the current intercept prediction; "
                "skipping that replay. A new qualifying episode will replace it during training.",
                flush=True,
            )
            return
        qpos_frames = saved["qpos"].copy()
        qvel_frames = saved["qvel"].copy()
        times = saved["times"].copy()
        touches = int(saved["touches"])
        episode_number = int(saved["episode"])
        training_steps = int(saved["training_steps"])

    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            print(
                f"[viewer] Replaying best recorded episode: {touches} touches "
                f"(episode {episode_number}, step {training_steps:,}). "
                "Close the viewer to continue.",
                flush=True,
            )
            for frame_index in range(len(times)):
                if not viewer.is_running():
                    break
                env.data.qpos[:] = qpos_frames[frame_index]
                env.data.qvel[:] = qvel_frames[frame_index]
                env.data.time = times[frame_index]
                mujoco.mj_forward(env.model, env.data)
                update_intercept_marker(viewer, env)
                viewer.sync()

                if frame_index + 1 < len(times):
                    delay = (times[frame_index + 1] - times[frame_index]) / VIEW_SPEED
                    time.sleep(delay)

            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)
    except RuntimeError as error:
        print(f"[viewer] Could not open best-episode replay: {error}", flush=True)
        print("[viewer] On macOS, run with: mjpython ppo.py", flush=True)


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    env = PingPongEnv()
    env.capture_substeps = True
    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    total_steps = 0

    if MODEL_PATH.exists():
        shutil.copy2(MODEL_PATH, PREVIOUS_MODEL_PATH)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        saved_policy = checkpoint["model_state_dict"]
        saved_observation_size = int(saved_policy["actor.0.weight"].shape[1])
        saved_action_size = int(saved_policy["log_standard_deviation"].numel())
        compatible_policy = (
            saved_observation_size == env.observation_space.shape[0]
            and saved_action_size == env.action_space.shape[0]
            and int(checkpoint.get("observation_version", 1)) == OBSERVATION_VERSION
            and int(checkpoint.get("reward_version", 0)) == REWARD_VERSION
        )
        if compatible_policy:
            try:
                model.load_state_dict(saved_policy)
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                total_steps = int(checkpoint["total_steps"])
            except (RuntimeError, ValueError):
                compatible_policy = False

        if compatible_policy:
            print(
                f"Resuming at {total_steps:,} steps; previous policy backed up to {PREVIOUS_MODEL_PATH}.",
                flush=True,
            )
        else:
            # Preserve the checkpoint, but start a fresh network when the
            # observation or action dimensions no longer match.
            model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            total_steps = 0
            print(
                f"The saved checkpoint is incompatible: dimensions "
                f"{saved_observation_size}/{saved_action_size}, versions "
                f"{checkpoint.get('observation_version', 'unknown')}/"
                f"{checkpoint.get('reward_version', 'unknown')}; current dimensions "
                f"{env.observation_space.shape[0]}/{env.action_space.shape[0]}, versions "
                f"{OBSERVATION_VERSION}/{REWARD_VERSION}. Starting fresh; old policy is backed up to "
                f"{PREVIOUS_MODEL_PATH}.",
                flush=True,
            )
    else:
        print("Starting a new PyTorch PPO policy.", flush=True)

    stats = {
        "episode_count": 0,
        "episode_reward": 0.0,
        "episode_length": 0,
        "episode_guidance_strength": 0.0,
        "episode_approach_reward": 0.0,
        "episode_predicted_target_reward": 0.0,
        "episode_ball_alignment_reward": 0.0,
        "episode_contact_reward": 0.0,
        "episode_contact_quality_reward": 0.0,
        "episode_return_reward": 0.0,
        "episode_milestone_reward": 0.0,
        "total_steps": 0,
        "steps_this_run": 0,
        "best_touches": 0,
        "curriculum_stage": 0,
        "current_episode_trace": None,
        "rewards": [],
        "lengths": [],
        "termination_reasons": [],
        "guidance_strengths": [],
        "touches": [],
        "recent_touches": [],
        "regular_touches": [],
        "recent_regular_touches": [],
        "practice_flags": [],
        "practice_successes": [],
        "approach_rewards": [],
        "predicted_target_rewards": [],
        "ball_alignment_rewards": [],
        "contact_rewards": [],
        "contact_quality_rewards": [],
        "return_rewards": [],
        "milestone_rewards": [],
        "first_contact_vy": [],
        "first_contact_vz": [],
        "first_contact_valid": [],
        "later_contact_vy": [],
        "later_contact_vz": [],
        "later_contact_valid": [],
    }
    if BEST_EPISODE_PATH.exists():
        with np.load(BEST_EPISODE_PATH) as saved:
            if "version" in saved.files and int(saved["version"]) == BEST_EPISODE_VERSION:
                stats["best_touches"] = int(saved["touches"])
            else:
                print("Ignoring best-episode trace recorded under an older intercept predictor.", flush=True)
    stats["total_steps"] = total_steps
    observation, _ = env.reset()
    stats["current_episode_trace"] = capture_state(env)
    steps_left = max(TRAINING_STEPS - total_steps, 0)
    if steps_left == 0:
        print(
            f"Checkpoint already reached the {TRAINING_STEPS:,}-step training target.",
            flush=True,
        )

    while steps_left > 0:
        rollout_steps = min(ROLLOUT_STEPS, steps_left)
        rollout, observation = collect_rollout(model, env, observation, rollout_steps, stats)
        update_policy(model, optimizer, rollout)

        total_steps = stats["total_steps"]
        steps_left -= rollout_steps
        save_checkpoint(model, optimizer, total_steps)

    print(f"Training complete at {total_steps:,} total steps. Saved {MODEL_PATH}.", flush=True)
    show_best_episode(env)
    show_episode(model, env, "final")
    env.close()


if __name__ == "__main__":
    main()
