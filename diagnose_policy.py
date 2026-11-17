"""Evaluate the saved policy using the current relative-intercept observation."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from pingpong_env import PingPongEnv
from ppo import ActorCritic, DEVICE, MODEL_PATH


def evaluate_episode(env, model, seed, curriculum_stage):
    env.guidance_strength = 0.0
    env.curriculum_stage = curriculum_stage
    env.second_return_practice_probability = 0.0
    env._first_return_state_bank.clear()
    observation, reset_info = env.reset(seed=seed)
    start_ball = env.data.qpos[env.ball_qpos:env.ball_qpos + 3].copy()
    start_velocity = env.data.qvel[env.ball_qvel:env.ball_qvel + 3].copy()
    start_prediction_valid = bool(env._intercept_prediction[3])
    start_prediction_reason = env._intercept_debug.get("reason", "unknown")
    terminated = truncated = False
    steps = valid_steps = 0
    total_reward = 0.0
    target_distances = []
    action_magnitudes = []
    contact_errors = []
    contact_details = []
    last_info = {}

    while not (terminated or truncated):
        if env._intercept_prediction[3]:
            target_x, target_z, _, _ = env._intercept_prediction
            target = np.array([target_x, env._intercept_plane_y, target_z])
            target_distances.append(
                float(np.linalg.norm(target - env.data.geom_xpos[env.paddle_geom]))
            )
            valid_steps += 1

        obs_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=DEVICE
        ).unsqueeze(0)
        with torch.no_grad():
            action = torch.tanh(model.actor(obs_tensor)).squeeze(0).cpu().numpy()
        action_magnitudes.append(float(np.mean(np.abs(action))))
        observation, reward, terminated, truncated, last_info = env.step(action)
        total_reward += float(reward)
        steps += 1

        for event in last_info.get("paddle_contact_events", []):
            contact_details.append({
                "incoming_velocity": event["incoming_velocity"].copy(),
                "outgoing_velocity": event["outgoing_velocity"].copy(),
                "paddle_normal": event["paddle_normal"].copy(),
            })
            if event.get("prediction_valid", False):
                contact_errors.append(
                    float(np.hypot(event["target_error_x"], event["target_error_z"]))
                )

    return {
        "seed": seed,
        "start_ball": start_ball,
        "start_velocity": start_velocity,
        "start_prediction_valid": start_prediction_valid,
        "start_prediction_reason": start_prediction_reason,
        "shot_resamples": reset_info.get("shot_resamples", 0),
        "steps": steps,
        "touches": int(last_info.get("touches", 0)),
        "reward": total_reward,
        "valid_rate": valid_steps / max(steps, 1),
        "closest_distance": min(target_distances) if target_distances else np.nan,
        "mean_abs_action": float(np.mean(action_magnitudes)) if action_magnitudes else 0.0,
        "contact_errors": contact_errors,
        "contact_details": contact_details,
        "termination_reason": last_info.get("termination_reason", "unknown"),
        "illegal_geoms": last_info.get("illegal_robot_contact_geoms", []),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    env = PingPongEnv()
    try:
        checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
        model = ActorCritic(
            env.observation_space.shape[0], env.action_space.shape[0]
        ).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        print(
            f"checkpoint={args.checkpoint} steps={checkpoint.get('total_steps', 'unknown')} "
            f"observation_version={checkpoint.get('observation_version', 'unknown')} "
            f"reward_version={checkpoint.get('reward_version', 'unknown')}"
        )
        results = []
        for episode_index in range(args.episodes):
            result = evaluate_episode(
                env, model, args.seed + episode_index, args.stage
            )
            results.append(result)
            contacts = result["contact_details"]
            mean_outgoing = (
                np.mean([item["outgoing_velocity"] for item in contacts], axis=0).round(2).tolist()
                if contacts else None
            )
            mean_normal = (
                np.mean([item["paddle_normal"] for item in contacts], axis=0).round(2).tolist()
                if contacts else None
            )
            print(
                f"episode={episode_index + 1}/{args.episodes} seed={result['seed']} "
                f"start_ball_xyz={result['start_ball'].round(3).tolist()} "
                f"start_ball_vxyz={result['start_velocity'].round(2).tolist()} "
                f"start_target_valid={result['start_prediction_valid']} "
                f"prediction_reason={result['start_prediction_reason']} "
                f"shot_resamples={result['shot_resamples']} "
                f"touches={result['touches']} steps={result['steps']} "
                f"reward={result['reward']:.2f} valid_prediction={result['valid_rate']:.0%} "
                f"closest_target_distance={result['closest_distance']:.3f}m "
                f"mean_abs_action={result['mean_abs_action']:.3f} "
                f"mean_outgoing_vxyz={mean_outgoing} mean_paddle_normal={mean_normal} "
                f"end={result['termination_reason']} illegal_geoms={result['illegal_geoms']}"
            )

        causes = Counter(result["termination_reason"] for result in results)
        touch_counts = [result["touches"] for result in results]
        contact_errors = [
            error for result in results for error in result["contact_errors"]
        ]
        all_contacts = [
            contact for result in results for contact in result["contact_details"]
        ]
        distances = [
            result["closest_distance"]
            for result in results
            if np.isfinite(result["closest_distance"])
        ]
        mean_distance = f"{np.mean(distances):.3f}m" if distances else "n/a"
        mean_contact_error = f"{np.mean(contact_errors):.3f}m" if contact_errors else "n/a"
        mean_outgoing = (
            np.mean([item["outgoing_velocity"] for item in all_contacts], axis=0).round(2).tolist()
            if all_contacts else None
        )
        mean_normal = (
            np.mean([item["paddle_normal"] for item in all_contacts], axis=0).round(2).tolist()
            if all_contacts else None
        )
        print(
            f"summary stage={args.stage} episodes={len(results)} "
            f"valid_start_rate={np.mean([r['start_prediction_valid'] for r in results]):.0%} "
            f"mean_shot_resamples={np.mean([r['shot_resamples'] for r in results]):.2f} "
            f"mean_touches={np.mean(touch_counts):.2f} "
            f"first_touch_rate={np.mean(np.asarray(touch_counts) >= 1):.0%} "
            f"mean_closest_target_distance={mean_distance} "
            f"mean_contact_target_error={mean_contact_error} "
            f"mean_outgoing_vxyz={mean_outgoing} mean_paddle_normal={mean_normal} "
            f"termination_counts={dict(causes)}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
