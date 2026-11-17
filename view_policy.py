"""View one episode from the saved policy without running more training."""

import time

import mujoco.viewer
import torch

from pingpong_env import PingPongEnv
from ppo import ActorCritic, DEVICE, MODEL_PATH, VIEW_SPEED, update_intercept_marker


def main() -> None:
    env = PingPongEnv()
    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as error:
        env.close()
        raise SystemExit(
            "The saved policy uses different observation or action dimensions. Run mjpython ppo.py; "
            "the trainer will back up the old checkpoint and create a compatible policy."
        ) from error
    print(
        f"Loaded checkpoint: training_steps={checkpoint.get('total_steps', 'unknown')}, "
        f"reward_version={checkpoint.get('reward_version', 'unknown')}",
        flush=True,
    )
    observation, _ = env.reset()
    done = False
    last_action = [0.0] * env.action_space.shape[0]
    last_info = {}

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        print(f"Showing one episode at {VIEW_SPEED:.0%} speed. Close the viewer to exit.", flush=True)
        last_position_report = 0.0
        while viewer.is_running():
            if not done:
                observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    action = torch.tanh(model.actor(observation_tensor)).squeeze(0).cpu().numpy()
                last_action = action.copy()
                observation, _, terminated, truncated, last_info = env.step(action)
                done = terminated or truncated

            update_intercept_marker(viewer, env)
            now = time.monotonic()
            if now - last_position_report >= 0.75:
                paddle_pos = env.data.geom_xpos[env.paddle_geom]
                ball_pos = env.data.qpos[env.ball_qpos:env.ball_qpos + 3]
                ball_vel = env.data.qvel[env.ball_qvel:env.ball_qvel + 3]
                target_x, target_z, time_to_intercept, valid = env._intercept_prediction
                if valid:
                    intercept_pos = [target_x, env._intercept_plane_y, target_z]
                    distance = float(((paddle_pos - intercept_pos) ** 2).sum() ** 0.5)
                    intercept_text = f"({target_x:+.3f}, {env._intercept_plane_y:+.3f}, {target_z:+.3f})"
                    intercept_detail = f"distance={distance:.3f}m, time={time_to_intercept:.3f}s"
                else:
                    intercept_text = "invalid"
                    intercept_detail = f"reason={env._intercept_debug.get('reason', 'unknown')}"
                contact_flags = ", ".join(
                    f"{key}:{last_info.get(key)}"
                    for key in (
                        "paddle_touch",
                        "wall_contact",
                        "table_contact",
                        "ground_contact",
                        "illegal_robot_contact",
                        "illegal_robot_contact_geoms",
                    )
                )
                print(
                    f"t={env.data.time:.3f}s done={done} "
                    f"paddle_xyz=({paddle_pos[0]:+.3f},{paddle_pos[1]:+.3f},{paddle_pos[2]:+.3f}) "
                    f"ball_xyz=({ball_pos[0]:+.3f},{ball_pos[1]:+.3f},{ball_pos[2]:+.3f}) "
                    f"ball_vxyz=({ball_vel[0]:+.2f},{ball_vel[1]:+.2f},{ball_vel[2]:+.2f}) "
                    f"intercept_xyz={intercept_text} valid={bool(valid)} {intercept_detail}\n"
                    f"  joints_rad={env.data.qpos[env.qpos_ids[:-1]].round(3).tolist()} "
                    f"action_norm={last_action.round(3).tolist()} "
                    f"reward={last_info.get('predicted_target_reward', 0.0):.3f} "
                    f"reward_target=({last_info.get('reward_target_x', 0.0):+.3f},"
                    f"{env._intercept_plane_y:+.3f},"
                    f"{last_info.get('reward_target_z', 0.0):+.3f}) "
                    f"end_reason={last_info.get('termination_reason', 'running')} "
                    f"contacts={{{contact_flags}}}",
                    flush=True,
                )
                last_position_report = now
            viewer.sync()
            time.sleep(0.02 if done else env.model.opt.timestep * env.frame_skip / VIEW_SPEED)

    print()
    env.close()


if __name__ == "__main__":
    main()
