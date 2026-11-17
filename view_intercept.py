"""Show the ball intercept prediction with the arm held in its ready pose."""

import time

import mujoco.viewer

from pingpong_env import PingPongEnv, READY_POSE
from ppo import VIEW_SPEED, update_intercept_marker


def main() -> None:
    env = PingPongEnv()
    observation, _ = env.reset(seed=0)
    ready_action = (
        2.0 * (READY_POSE[:-1] - env.ctrl_low) / (env.ctrl_high - env.ctrl_low) - 1.0
    )
    print("Cyan sphere = predicted paddle contact point; close the viewer to exit.", flush=True)

    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            while viewer.is_running():
                if observation is None:
                    observation, _ = env.reset()

                observation, _, terminated, truncated, _ = env.step(ready_action)
                if terminated or truncated:
                    observation = None

                update_intercept_marker(viewer, env)
                viewer.sync()
                time.sleep(env.model.opt.timestep * env.frame_skip / VIEW_SPEED)
    except RuntimeError as error:
        print(f"Could not open the MuJoCo viewer: {error}", flush=True)
        print("On macOS, run with: mjpython view_intercept.py", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
