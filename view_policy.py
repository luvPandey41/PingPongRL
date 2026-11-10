"""View one episode from the saved policy without running more training."""

import time

import mujoco.viewer
import torch

from pingpong_env import PingPongEnv
from ppo import ActorCritic, DEVICE, MODEL_PATH, VIEW_SPEED


def main() -> None:
    env = PingPongEnv()
    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    observation, _ = env.reset()
    done = False

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        print(f"Showing one episode at {VIEW_SPEED:.0%} speed. Close the viewer to exit.", flush=True)
        while viewer.is_running():
            if not done:
                observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    action = torch.tanh(model.actor(observation_tensor)).squeeze(0).cpu().numpy()
                observation, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

            viewer.sync()
            time.sleep(0.02 if done else env.model.opt.timestep * env.frame_skip / VIEW_SPEED)

    env.close()


if __name__ == "__main__":
    main()
