"""Replay the highest-touch episode recorded by ppo.py."""

from pingpong_env import PingPongEnv
from ppo import show_best_episode


def main() -> None:
    env = PingPongEnv()
    show_best_episode(env)
    env.close()


if __name__ == "__main__":
    main()
