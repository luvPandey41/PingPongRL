# PingPongRL

This is really just the environemnt for the ping pong RL project. Currently, everything is at 10% speed so its actualyl viewable. Ready for adding changes. 


A MuJoCo scene with an SO-ARM100 arm, a paddle, a ping-pong ball, and a wall. The `main.py` demo starts the arm in a ready pose and launches the ball into the paddle, then toward the wall.

## Requirements

- Python 3.14 (the version used for this project)
- MuJoCo, installed from `requirements.txt`

## Setup

Create and activate a virtual environment, then install the dependency:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

On macOS, use MuJoCo's `mjpython` launcher because the demo uses the passive viewer:

```bash
mjpython main.py
```

On Linux or Windows, run:

```bash
python main.py
```

The terminal prints the ball's contacts with the paddle and wall.

## PPO training

`pingpong_env.py` provides a Gymnasium environment for learning to keep the
ball in a self-rally. Actions are six normalized joint targets. Observations
include ball position and velocity, all six joint positions and velocities,
and paddle position. A paddle touch earns the elapsed time since the previous
paddle touch, so cumulative reward measures the interval from the first touch
to the last touch. Wall and green-table contacts are allowed during the rally;
a ball that hits the surrounding ground plane or leaves the play area ends the
episode.

The main trainer is `ppo.py`, which implements PPO directly in PyTorch.
Install the dependencies and start training on macOS with:

```bash
mjpython ppo.py
```

It trains for 1,000,000 additional steps each run and saves the active policy
to `models/ppo.pt`. Later runs resume that policy and back it up to
`models/ppo.previous.pt`. It logs every 1,000 completed episodes and opens a
viewer every 5,000 episodes plus once at the end. Use `mjpython` on macOS for
the viewer.

To inspect the saved policy without starting another training run, use:

```bash
mjpython view_policy.py
```

To check that the ball's free-fall motion matches the configured gravity,
run this headless physics sanity check:

```bash
python check_ball_physics.py
```

`train_ppo.py` is the earlier Stable-Baselines3 trainer. It remains separate
from `ppo.py` and uses a different checkpoint format; use `ppo.py` for the
manual PyTorch policy described above.
