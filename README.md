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
ball in a self-rally. Actions are five normalized joint targets; the jaw stays
closed. Observations include joint positions and velocities, plus the distance
and direction from the paddle to the predicted future ball intercept, time to
intercept, and a validity flag. Absolute ball and paddle xyz coordinates are
hidden from the policy. The predictor estimates table and wall bounces. The
sole reward is `exp(-distance / 0.30)`, where distance is measured from the
paddle to the next predicted intercept after each action's physics steps. A
paddle hit therefore changes the scored target to the next predicted return.
Training uses no scripted interception guidance. The viewer draws the predicted target
as a cyan marker. The initial shot is fixed; shot variation and practice from
saved post-return states unlock only after the policy reaches return thresholds.
Randomized starts are resampled when they have no valid predicted intercept.

Inspect the predictor without a trained policy by running the arm in its
ready pose:

```bash
mjpython view_intercept.py
```

The cyan marker uses approximate table and wall bounces. Compare it with the
ball's actual crossing before starting a long training run.

The main trainer is `ppo.py`, which implements PPO directly in PyTorch.
Install the dependencies and start training on macOS with:

```bash
mjpython ppo.py
```

It trains to 10,000,000 total policy steps and saves the active policy to
`models/ppo.pt`. Each policy step advances five MuJoCo physics steps. An
interrupted compatible run resumes toward the same total target; the active
checkpoint is backed up to `models/ppo.previous.pt`. Changed observation or
reward versions start a fresh policy. Training prints step progress and
touch/episode-ending summaries every 1,000 completed
episodes, then shows the best and final episodes. Use `mjpython` on macOS for
the viewer. Run `python diagnose_policy.py` to evaluate 50 fixed-shot episodes
and see target distance, touches, and termination causes.

The trainer backs up an existing checkpoint before each run. If its observation
or action dimensions do not match the current environment, it starts a fresh
policy and preserves the old checkpoint as the backup.

When training discovers a new episode with the highest valid paddle-touch
count, `ppo.py` records its MuJoCo states in `models/best_episode.npz` and
replays it in the viewer at the end of training. Episodes from earlier reward
rules are skipped by the replay viewer.

Replay the saved best episode without training again with:

```bash
mjpython view_best_episode.py
```

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
