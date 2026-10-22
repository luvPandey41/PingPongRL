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
