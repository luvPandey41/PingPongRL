"""Gymnasium environment for learning sustained SO-ARM100 ping-pong rallies."""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


SCENE_PATH = Path(__file__).parent / "mujoco_menagerie/trs_so_arm100/pingpong_scene.xml"
JOINT_NAMES = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
READY_POSE = np.array([-1.54, -3.32, 0.34, -0.0332, -1.93, -0.174], dtype=np.float64)


class PingPongEnv(gym.Env):
    """One episode is one ball's attempt to sustain a self-rally.

    A positive reward is issued when the paddle hits the ball again. Its size
    is the elapsed simulation time since the preceding paddle hit, so returns
    directly add up the time between the first and last touches.
    """

    metadata = {"render_modes": []}

    def __init__(self, frame_skip: int = 5, max_episode_seconds: float = 8.0):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_episode_steps = int(max_episode_seconds / (self.model.opt.timestep * frame_skip))
        self.step_count = 0

        self.joint_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES])
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        self.qvel_ids = self.model.jnt_dofadr[self.joint_ids]
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        self.ball_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        self.ball_qpos = self.model.jnt_qposadr[self.ball_joint_id]
        self.ball_qvel = self.model.jnt_dofadr[self.ball_joint_id]
        self.ball_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self.paddle_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "pingpong_paddle")
        self.table_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "pingpong_floor")
        self.ground_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.wall_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "pingpong_wall")

        # Ball xyz + velocity xyz, six joint angles + velocities, paddle xyz.
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(21,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.last_touch_time = None
        self.touch_count = 0
        self._ball_was_touching_paddle = False
        self._ball_was_touching_ground = False

    def _contacts(self):
        paddle = table = ground = wall = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if self.ball_geom in pair:
                paddle |= self.paddle_geom in pair
                table |= self.table_geom in pair
                ground |= self.ground_geom in pair
                wall |= self.wall_geom in pair
        return paddle, table, ground, wall

    def _observation(self):
        ball_pos = self.data.qpos[self.ball_qpos:self.ball_qpos + 3]
        ball_vel = self.data.qvel[self.ball_qvel:self.ball_qvel + 3]
        paddle_pos = self.data.geom_xpos[self.paddle_geom]
        obs = np.concatenate((ball_pos, ball_vel, self.data.qpos[self.qpos_ids], self.data.qvel[self.qvel_ids], paddle_pos))
        return obs.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_ids] = READY_POSE
        self.data.ctrl[:] = READY_POSE

        # Randomize the incoming shot mildly, while preserving the demo's
        # paddle-facing direction. This provides useful variation from the start.
        self.data.qpos[self.ball_qpos:self.ball_qpos + 3] = [
            self.np_random.uniform(0.40, 0.47),
            self.np_random.uniform(0.31, 0.39),
            self.np_random.uniform(0.035, 0.07),
        ]
        self.data.qvel[self.ball_qvel:self.ball_qvel + 6] = [
            self.np_random.uniform(-0.5, 0.5),
            self.np_random.uniform(-8.5, -7.5),
            self.np_random.uniform(3.5, 4.5),
            0.0, 0.0, 0.0,
        ]
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self.last_touch_time = None
        self.touch_count = 0
        self._ball_was_touching_paddle = False
        self._ball_was_touching_ground = False
        return self._observation(), {"touches": 0, "rally_seconds": 0.0}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.data.ctrl[:] = self.ctrl_low + (action + 1.0) * 0.5 * (self.ctrl_high - self.ctrl_low)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        paddle, table, ground, wall = self._contacts()
        new_touch = paddle and not self._ball_was_touching_paddle
        new_ground_hit = ground and not self._ball_was_touching_ground
        self._ball_was_touching_paddle = paddle
        self._ball_was_touching_ground = ground

        reward = 0.0
        if new_touch:
            if self.last_touch_time is not None:
                reward = float(self.data.time - self.last_touch_time)
            self.last_touch_time = float(self.data.time)
            self.touch_count += 1

        ball = self.data.qpos[self.ball_qpos:self.ball_qpos + 3]
        out_of_bounds = bool(np.any(np.abs(ball[:2]) > 0.8) or ball[2] > 1.5)
        # A bounce on the green table is part of play. Only the separate
        # default ground plane ends the rally.
        terminated = new_ground_hit or out_of_bounds
        truncated = self.step_count >= self.max_episode_steps
        rally_seconds = 0.0 if self.last_touch_time is None else max(0.0, self.data.time - self.last_touch_time)
        info = {
            "touches": self.touch_count,
            "rally_seconds": rally_seconds,
            "wall_contact": wall,
            "table_contact": table,
            "ground_contact": ground,
            "paddle_touch": new_touch,
        }
        return self._observation(), reward, terminated, truncated, info
