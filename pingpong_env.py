"""Gymnasium environment for learning sustained SO-ARM100 ping-pong rallies."""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


SCENE_PATH = Path(__file__).parent / "mujoco_menagerie/trs_so_arm100/pingpong_scene.xml"
JOINT_NAMES = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
CONTROLLED_JOINT_NAMES = JOINT_NAMES[:-1]
READY_POSE = np.array([-1.54, -3.32, 0.34, -0.0332, -1.93, -0.174], dtype=np.float64)
PADDLE_CONTACT_COOLDOWN = 0.05  # seconds; suppress contact chatter while dragging
FUTURE_POSITION_REWARD_SCALE = 1.0
FUTURE_POSITION_REWARD_WIDTH = 0.30  # meters
GUIDANCE_JOINT_STEP_LIMIT = 0.12  # radians per policy step for the IK teacher
GUIDANCE_ORIENTATION_WEIGHT = 0.25  # position-equivalent error per radian while approaching
GUIDANCE_WRIST_ROLL_NORMAL_GAIN = 1.0  # direct normal-alignment correction for the paddle face
GUIDANCE_DESIRED_OUTGOING_VELOCITY = np.array([0.0, 4.0, 3.0], dtype=np.float64)
APPROACH_SHAPING_DISCOUNT = 0.9995  # keep aligned with PPO's GAMMA
INTERCEPT_MAX_TIME = 0.80  # allow an outbound ball to hit the wall and return
INTERCEPT_DISTANCE_SCALE = 0.75  # meters; sets how quickly distance potential changes
TABLE_RESTITUTION_ESTIMATE = 0.53
WALL_RESTITUTION_ESTIMATE = 0.53
TABLE_TANGENTIAL_RETENTION = 0.98
PREDICTION_MAX_EVENTS = 8


class PingPongEnv(gym.Env):
    """One episode is one ball's attempt to sustain a self-rally.

    Reward is solely a strong proximity score between the paddle and the
    predicted future ball position.
    """

    metadata = {"render_modes": []}

    def __init__(self, frame_skip: int = 5, max_episode_seconds: float = 8.0):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.capture_substeps = False
        self.last_substep_states = []
        self.max_episode_steps = int(max_episode_seconds / (self.model.opt.timestep * frame_skip))
        self.step_count = 0

        self.joint_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES])
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        self.qvel_ids = self.model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINT_NAMES
        ])
        self.controlled_actuator_ids = self.actuator_ids[:-1]
        self.jaw_actuator_id = int(self.actuator_ids[-1])
        self.ctrl_low = self.model.actuator_ctrlrange[self.controlled_actuator_ids, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[self.controlled_actuator_ids, 1].copy()
        self.jaw_closed_position = float(READY_POSE[-1])
        self.ball_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        self.ball_qpos = self.model.jnt_qposadr[self.ball_joint_id]
        self.ball_qvel = self.model.jnt_dofadr[self.ball_joint_id]
        self.ball_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self.paddle_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "pingpong_paddle")
        self.paddle_handle_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "paddle_handle"
        )
        self.table_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "pingpong_floor")
        self.ground_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.wall_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "pingpong_wall")

        # Joint state plus paddle-to-future-intercept distance/direction,
        # time-to-intercept, and validity. Absolute ball/paddle xyz is hidden.
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(18,), dtype=np.float32)
        # The jaw remains in the observation, but the policy has no jaw action.
        self.action_space = spaces.Box(-1.0, 1.0, shape=(len(CONTROLLED_JOINT_NAMES),), dtype=np.float32)
        self.last_touch_time = None
        self.touch_count = 0
        self.rally_return_count = 0
        self.paddle_contact_count = 0
        self._second_return_practice = False
        self.second_return_practice_probability = 0.5
        self._first_return_state_bank = []
        self.guidance_strength = 0.0
        self._ball_was_touching_paddle = False
        self._ball_was_touching_ground = False
        self._last_paddle_event_time = None
        self._previous_intercept_potential = 0.0
        self._previous_intercept_valid = False
        self._previous_ball_alignment_potential = 0.0
        self.curriculum_stage = 0
        calibration_data = mujoco.MjData(self.model)
        calibration_data.qpos[self.qpos_ids] = READY_POSE
        mujoco.mj_forward(self.model, calibration_data)
        self._intercept_plane_y = float(calibration_data.geom_xpos[self.paddle_geom, 1])
        self._ready_paddle_normal = calibration_data.geom_xmat[
            self.paddle_geom
        ].reshape(3, 3)[:, 2].copy()
        self._guided_target_normal = self._ready_paddle_normal.copy()
        self._intercept_prediction = (0.0, 0.0, 0.0, False)
        self._predicted_intercept_velocity = np.zeros(3, dtype=np.float64)
        self._intercept_debug = {"reason": "not_evaluated"}

    def _intercept_potential(self) -> float:
        """Return a bounded potential for reaching the predicted contact point."""
        target_x, target_z, _, valid = self._intercept_prediction
        if not valid:
            return 0.0
        target = np.array([target_x, self._intercept_plane_y, target_z])
        paddle_pos = self.data.geom_xpos[self.paddle_geom]
        distance = float(np.linalg.norm(target - paddle_pos))
        # Smooth potential keeps a gradient at every distance, including when
        # the paddle starts far from the predicted contact point.
        return -float(np.tanh(distance / INTERCEPT_DISTANCE_SCALE))

    def _predicted_target_reward(self) -> float:
        """Return the sole reward: strong proximity to the forecast ball position."""
        target_x, target_z, _, valid = self._intercept_prediction
        if not valid:
            return 0.0
        target = np.array([target_x, self._intercept_plane_y, target_z])
        paddle_pos = self.data.geom_xpos[self.paddle_geom]
        distance = float(np.linalg.norm(target - paddle_pos))
        # Exponential distance keeps a useful gradient while the paddle is
        # still far from the target; a squared-distance Gaussian is nearly
        # flat there and can leave a fresh policy with no learning signal.
        return float(FUTURE_POSITION_REWARD_SCALE * np.exp(-distance / FUTURE_POSITION_REWARD_WIDTH))

    def _ball_alignment_potential(self) -> float:
        """Measure current horizontal/vertical alignment for an inbound ball."""
        if not self._intercept_prediction[3]:
            return 0.0
        ball_pos = self.data.qpos[self.ball_qpos:self.ball_qpos + 3]
        paddle_pos = self.data.geom_xpos[self.paddle_geom]
        lateral_error = ball_pos[[0, 2]] - paddle_pos[[0, 2]]
        return -float(np.tanh(np.linalg.norm(lateral_error) / 0.25))

    def _guided_intercept_action(self) -> np.ndarray | None:
        """Map the predicted target to a bounded joint-target action using IK."""
        target_x, target_z, _, valid = self._intercept_prediction
        if not valid:
            return None

        target = np.array([target_x, self._intercept_plane_y, target_z])
        paddle_pos = self.data.geom_xpos[self.paddle_geom]
        position_error = target - paddle_pos
        orientation_weight = (
            1.0 if float(np.linalg.norm(position_error)) < 0.10
            else GUIDANCE_ORIENTATION_WEIGHT
        )
        jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacGeom(
            self.model,
            self.data,
            jacobian_position,
            jacobian_rotation,
            self.paddle_geom,
        )
        joint_jacobian_position = jacobian_position[:, self.qvel_ids[:-1]]
        joint_jacobian_rotation = jacobian_rotation[:, self.qvel_ids[:-1]]
        current_normal = self.data.geom_xmat[self.paddle_geom].reshape(3, 3)[:, 2]
        normal_hint = self._predicted_intercept_velocity - GUIDANCE_DESIRED_OUTGOING_VELOCITY
        normal_norm = float(np.linalg.norm(normal_hint))
        target_normal = (
            normal_hint / normal_norm if normal_norm > 1e-8 else self._ready_paddle_normal
        )
        if np.dot(current_normal, target_normal) < 0.0:
            target_normal = -target_normal
        self._guided_target_normal = target_normal.copy()
        cross = np.cross(current_normal, target_normal)
        cross_norm = float(np.linalg.norm(cross))
        normal_dot = float(np.clip(np.dot(current_normal, target_normal), -1.0, 1.0))
        orientation_error = (
            cross / cross_norm * np.arctan2(cross_norm, normal_dot)
            if cross_norm > 1e-8 else np.zeros(3)
        )
        joint_jacobian = np.vstack((
            joint_jacobian_position,
            orientation_weight * joint_jacobian_rotation,
        ))
        task_error = np.concatenate((
            position_error,
            orientation_weight * orientation_error,
        ))
        damping = 1e-3
        task_matrix = joint_jacobian @ joint_jacobian.T + damping**2 * np.eye(6)
        joint_delta = joint_jacobian.T @ np.linalg.solve(task_matrix, task_error)
        joint_delta = np.clip(joint_delta, -GUIDANCE_JOINT_STEP_LIMIT, GUIDANCE_JOINT_STEP_LIMIT)

        current_joint_targets = self.data.qpos[self.qpos_ids[:-1]]
        guided_joint_targets = current_joint_targets + joint_delta
        guided_joint_targets[4] += GUIDANCE_WRIST_ROLL_NORMAL_GAIN * (
            target_normal[2] - self._ready_paddle_normal[2]
        )
        guided_action = 2.0 * (
            (guided_joint_targets - self.ctrl_low) / (self.ctrl_high - self.ctrl_low)
        ) - 1.0
        return np.clip(guided_action, -1.0, 1.0)

    def _predict_intercept(self) -> tuple[float, float, float, bool]:
        """Predict the next inbound crossing of a fixed plane using simple bounces."""
        self._predicted_intercept_velocity[:] = 0.0
        ball_pos = self.data.qpos[self.ball_qpos:self.ball_qpos + 3]
        ball_vel = self.data.qvel[self.ball_qvel:self.ball_qvel + 3]

        def reject(reason: str, **details) -> tuple[float, float, float, bool]:
            self._intercept_debug = {
                "reason": reason,
                "ball_position": ball_pos.copy(),
                "ball_velocity": ball_vel.copy(),
                **details,
            }
            return 0.0, 0.0, 0.0, False

        position = ball_pos.copy()
        velocity = ball_vel.copy()
        gravity_z = float(self.model.opt.gravity[2])
        table_top = float(
            self.model.geom_pos[self.table_geom, 2] + self.model.geom_size[self.table_geom, 2]
        )
        ball_radius = float(self.model.geom_size[self.ball_geom, 0])
        table_center = self.model.geom_pos[self.table_geom]
        table_half_size = self.model.geom_size[self.table_geom]
        wall_center = self.model.geom_pos[self.wall_geom]
        wall_half_size = self.model.geom_size[self.wall_geom]
        table_ball_center_z = table_top + ball_radius
        wall_ball_center_y = wall_center[1] - wall_half_size[1] - ball_radius
        elapsed = 0.0

        # Advance analytically between collision events. Reflection estimates
        # are intentionally simple; the live MuJoCo simulation remains the
        # source of truth, and this prediction is only an observation hint.
        for _ in range(PREDICTION_MAX_EVENTS):
            time_to_plane = np.inf
            if velocity[1] < -1e-6 and position[1] > self._intercept_plane_y:
                time_to_plane = (self._intercept_plane_y - position[1]) / velocity[1]

            time_to_wall = np.inf
            if velocity[1] > 1e-6 and position[1] < wall_ball_center_y:
                time_to_wall = (wall_ball_center_y - position[1]) / velocity[1]

            time_to_table = np.inf
            if velocity[2] < 0.0 and position[2] > table_ball_center_z:
                discriminant = velocity[2] ** 2 - 2.0 * gravity_z * (
                    position[2] - table_ball_center_z
                )
                if discriminant >= 0.0:
                    root = np.sqrt(discriminant)
                    candidates = (
                        (-velocity[2] - root) / gravity_z,
                        (-velocity[2] + root) / gravity_z,
                    )
                    positive_descending = [
                        candidate for candidate in candidates
                        if candidate > 1e-6 and velocity[2] + gravity_z * candidate < 0.0
                    ]
                    if positive_descending:
                        time_to_table = min(positive_descending)

            next_event_time = min(time_to_plane, time_to_wall, time_to_table)
            if not np.isfinite(next_event_time) or elapsed + next_event_time > INTERCEPT_MAX_TIME:
                return reject(
                    "no predicted collision/crossing within horizon",
                    elapsed=elapsed,
                    next_event_time=float(next_event_time),
                    horizon=INTERCEPT_MAX_TIME,
                )

            position += velocity * next_event_time
            position[2] += 0.5 * gravity_z * next_event_time**2
            velocity[2] += gravity_z * next_event_time
            elapsed += next_event_time

            if time_to_plane <= time_to_wall and time_to_plane <= time_to_table:
                inside_table_x = abs(position[0] - table_center[0]) <= table_half_size[0] - ball_radius
                above_table = position[2] >= table_ball_center_z + 0.005
                if inside_table_x and above_table:
                    self._predicted_intercept_velocity[:] = velocity
                    self._intercept_debug = {
                        "reason": "valid",
                        "ball_position": ball_pos.copy(),
                        "ball_velocity": ball_vel.copy(),
                        "predicted_velocity": velocity.copy(),
                        "crossing_position": position.copy(),
                    }
                    return float(position[0]), float(position[2]), float(elapsed), True
                reason = "crossing outside table width" if not inside_table_x else "crossing too low"
                return reject(
                    reason,
                    crossing_position=position.copy(),
                    crossing_time=float(elapsed),
                    inside_table_x=bool(inside_table_x),
                    above_table=bool(above_table),
                )

            if time_to_wall <= time_to_table:
                wall_x_ok = abs(position[0] - wall_center[0]) <= wall_half_size[0] + ball_radius
                wall_z_ok = (
                    wall_center[2] - wall_half_size[2] - ball_radius
                    <= position[2]
                    <= wall_center[2] + wall_half_size[2] + ball_radius
                )
                if not (wall_x_ok and wall_z_ok):
                    return reject(
                        "predicted wall hit misses wall",
                        wall_position=position.copy(),
                        wall_x_ok=bool(wall_x_ok),
                        wall_z_ok=bool(wall_z_ok),
                        elapsed=float(elapsed),
                    )
                position[1] = wall_ball_center_y
                velocity[1] = -velocity[1] * WALL_RESTITUTION_ESTIMATE
                continue

            inside_table = (
                abs(position[0] - table_center[0]) <= table_half_size[0] - ball_radius
                and abs(position[1] - table_center[1]) <= table_half_size[1] - ball_radius
            )
            if not inside_table:
                return reject(
                    "predicted table hit misses table",
                    table_position=position.copy(),
                    elapsed=float(elapsed),
                )
            position[2] = table_ball_center_z
            velocity[2] = -velocity[2] * TABLE_RESTITUTION_ESTIMATE
            velocity[:2] *= TABLE_TANGENTIAL_RETENTION

        return reject("prediction event limit reached", events=PREDICTION_MAX_EVENTS, elapsed=float(elapsed))

    def refresh_intercept_prediction(self) -> tuple[float, float, float, bool]:
        """Refresh and return the predicted target for the current physics state."""
        self._intercept_prediction = self._predict_intercept()
        return self._intercept_prediction

    def _contacts(self):
        paddle = table = ground = wall = illegal_robot = False
        illegal_robot_geoms = set()
        allowed_other_geoms = {
            self.paddle_geom,
            self.paddle_handle_geom,
            self.table_geom,
            self.ground_geom,
            self.wall_geom,
        }
        allowed_other_geoms.discard(-1)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if self.ball_geom in pair:
                other_geom = c.geom2 if c.geom1 == self.ball_geom else c.geom1
                paddle |= self.paddle_geom in pair
                table |= self.table_geom in pair
                ground |= self.ground_geom in pair
                wall |= self.wall_geom in pair
                illegal_robot |= other_geom not in allowed_other_geoms
                if other_geom not in allowed_other_geoms:
                    geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom)
                    body_id = int(self.model.geom_bodyid[other_geom])
                    body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                    label = geom_name or f"{body_name or 'unnamed'}_geom_id_{other_geom}"
                    illegal_robot_geoms.add(label)
        return paddle, table, ground, wall, illegal_robot, sorted(illegal_robot_geoms)

    def _observation(self):
        paddle_pos = self.data.geom_xpos[self.paddle_geom]
        target_x, target_z, time_to_intercept, target_valid = self._intercept_prediction
        target_offset = np.zeros(3, dtype=np.float64)
        target_distance = 0.0
        target_direction = np.zeros(3, dtype=np.float64)
        if target_valid:
            target_position = np.array([target_x, self._intercept_plane_y, target_z])
            target_offset = target_position - paddle_pos
            target_distance = float(np.linalg.norm(target_offset))
            if target_distance > 1e-8:
                target_direction = target_offset / target_distance

        # The policy sees how far and in which direction the paddle must move
        # to reach the predicted intercept, without absolute ball/paddle xyz.
        joint_position_scale = np.maximum(np.abs(self.ctrl_low), np.abs(self.ctrl_high))
        obs = np.concatenate((
            self.data.qpos[self.qpos_ids] / np.append(joint_position_scale, max(abs(self.jaw_closed_position), 1.0)),
            self.data.qvel[self.qvel_ids] / 10.0,
            np.array([target_distance], dtype=np.float64),
            target_direction,
            np.array([time_to_intercept / INTERCEPT_MAX_TIME, float(target_valid)]),
        ))
        return obs.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_ids] = READY_POSE
        self.data.ctrl[self.actuator_ids] = READY_POSE
        self.data.ctrl[self.jaw_actuator_id] = self.jaw_closed_position

        # Once first returns are learned, spend half of stage-1/2 episodes
        # practicing recovery from a real saved post-return state.
        self._second_return_practice = bool(
            self.curriculum_stage >= 1
            and self._first_return_state_bank
            and self.np_random.random() < self.second_return_practice_probability
        )
        if self._second_return_practice:
            saved_state = self._first_return_state_bank[
                int(self.np_random.integers(len(self._first_return_state_bank)))
            ]
            self.data.qpos[:] = saved_state["qpos"]
            self.data.qvel[:] = saved_state["qvel"]
            self.data.ctrl[:] = saved_state["ctrl"]
        # Otherwise sample a curriculum shot that gives the policy a valid
        # intercept target. Random cases that would miss the table entirely
        # provide no target-position signal and are resampled.
        self._shot_resamples = 0
        if not self._second_return_practice:
            accepted_shot = False
            max_shot_attempts = 50
            for attempt in range(max_shot_attempts):
                if self.curriculum_stage == 0:
                    ball_position = [0.435, 0.35, 0.05]
                    ball_velocity = [0.0, -8.0, 4.0]
                elif self.curriculum_stage == 1:
                    ball_position = [
                        self.np_random.uniform(0.40, 0.47),
                        self.np_random.uniform(0.31, 0.39),
                        self.np_random.uniform(0.035, 0.07),
                    ]
                    ball_velocity = [
                        self.np_random.uniform(-0.5, 0.5),
                        self.np_random.uniform(-8.5, -7.5),
                        self.np_random.uniform(3.5, 4.5),
                    ]
                else:
                    ball_position = [
                        self.np_random.uniform(0.37, 0.50),
                        self.np_random.uniform(0.29, 0.43),
                        self.np_random.uniform(0.03, 0.09),
                    ]
                    ball_velocity = [
                        self.np_random.uniform(-1.0, 1.0),
                        self.np_random.uniform(-9.0, -7.0),
                        self.np_random.uniform(3.3, 4.7),
                    ]

                self.data.qpos[self.ball_qpos:self.ball_qpos + 3] = ball_position
                self.data.qvel[self.ball_qvel:self.ball_qvel + 6] = [
                    *ball_velocity, 0.0, 0.0, 0.0
                ]
                mujoco.mj_forward(self.model, self.data)
                self._shot_resamples = attempt
                if self._predict_intercept()[3]:
                    accepted_shot = True
                    break

            if not accepted_shot:
                # The original stage-0 shot is known to have a valid intercept.
                self.data.qpos[self.ball_qpos:self.ball_qpos + 3] = [0.435, 0.35, 0.05]
                self.data.qvel[self.ball_qvel:self.ball_qvel + 6] = [0.0, -8.0, 4.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)
        self.refresh_intercept_prediction()
        self.step_count = 0
        self.last_touch_time = None
        self.touch_count = 0
        self.rally_return_count = int(self._second_return_practice)
        self.paddle_contact_count = 0
        initial_paddle_contact, _, initial_ground_contact, _, _, _ = self._contacts()
        self._ball_was_touching_paddle = initial_paddle_contact
        self._ball_was_touching_ground = initial_ground_contact
        self._last_paddle_event_time = float(self.data.time) if initial_paddle_contact else None
        self._previous_intercept_valid = self._intercept_prediction[3]
        self._previous_intercept_potential = self._intercept_potential()
        self._previous_ball_alignment_potential = self._ball_alignment_potential()
        return self._observation(), {
            "touches": 0,
            "rally_seconds": 0.0,
            "second_return_practice": self._second_return_practice,
            "shot_resamples": self._shot_resamples,
        }

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        prediction_for_action = self._intercept_prediction
        predicted_velocity_for_action = self._predicted_intercept_velocity.copy()
        applied_guidance_strength = 0.0
        if self.guidance_strength > 0.0:
            guided_action = self._guided_intercept_action()
            if guided_action is not None:
                strength = float(np.clip(self.guidance_strength, 0.0, 1.0))
                action = (1.0 - strength) * action + strength * guided_action
                applied_guidance_strength = strength
        self.data.ctrl[self.controlled_actuator_ids] = (
            self.ctrl_low + (action + 1.0) * 0.5 * (self.ctrl_high - self.ctrl_low)
        )
        self.data.ctrl[self.jaw_actuator_id] = self.jaw_closed_position
        self.step_count += 1
        reward = 0.0
        approach_reward = 0.0
        predicted_target_reward = 0.0
        ball_alignment_reward = 0.0
        contact_reward = 0.0
        contact_quality_reward = 0.0
        return_reward = 0.0
        milestone_reward = 0.0
        new_touch_count = 0
        new_ground_hit = False
        illegal_robot_contact = False
        illegal_robot_contact_geoms = set()
        invalid_paddle_hit = False
        paddle_touch_times = []
        paddle_contact_events = []
        table_contact = ground_contact = wall_contact = False
        self.last_substep_states.clear()

        # Inspect every physics substep. A brief collision can begin and end
        # between policy actions, so checking only after frame_skip can miss it.
        for _ in range(self.frame_skip):
            incoming_ball_velocity = self.data.qvel[self.ball_qvel:self.ball_qvel + 3].copy()
            mujoco.mj_step(self.model, self.data)
            paddle, table, ground, wall, illegal_robot, illegal_geoms = self._contacts()
            if self.capture_substeps:
                self.last_substep_states.append(
                    (self.data.qpos.copy(), self.data.qvel.copy(), float(self.data.time))
                )

            table_contact |= table
            ground_contact |= ground
            wall_contact |= wall
            illegal_robot_contact |= illegal_robot
            illegal_robot_contact_geoms.update(illegal_geoms)

            if paddle and not self._ball_was_touching_paddle:
                event_time = float(self.data.time)
                cooldown_passed = (
                    self._last_paddle_event_time is None
                    or event_time - self._last_paddle_event_time >= PADDLE_CONTACT_COOLDOWN
                )
                if cooldown_passed:
                    self._last_paddle_event_time = event_time
                    self.paddle_contact_count += 1
                    contact_vy = float(self.data.qvel[self.ball_qvel + 1])
                    contact_vz = float(self.data.qvel[self.ball_qvel + 2])
                    paddle_contact_events.append({
                        "contact_number": self.paddle_contact_count,
                        "contact_time": event_time,
                        "rally_return_index": 0,
                        "contact_vy": contact_vy,
                        "contact_vz": contact_vz,
                        "incoming_velocity": incoming_ball_velocity.copy(),
                        "outgoing_velocity": self.data.qvel[
                            self.ball_qvel:self.ball_qvel + 3
                        ].copy(),
                        "ball_vy": contact_vy,
                        "ball_vz": contact_vz,
                        "incoming_vy": float(incoming_ball_velocity[1]),
                        "incoming_vz": float(incoming_ball_velocity[2]),
                        "predicted_incoming_velocity": predicted_velocity_for_action.copy(),
                        "guided_target_normal": self._guided_target_normal.copy(),
                        "paddle_normal": self.data.geom_xmat[
                            self.paddle_geom
                        ].reshape(3, 3)[:, 2].copy(),
                        "joint_positions": self.data.qpos[self.qpos_ids[:-1]].copy(),
                        "valid_return": False,
                        "quality_reward": 0.0,
                        "guidance_strength": applied_guidance_strength,
                        "prediction_valid": bool(prediction_for_action[3]),
                        "predicted_x": float(prediction_for_action[0]),
                        "predicted_z": float(prediction_for_action[1]),
                        "contact_x": float(self.data.qpos[self.ball_qpos]),
                        "contact_z": float(self.data.qpos[self.ball_qpos + 2]),
                        "target_error_x": (
                            float(self.data.qpos[self.ball_qpos] - prediction_for_action[0])
                            if prediction_for_action[3] else np.nan
                        ),
                        "target_error_z": (
                            float(self.data.qpos[self.ball_qpos + 2] - prediction_for_action[1])
                            if prediction_for_action[3] else np.nan
                        ),
                    })

            if ground and not self._ball_was_touching_ground:
                new_ground_hit = True

            # Keep the edge detectors in sync with the most recent physics
            # substep, so sustained contact counts once and a later hit counts again.
            self._ball_was_touching_paddle = paddle
            self._ball_was_touching_ground = ground

        # Let all physics substeps in this control frame resolve the paddle
        # impulse before deciding whether the outgoing shot is valid.
        settled_vy = float(self.data.qvel[self.ball_qvel + 1])
        settled_vz = float(self.data.qvel[self.ball_qvel + 2])
        for event in paddle_contact_events:
            event["rally_return_index"] = self.rally_return_count + 1
            event["ball_vy"] = settled_vy
            event["ball_vz"] = settled_vz
            event["ball_vx"] = float(self.data.qvel[self.ball_qvel])
            event["outgoing_velocity"] = self.data.qvel[
                self.ball_qvel:self.ball_qvel + 3
            ].copy()
            valid_return = True
            event["valid_return"] = valid_return
            if valid_return:
                new_touch_count += 1
                paddle_touch_times.append(float(event["contact_time"]))
                self.touch_count += 1
                self.rally_return_count += 1
            else:
                invalid_paddle_hit = False

        for touch_time in paddle_touch_times:
            self.last_touch_time = touch_time

        if self.rally_return_count == 1 and new_touch_count:
            self._first_return_state_bank.append({
                "qpos": self.data.qpos.copy(),
                "qvel": self.data.qvel.copy(),
                "ctrl": self.data.ctrl.copy(),
            })
            if len(self._first_return_state_bank) > 256:
                self._first_return_state_bank.pop(0)

        ball = self.data.qpos[self.ball_qpos:self.ball_qpos + 3]
        out_of_bounds = bool(np.any(np.abs(ball[:2]) > 0.8) or ball[2] > 1.5)
        # A bounce on the green table is part of play. Only the separate
        # default ground plane ends the rally.
        terminated = new_ground_hit or illegal_robot_contact or invalid_paddle_hit or out_of_bounds
        truncated = self.step_count >= self.max_episode_steps
        self.refresh_intercept_prediction()
        current_intercept_valid = self._intercept_prediction[3]
        current_intercept_potential = self._intercept_potential()
        # Score distance to the next intercept after this action's physics
        # steps. If the ball hit the paddle, the updated target reflects the
        # resulting shot and rewards positioning for the next return.
        predicted_target_reward = self._predicted_target_reward()
        reward = predicted_target_reward
        current_ball_alignment = self._ball_alignment_potential()
        self._previous_intercept_valid = current_intercept_valid
        self._previous_intercept_potential = current_intercept_potential
        self._previous_ball_alignment_potential = current_ball_alignment
        rally_seconds = 0.0 if self.last_touch_time is None else max(0.0, self.data.time - self.last_touch_time)
        info = {
            "touches": self.touch_count,
            "rally_returns": self.rally_return_count,
            "second_return_practice": self._second_return_practice,
            "guidance_strength": applied_guidance_strength,
            "rally_seconds": rally_seconds,
            "approach_reward": approach_reward,
            "predicted_target_reward": predicted_target_reward,
            "ball_alignment_reward": ball_alignment_reward,
            "contact_reward": contact_reward,
            "contact_quality_reward": contact_quality_reward,
            "paddle_contact_events": paddle_contact_events,
            "return_reward": return_reward,
            "milestone_reward": milestone_reward,
            "predicted_intercept_valid": current_intercept_valid,
            "predicted_intercept_x": self._intercept_prediction[0],
            "predicted_intercept_z": self._intercept_prediction[1],
            "time_to_intercept": self._intercept_prediction[2],
            "wall_contact": wall_contact,
            "table_contact": table_contact,
            "ground_contact": ground_contact,
            "illegal_robot_contact": illegal_robot_contact,
            "illegal_robot_contact_geoms": sorted(illegal_robot_contact_geoms),
            "invalid_paddle_hit": invalid_paddle_hit,
            "out_of_bounds": out_of_bounds,
            "termination_reason": self._termination_reason(
                new_ground_hit, illegal_robot_contact, out_of_bounds, truncated
            ),
            "reward_target_valid": bool(current_intercept_valid),
            "reward_target_x": float(self._intercept_prediction[0]),
            "reward_target_z": float(self._intercept_prediction[1]),
            "paddle_touch": new_touch_count > 0,
        }
        return self._observation(), reward, terminated, truncated, info

    @staticmethod
    def _termination_reason(ground_hit, illegal_robot_contact, out_of_bounds, truncated):
        reasons = []
        if ground_hit:
            reasons.append("ground")
        if illegal_robot_contact:
            reasons.append("illegal_robot_contact")
        if out_of_bounds:
            reasons.append("out_of_bounds")
        if truncated:
            reasons.append("time_limit")
        return "+".join(reasons) if reasons else "running"
