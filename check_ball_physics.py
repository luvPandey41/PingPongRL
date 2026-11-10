"""Compare a free-falling MuJoCo ball with the analytical gravity equations."""

from pathlib import Path

import mujoco
import numpy as np


SCENE_PATH = Path(__file__).parent / "mujoco_menagerie/trs_so_arm100/pingpong_scene.xml"
START_POSITION = np.array([1.5, 1.5, 1.5], dtype=np.float64)
SAMPLE_TIMES = (0.1, 0.2, 0.3)
HEIGHT_TOLERANCE = 0.005  # meters; allows for the scene's 0.002 s Euler steps
VELOCITY_TOLERANCE = 1e-4  # meters/second


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)

    ball_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
    qpos_address = model.jnt_qposadr[ball_joint]
    dof_address = model.jnt_dofadr[ball_joint]

    # Put the ball well away from the table and wall. Starting from rest keeps
    # this a clean test of gravity, without paddle or surface impacts.
    data.qpos[qpos_address:qpos_address + 3] = START_POSITION
    data.qvel[dof_address:dof_address + 6] = 0.0
    data.ctrl[:] = data.qpos[:model.nu]
    mujoco.mj_forward(model, data)

    gravity_z = float(model.opt.gravity[2])
    initial_height = float(data.qpos[qpos_address + 2])
    initial_vertical_velocity = float(data.qvel[dof_address + 2])
    max_height_error = 0.0
    max_velocity_error = 0.0

    print(f"MuJoCo timestep: {model.opt.timestep:.4f} s; gravity z: {gravity_z:.4f} m/s^2")
    print("time(s)  simulated_z(m)  expected_z(m)  z_error(m)  simulated_vz  expected_vz  vz_error")

    steps_per_sample = [round(sample_time / model.opt.timestep) for sample_time in SAMPLE_TIMES]
    previous_step = 0
    for sample_time, target_step in zip(SAMPLE_TIMES, steps_per_sample):
        for _ in range(target_step - previous_step):
            mujoco.mj_step(model, data)
        previous_step = target_step

        elapsed = float(data.time)
        expected_height = initial_height + initial_vertical_velocity * elapsed + 0.5 * gravity_z * elapsed**2
        expected_velocity = initial_vertical_velocity + gravity_z * elapsed
        simulated_height = float(data.qpos[qpos_address + 2])
        simulated_velocity = float(data.qvel[dof_address + 2])
        height_error = simulated_height - expected_height
        velocity_error = simulated_velocity - expected_velocity
        max_height_error = max(max_height_error, abs(height_error))
        max_velocity_error = max(max_velocity_error, abs(velocity_error))

        print(
            f"{elapsed:7.3f}  {simulated_height:14.6f}  {expected_height:12.6f}  "
            f"{height_error:10.6f}  {simulated_velocity:12.6f}  "
            f"{expected_velocity:11.6f}  {velocity_error:9.6f}"
        )

    passed = max_height_error <= HEIGHT_TOLERANCE and max_velocity_error <= VELOCITY_TOLERANCE
    print(
        f"\n{'PASS' if passed else 'CHECK'}: maximum height error "
        f"{max_height_error:.6f} m; maximum velocity error {max_velocity_error:.6f} m/s."
    )
    print("This checks free fall only; it does not validate table, wall, or paddle collision response.")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
