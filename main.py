
import mujoco
import mujoco.viewer
import time

# Path to the SO-ARM100 robot model
model_path = "mujoco_menagerie/trs_so_arm100/pingpong_scene.xml"
# Load the model
model = mujoco.MjModel.from_xml_path(model_path)

# Create simulation data
data = mujoco.MjData(model)

# Hold the arm at the requested starting pose:
# rotation, pitch, elbow, wrist pitch, wrist roll, jaw.
robot_pose = [-1.54, -3.32, 0.34, -0.0332, -1.93, -0.174]
data.qpos[:len(robot_pose)] = robot_pose
data.ctrl[:] = robot_pose

# Start the ball on the wall side of the paddle and send it into the paddle.
# The paddle redirects it toward the wall.
ball_joint = model.joint("ball_joint").id
ball_qpos = model.jnt_qposadr[ball_joint]
ball_dof = model.jnt_dofadr[ball_joint]
data.qpos[ball_qpos:ball_qpos + 3] = [0.437, 0.35, 0.04]
data.qvel[ball_dof:ball_dof + 6] = [0.0, -8.0, 4.0, 0.0, 0.0, 0.0]
mujoco.mj_forward(model, data)

# Print each ball collision once so the paddle and wall hits are easy to verify.
ball_geom = model.geom("ball_geom").id
reported_contacts = set()
playback_speed = 0.1  # Run the simulation at 10% real time so the shot is visible.

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)

        for i in range(data.ncon):
            contact = data.contact[i]
            if ball_geom not in (contact.geom1, contact.geom2):
                continue
            other_geom = contact.geom2 if contact.geom1 == ball_geom else contact.geom1
            other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other_geom)
            if other_name and other_name not in reported_contacts:
                reported_contacts.add(other_name)
                print(f"Ball contact: ball_geom <-> {other_name} at t={data.time:.3f}s")

        viewer.sync()
        step_duration = model.opt.timestep / playback_speed
        time.sleep(max(0.0, step_duration - (time.time() - step_start)))

