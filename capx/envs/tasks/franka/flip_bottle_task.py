# capx/envs/tasks/franka/flip_bottle_task.py

from capx.envs.tasks.base import CodeExecutionEnvBase

class FrankaFlipBottleTaskEnv(CodeExecutionEnvBase):
    """Flip the bottle 360 degrees. Half reward for 180 degrees flip."""

    prompt = """
    You are controlling a Franka Panda robot.
    A bottle is standing upright on the table.
    Your goal is to flip the bottle 360 degrees. You will get half reward if you flip it 180 degrees.
    You can flip it by pushing it from the side or by grasping and rotating it.

    You can get the positions using:
    obs = env.get_observation()
    bottle_pose = obs["object_poses"]["bottle"] # [x, y, z, w, x, y, z]

    Write ONLY executable Python code. No code fences.
    """

    oracle_code = """
import numpy as np

obs = env.get_observation()
bottle_pose = obs["object_poses"]["bottle"]
ee_pose = obs["robot_cartesian_pos"]

# 1. Move to grasp the bottle from the side
side_pos = np.array(bottle_pose[:3])
side_pos[1] -= 0.15  # Approach from the side
side_pos[2] += 0.05  # Slightly above the middle

env.move_to_pose(side_pos, ee_pose[3:])

# 2. Close gripper to grasp the bottle
env.close_gripper()

# 3. Lift the bottle slightly
lift_pos = side_pos.copy()
lift_pos[2] += 0.10  # Lift up
env.move_to_pose(lift_pos, ee_pose[3:])

# 4. Rotate the bottle 360 degrees by moving in a circular arc
# First rotation - push the top away to flip it
flip_pos = lift_pos.copy()
flip_pos[2] -= 0.15  # Move down and push to flip
env.move_to_pose(flip_pos, ee_pose[3:])

# 5. Complete the rotation by continuing the motion
flip_pos[2] += 0.20  # Move back up to complete the flip
env.move_to_pose(flip_pos, ee_pose[3:])

# 6. Return to safe position
env.move_to_pose(lift_pos, ee_pose[3:])

# 7. Open gripper
env.open_gripper()
"""
