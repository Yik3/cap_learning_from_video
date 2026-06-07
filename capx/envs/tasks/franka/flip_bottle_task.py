from capx.envs.tasks.base import CodeExecutionEnvBase

class FrankaFlipBottleTaskEnv(CodeExecutionEnvBase):
    """Flip the bottle 180 degrees using a horizontal side-grasp and wrist roll."""

    prompt = """
    You are controlling a Franka Panda robot.
    A bottle is standing inverted on the table.
    Your goal is to flip the bottle 180 degrees upright.

    [Helpful APIs]
    - pos, quat, bbox = get_object_pose(obj_name, return_bbox_extent=True) # Returns pos [x,y,z], quat [x,y,z,w]
    - goto_pose(position, quaternion_wxyz) # IMPORTANT: requires wxyz quaternion format!
    - open_gripper()
    - close_gripper()
    - home_pose()
    - You MUST use these exact import statements at the top of your code:
      import numpy as np
      from scipy.spatial.transform import Rotation as R
    
    [CONSTRAINTS & STRATEGY]
    To achieve this efficiently without kinematic singularities or collisions, reference this:
    To achieve this efficiently with the robot's kinematics, follow these steps:
1. Find a pre-grasp position at a horizontal distance from the bottle (e.g., offset by -0.15 on the Y-axis). You can sample several waypoints to avoid any possible collision with the bottle.
2. Orient the gripper horizontally so it approaches from the side at the bottle's half-height.
3. Move horizontally to grasp the bottle.
4. Lift the bottle straight up to clear the table.
5. Rotate the wrist joint 180 degrees (roll around your approach axis, e.g., the Y-axis) to flip the bottle upright.
6. Lower the bottle gently back onto the table.
7. Open the gripper and retreat horizontally.

    [Tips for reference [you do not have to strictly follow these, but they are a good reference for an efficient solution]:]
    1. Grasp Orientation: We can approach horizontally from the -Y axis.:
       - The gripper's approach axis (Z_grip) points to the world's +Y direction [0, 1, 0].
       - The finger opening direction (Y_grip) points to the world's +Z direction [0, 0, 1].
       - The orthogonal X_grip points to the world's +X direction [1, 0, 0].
    2. Safe Approach (L-Shape): Do not move directly to the pre-grasp. First, calculate a high safe waypoint (offset target by -0.15 in Y, +0.15 in Z). Then drop straight down to the pre-grasp waypoint (offset target by -0.15 in Y).
    3. Grasp: Move horizontally to the bottle's position and `close_gripper()`.
    4. Lift: Lift the bottle straight up (offset by +0.25 in Z).
    5. Flip: Calculate the flipped orientation by applying a 180-degree (np.pi) rotation around the gripper's LOCAL approach axis (local Z-axis) to the original grasp rotation. `goto_pose` to execute the flip.
    6. Place: Lower the bottle back down. Add a strict +0.01m Z-offset to the placement position to ensure it rests gently without colliding with the physics engine's table mesh.
    7. Safe Retreat: `open_gripper()`. Retreat horizontally (-0.15 in Y), move back up to a high safe position (-0.15 in Y, +0.15 in Z), and finally call `home_pose()`.

    Note: Always convert scipy's default `xyzw` quaternions to `wxyz` before passing them to `goto_pose`.
    Write ONLY executable Python code. Do not write it in code fences.
    """

    oracle_code = """
import numpy as np
import viser.transforms as vtf

obs = env.get_observation()
bottle_pose = obs["object_poses"]["bottle"]
ee_pose = obs["robot_cartesian_pos"]
original_orientation = ee_pose[3:]

# 1. Calculate grasp position (assuming bottle_pose[:3] is the center of the bottle)
grasp_pos = np.array(bottle_pose[:3])

# 2. Compute horizontal gripper orientation
# Rotate default downward orientation (-90 deg around X) to point horizontally along +Y
base_rot = vtf.SO3(wxyz=original_orientation)
horizontal_rot = vtf.SO3.from_x_radians(-np.pi / 2) @ base_rot

# 3. Move to pre-grasp position (offset along -Y)
pre_grasp = grasp_pos.copy()
pre_grasp[1] -= 0.15
env.move_to_pose(pre_grasp, horizontal_rot.wxyz)

# 4. Move horizontally to grasp
env.move_to_pose(grasp_pos, horizontal_rot.wxyz)
env.close_gripper()

# 5. Lift the bottle
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.20
env.move_to_pose(lift_pos, horizontal_rot.wxyz)

# 6. Rotate the wrist 180 degrees around the approach axis (+Y axis)
# This flips the bottle upright by purely twisting the wrist joint
flip_rot = vtf.SO3.from_y_radians(np.pi) @ horizontal_rot
env.move_to_pose(lift_pos, flip_rot.wxyz)

# 7. Lower gently onto the table
place_pos = lift_pos.copy()
place_pos[2] -= 0.18 # Lower back down carefully
env.move_to_pose(place_pos, flip_rot.wxyz)

# 8. Open gripper and retreat horizontally
env.open_gripper()
retreat_pos = place_pos.copy()
retreat_pos[1] -= 0.15
env.move_to_pose(retreat_pos, flip_rot.wxyz)
"""

