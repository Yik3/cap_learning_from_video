# capx/envs/tasks/franka/push_bottle_task.py

from capx.envs.tasks.base import CodeExecutionEnvBase

class FrankaPushBottleTaskEnv(CodeExecutionEnvBase):
    """Push the bottle over so it lies flat on the table."""

    prompt = """
    You are controlling a Franka Panda robot.
    A bottle is standing upright on the table.
    Your goal is to push the bottle over so that it falls on its side. Do not pick it up.

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

# 1. 张开夹爪 (用拳头去推)
env.open_gripper()

# 2. 移动到水瓶的右侧
side_pos = np.array(bottle_pose[:3])
side_pos[1] -= 0.15  # 在水瓶右侧 15cm 处
side_pos[2] += 0.08  # 抬高 8cm，击打真实的瓶身上半部分

env.move_to_pose(side_pos, ee_pose[3:])

# 3. 水平向左移动，撞击并推倒水瓶
push_pos = side_pos.copy()
push_pos[1] += 0.20  # 向左平移 20cm 穿过水瓶
env.move_to_pose(push_pos, ee_pose[3:])

# 4. 回退到安全位置
env.move_to_pose(side_pos, ee_pose[3:])
"""