# capx/envs/simulators/push_bottle_sim.py

import numpy as np
import robosuite as suite
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena            # 显式导入桌面场景
from robosuite.models.objects import BottleObject         # 原生水瓶模型
from robosuite.models.tasks import ManipulationTask       # 任务打包器
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.controllers import load_composite_controller_config
import viser.transforms as vtf

from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv

# =================================================================
# 1. 定义自定义的 Robosuite 环境 (标准底层重写方案)
# =================================================================
class PushBottleRobosuiteEnv(Lift):
    def _load_model(self):
        # 1. 调用爷爷类 (ManipulationEnv) 的 _load_model 来安全加载机器人
        super(Lift, self)._load_model()

        # 2. 调整机器人的基座位置以适应桌子
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # 3. 显式创建桌面场景 (彻底解决 arena 找不到的报错)
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0.16, 0, 0])

        # 4. 初始化自带的真实水瓶模型
        self.cube = BottleObject(name="cube")
        
        # 5. 配置物体的初始随机采样器
        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=[self.cube],
            x_range=[-0.1, 0.1],
            y_range=[-0.1, 0.1],
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        )

        # 6. 将场景、机器人和水瓶打包编译进底层物理引擎
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.cube],
        )

# =================================================================
# 2. 定义 Cap-X 框架所需的 Simulator Wrapper
# =================================================================
class PushBottleSimLowLevel(RobosuiteBaseEnv):
    _SUBSAMPLE_RATE = 5

    def __init__(
        self,
        controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json",
        max_steps: int = 1500,
        seed: int | None = None,
        viser_debug: bool = False,
        privileged: bool = False,
        enable_render: bool = False,
    ) -> None:
        super().__init__(
            controller_cfg=controller_cfg, max_steps=max_steps, seed=seed, 
            viser_debug=False, privileged=privileged, enable_render=enable_render
        )

        self.robosuite_env = PushBottleRobosuiteEnv(
            robots=["Panda"],
            has_renderer=not privileged,
            has_offscreen_renderer=True,
            camera_names=self.render_camera_names,
            camera_depths=True,
            renderer="mujoco",
            camera_heights=self._render_height,
            camera_widths=self._render_width,
            controller_configs=load_composite_controller_config(controller=self.controller_cfg),
            horizon=max_steps,
            reward_shaping=True,
        )

        # 覆盖默认的生成位置，保证水瓶在桌子中央附近
        self.robosuite_env.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=[self.robosuite_env.cube],
            x_range=[-0.1, 0.1],
            y_range=[-0.1, 0.1],
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.robosuite_env.table_offset,
            z_offset=0.01,
            rng=self.robosuite_env.rng,
        )

        self._init_robot_links()
        self._init_viser_debug(viser_debug)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.robosuite_env.reset()
        self.robosuite_env.sim.data.qpos[6] -= np.pi

        self._step_count = 0
        self._sim_step_count = 0

        # 让物理引擎运行几步，让瓶子在桌面上落稳
        for _ in range(50):
            self.robosuite_env.sim.forward()
            self.robosuite_env.sim.step()
            self._set_gripper(1.0)

        robosuite_obs = self.robosuite_env._get_observations()
        self._current_joints = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)
        self._current_joints[6] -= np.pi

        obs = self.get_observation()
        self.gripper_link_wxyz_xyz = np.concatenate([
            self.robosuite_env.sim.data.xquat[self.gripper_link_idx],
            self.robosuite_env.sim.data.xpos[self.gripper_link_idx],
        ])

        info = {"task_prompt": "Push the bottle over so that it falls on its side."}
        return obs, info

    def get_observation(self) -> dict:
        robosuite_obs = self.robosuite_env._get_observations()
        base_link_wxyz_xyz = np.concatenate([
            self.robosuite_env.sim.data.xquat[self.base_link_idx],
            self.robosuite_env.sim.data.xpos[self.base_link_idx],
        ])

        # 提取水瓶位姿
        bottle_world = vtf.SE3(wxyz_xyz=np.concatenate([robosuite_obs["cube_quat"], robosuite_obs["cube_pos"]]))
        base_transform = vtf.SE3(wxyz_xyz=base_link_wxyz_xyz).inverse()
        bottle_robot = base_transform @ bottle_world

        # 暴露给 Task 层的位姿
        robosuite_obs["object_poses"] = {
            "bottle": np.concatenate([bottle_robot.translation(), bottle_robot.rotation().wxyz]).astype(np.float32),
        }

        self._process_camera_observations(robosuite_obs)
        self._compute_gripper_obs(robosuite_obs)
        return robosuite_obs

    def compute_reward(self):
        return 1.0 if self.task_completed() else 0.0

    def task_completed(self):
        obs = self.robosuite_env._get_observations()
        z_pos = obs["cube_pos"][2]
        table_height = self.robosuite_env.table_offset[2]
        
        # 判定水瓶重心高度是否低于阈值 (即已推倒)
        return z_pos < (table_height + 0.05)