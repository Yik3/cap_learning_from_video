# capx/envs/simulators/flip_bottle_sim.py

import numpy as np
import robosuite as suite
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BottleObject  
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.controllers import load_composite_controller_config
import viser.transforms as vtf

from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv
from capx.envs.base import register_env 

class FlipBottleRobosuiteEnv(Lift):
    def _load_model(self):
        super(Lift, self)._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0.16, 0, 0])

        self.cube = BottleObject(name="cube") 
        
        # [CHANGE 1A]: Rotate 180 degrees around the X-axis to invert the bottle
        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=[self.cube],
            x_range=[-0.1, 0.1],
            y_range=[-0.1, 0.1],
            rotation=[np.pi, np.pi], 
            rotation_axis='x',       
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.cube],
        )

class FlipBottleSimLowLevel(RobosuiteBaseEnv):
    _SUBSAMPLE_RATE = 5

    def __init__(self, controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json", max_steps: int = 1500, seed: int | None = None, viser_debug: bool = False, privileged: bool = False, enable_render: bool = False) -> None:
        super().__init__(controller_cfg=controller_cfg, max_steps=max_steps, seed=seed, viser_debug=False, privileged=privileged, enable_render=enable_render)

        self.robosuite_env = FlipBottleRobosuiteEnv(
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

        # [CHANGE 1B]: Apply the same inverted rotation setup to the override sampler
        self.robosuite_env.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=[self.robosuite_env.cube],
            x_range=[-0.1, 0.1],
            y_range=[-0.1, 0.1],
            rotation=[np.pi, np.pi], 
            rotation_axis='x',       
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.robosuite_env.table_offset,
            z_offset=0.01,
            rng=self.robosuite_env.rng,
        )

        self._init_robot_links()
        self._init_viser_debug(viser_debug)
        
        self.cumulative_flip_angle = 0.0
        self.prev_bottle_quat_wxyz = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.robosuite_env.reset()
        self.robosuite_env.sim.data.qpos[6] -= np.pi

        self._step_count = 0
        self._sim_step_count = 0
        self.cumulative_flip_angle = 0.0
        self.prev_bottle_quat_wxyz = None

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

        # [CHANGE 4]: Update the prompt instructions to target 180 degrees
        info = {"task_prompt": "Flip the inverted bottle 180 degrees upright. You will get a continuous reward proportional to the flipped angle (angle / 180)."}
        return obs, info

    def get_observation(self) -> dict:
        robosuite_obs = self.robosuite_env._get_observations()
        
        base_link_wxyz_xyz = np.concatenate([
            self.robosuite_env.sim.data.xquat[self.base_link_idx],
            self.robosuite_env.sim.data.xpos[self.base_link_idx],
        ])

        cube_quat_xyzw = robosuite_obs["cube_quat"]
        cube_quat_wxyz = np.array([cube_quat_xyzw[3], cube_quat_xyzw[0], cube_quat_xyzw[1], cube_quat_xyzw[2]])
        
        if self.prev_bottle_quat_wxyz is not None:
            dot = np.clip(np.abs(np.dot(self.prev_bottle_quat_wxyz, cube_quat_wxyz)), 0.0, 1.0)
            self.cumulative_flip_angle += np.degrees(2 * np.arccos(dot))
        self.prev_bottle_quat_wxyz = cube_quat_wxyz

        bottle_world = vtf.SE3(wxyz_xyz=np.concatenate([cube_quat_wxyz, robosuite_obs["cube_pos"]]))
        base_transform = vtf.SE3(wxyz_xyz=base_link_wxyz_xyz).inverse()
        bottle_robot = base_transform @ bottle_world

        robosuite_obs["object_poses"] = {
            "bottle": np.concatenate([bottle_robot.translation(), bottle_robot.rotation().wxyz]).astype(np.float32),
        }

        self._process_camera_observations(robosuite_obs)
        self._compute_gripper_obs(robosuite_obs)
        return robosuite_obs

    def compute_reward(self):
        """
        Continuous reward proportional to the rotated angle.
        n / 180 where n is cumulative_flip_angle.
        """
        flip_angle = self.cumulative_flip_angle
        # [CHANGE 2]: Scale reward by 180.0 instead of 360.0
        reward = np.clip(flip_angle / 180.0, 0.0, 1.0)
        return float(reward)

    def task_completed(self):
        # [CHANGE 3]: Condition for success is now 180.0 degrees
        return self.cumulative_flip_angle >= 180.0

# Optional: Ensure you register the environment at the bottom if not done elsewhere
# register_env("flip_bottle_sim_low_level", FlipBottleSimLowLevel)