from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose  
from mani_skill.utils.logging_utils import logger

@register_env("StackPyramid-v1", max_episode_steps=50)
class StackPyramidEnv(BaseEnv):
    """
    **Task Description:**
    - The goal is to pick up a red cube, place it next to the green cube, and stack the blue cube on top of the red and green cube without it falling off.

    **Randomizations:**
    - both cubes have their z-axis rotation randomized
    - both cubes have their xy positions on top of the table scene randomized. The positions are sampled such that the cubes do not collide with each other

    **Success Conditions:**
    - the blue cube is static
    - the blue cube is on top of both the red and green cube (to within half of the cube size)
    - none of the red, green, blue cubes are grasped by the robot (robot must let go of the cubes)

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/StackPyramid-v1_rt.mp4"

    """

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    SUPPORTED_REWARD_MODES = ["none", "sparse"]
    
    agent: Union[Panda, Fetch]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        """
        reset_states: Optional dict to override the reset state.
          Expected format:
              {
                  "cubeA": {"p": [x, y, z], "q": [qx, qy, qz, qw]},  # optional "q" can be omitted
                  "cubeB": {"p": [x, y, z], "q": [qx, qy, qz, qw]},
                  "cubeC": {"p": [x, y, z], "q": [qx, qy, qz, qw]}
              }
        """
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if "reset_states" in kwargs.keys():
            self.reset_states = kwargs["reset_states"]
        else:
            self.reset_states = None
        kwargs.pop("reset_states", None)
        if "sample_region" in kwargs.keys():
            self.sample_region = kwargs["sample_region"]
            print("Sample region: ", self.sample_region)
            kwargs.pop("sample_region", None)
        else:
            self.sample_region = None
        if "eval_sample_region" in kwargs.keys():
            self.eval_sample_region = kwargs["eval_sample_region"]
            print("Eval Sample region: ", self.eval_sample_region)
            kwargs.pop("eval_sample_region", None)
        else:
            if self.sample_region is not None:
                self.eval_sample_region = self.sample_region
            else:
                self.eval_sample_region = None

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cubeA = actors.build_cube(
            self.scene, half_size=0.02, color=[1, 0, 0, 1], name="cubeA", initial_pose=sapien.Pose(p=[0, 0, 0.1])
        )
        self.cubeB = actors.build_cube(
            self.scene, half_size=0.02, color=[0, 1, 0, 1], name="cubeB", initial_pose=sapien.Pose(p=[1, 0, 0.1])
        )
        self.cubeC = actors.build_cube(
            self.scene, half_size=0.02, color=[0, 0, 1, 1], name="cubeC", initial_pose=sapien.Pose(p=[-1, 0, 0.1])
        )
        print(self.sample_region)
        if self.sample_region is not None:
            self.goal_site = actors.build_box(self.scene, half_sizes=[self.sample_region, self.sample_region, 0.01], color=[1,1,0,1], name="goal_site", body_type="kinematic", add_collision=False, initial_pose=sapien.Pose(p=[0,0,0]))
            self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            if self.sample_region is not None:
                goal_site_xy = torch.tensor([0.0, 0.0], device=self.device).repeat(b, 1)
                goal_site_xyz = torch.zeros((b, 3), device=self.device)
                goal_site_xyz[:, :2] = goal_site_xy
                self.goal_site.set_pose(Pose.create_from_pq(p=goal_site_xyz.clone(), q=torch.tensor([0,0,0,1])))
            
            # Default fixed quaternion if not provided in reset_states:
            # FIXED_QS_EULER = torch.zeros((1, 3))
            # FIXED_QS = matrix_to_quaternion(euler_angles_to_matrix(FIXED_QS_EULER, convention="XYZ"))
            FIXED_QS = torch.tensor([0,0,0,1], device=self.device)
            if self.sample_region is not None:
                if self.eval_sample_region is not None:
                    sample_region = torch.distributions.uniform.Uniform(-self.eval_sample_region, self.eval_sample_region)
                else:
                    sample_region = torch.distributions.uniform.Uniform(-self.sample_region, self.sample_region)
                base_xy = torch.tensor([0.0, 0.0], device=self.device).repeat(b, 1)

                offset_A, offset_B, offset_C = sample_safe_offsets(b, sample_region, min_distance=0.1, device=self.device)

                cubeA_xy = base_xy + offset_A
                cubeB_xy = base_xy + offset_B
                cubeC_xy = base_xy + offset_C

                xyz = torch.zeros((b, 3), device=self.device)
                xyz[:, 2] = 0.02  # table height offset

                # Cube A
                xyz[:, :2] = cubeA_xy
                self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=FIXED_QS.clone()))

                # Set Cube B
                xyz[:, :2] = cubeB_xy
                self.cubeB.set_pose(Pose.create_from_pq(p=xyz.clone(), q=FIXED_QS.clone()))
                    
                # Set Cube C
                xyz[:, :2] = cubeC_xy
                self.cubeC.set_pose(Pose.create_from_pq(p=xyz.clone(), q=FIXED_QS.clone()))
            else:
                if self.reset_states != {} and self.reset_states is not None:
                    logger.info(f"Reset States: {self.reset_states}")
                    # Use provided reset states for cubes if available. Expects one dict per cube.
                    for cube_name, cube in zip(["cubeA", "cubeB", "cubeC"], [self.cubeA, self.cubeB, self.cubeC]):
                        print(f"self.reset_states: {self.reset_states}")
                        if cube_name in self.reset_states:
                            state = self.reset_states[cube_name]
                            # Convert provided pose to torch tensors
                            p = torch.tensor(state.get("p"), device=self.device).view(1, 3)
                            # If quaternion provided, use it; otherwise use the fixed one.
                            q = torch.tensor(state.get("q"), device=self.device).view(1, 4) if "q" in state else FIXED_QS
                            cube.set_pose(Pose.create_from_pq(p=p, q=q))
                        else:
                            # Fallback to default fixed pose:
                            xyz = torch.zeros((b, 3))
                            xyz[:, 2] = 0.02
                            xyz[:, :2] = torch.zeros((b, 2), device=self.device)
                            cube.set_pose(Pose.create_from_pq(p=xyz, q=FIXED_QS))
                else:
                    base_xy = torch.tensor([0.0, 0.0], device=self.device).repeat(b, 1)
                    offset_A = torch.tensor([-0.07, 0.05], device=self.device).repeat(b, 1)
                    offset_B = torch.tensor([0.08, -0.1], device=self.device).repeat(b, 1)
                    cubeA_xy = base_xy + offset_A
                    cubeB_xy = base_xy + offset_B
                    cubeC_xy = torch.tensor([-0.3, -0.025], device=self.device).repeat(b, 1)

                    # Cube A
                    xyz = torch.zeros((b, 3), device=self.device)
                    xyz[:, 2] = 0.02  # table height offset
                    xyz[:, :2] = cubeA_xy
                    self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=FIXED_QS))

                    # Set Cube B
                    xyz[:, :2] = cubeB_xy
                    self.cubeB.set_pose(Pose.create_from_pq(p=xyz.clone(), q=FIXED_QS))
                    
                    # Set Cube C
                    xyz[:, 2] = 0.02
                    xyz[:, :2] = cubeC_xy
                    self.cubeC.set_pose(Pose.create_from_pq(p=xyz, q=FIXED_QS))

    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        pos_C = self.cubeC.pose.p

        offset_AB = pos_A - pos_B
        offset_BC = pos_B - pos_C
        offset_AC = pos_A - pos_C

        def evaluate_cube_distance(offset, cube_a, cube_b, top_or_next):
            tolerance = 0.5
            if top_or_next == "top":
                xy_offset = torch.linalg.norm(offset[..., :2], axis=-1) - torch.linalg.norm(self.cube_half_size[:2])
                z_offset = torch.linalg.norm(offset[..., 2]) - torch.linalg.norm(2 * self.cube_half_size[2])
                
                xy_flag = xy_offset <= tolerance
                z_flag = z_offset <= tolerance
            else:
                xy_offset = torch.linalg.norm(offset[..., :2], axis=-1) - torch.linalg.norm(2 * self.cube_half_size[:2])
                z_offset = torch.abs(offset[..., 2] - self.cube_half_size[2])
                xy_flag = xy_offset <= 0.05
                z_flag = z_offset <= 0.05
                
            is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
            is_cubeA_static = cube_a.is_static(lin_thresh=1e-2, ang_thresh=0.5)
            is_cubeA_grasped = self.agent.is_grasping(cube_a)

            success = is_cubeA_on_cubeB & is_cubeA_static & (~is_cubeA_grasped)            
            return success.bool()

        success_A_B = evaluate_cube_distance(offset_AB, self.cubeA, self.cubeB, "next_to")
        success_C_B = evaluate_cube_distance(offset_BC, self.cubeC, self.cubeB, "top")
        success_C_A = evaluate_cube_distance(offset_AC, self.cubeC, self.cubeA, "top")

        
        success = torch.logical_and(success_A_B, torch.logical_and(success_C_B, success_C_A))
        return {
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                cubeC_pose=self.cubeC.pose.raw_pose,
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeC_pos=self.cubeC.pose.p - self.agent.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
                cubeB_to_cubeC_pos=self.cubeC.pose.p - self.cubeB.pose.p,
                cubeA_to_cubeC_pos=self.cubeC.pose.p - self.cubeA.pose.p,
            )
        return obs

def sample_safe_offsets(b: int, sample_region: torch.distributions.Uniform, min_distance: float = 0.1, max_iter: int = 100, device: torch.device = torch.device("cpu")):
    """
    For each sample in the batch, sample three (2,) offsets such that
    all pairwise distances are at least min_distance.
    
    Returns three tensors of shape (b, 2) for offsets A, B, and C.
    """
    offsets_A = torch.zeros((b, 2), device=device)
    offsets_B = torch.zeros((b, 2), device=device)
    offsets_C = torch.zeros((b, 2), device=device)
    for i in range(b):
        valid = False
        iter_count = 0
        while (not valid) and (iter_count < max_iter):
            # Sample offsets for cube A, B, and C for the i-th sample.
            off = sample_region.sample((3, 2))  # shape (3,2)
            d_AB = torch.norm(off[0] - off[1])
            d_AC = torch.norm(off[0] - off[2])
            d_BC = torch.norm(off[1] - off[2])
            if d_AB >= min_distance and d_AC >= min_distance and d_BC >= min_distance:
                offsets_A[i] = off[0]
                offsets_B[i] = off[1]
                offsets_C[i] = off[2]
                valid = True
            iter_count += 1
        if not valid:
            # If no valid sample after max_iter iterations, simply use the last sample.
            print("no valid sample")
            offsets_A[i] = off[0]
            offsets_B[i] = off[1]
            offsets_C[i] = off[2]
    return offsets_A, offsets_B, offsets_C