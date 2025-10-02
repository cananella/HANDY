import os
import sapien
import numpy as np
import numpy as np
import sapien.physx as physx
import torch
from mani_skill.utils import sapien_utils, common
from mani_skill.utils.structs import *
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig

current_file = os.path.dirname(__file__)

project_root = os.path.abspath(os.path.join(current_file, "../../../../"))


@register_agent()
class AIWorker(BaseAgent):
    uid = "ffw"
    urdf_path = os.path.join(current_file, "ffw.urdf")
    srdf_path = os.path.join(current_file, "ffw.srdf")

    fix_root_link = True

    arm_l_joints = [f"arm_l_joint{i}" for i in range(1, 7)]
    arm_r_joints = [f"arm_r_joint{i}" for i in range(1, 7)]
    gripper_l_joints = ["gripper_l_joint1", "gripper_l_joint2", "gripper_l_joint3", "gripper_l_joint4"]
    gripper_r_joints = ["gripper_r_joint1", "gripper_r_joint2", "gripper_r_joint3", "gripper_r_joint4"]
    head_joints = ["head_joint1", "head_joint2"]
    base_joints = ["world_to_base_x_joint", "world_to_base_y_joint", "world_to_base_yaw_joint"]
    lift_joint = ["lift_joint"]

    arm_l_lower = [-3.14,       0,  -3.14, -2.9361,    -3.14,  -1.57]
    arm_l_upper = [ 3.14,    3.14,   3.14,  1.0786,     3.14,   1.57]
    arm_r_lower = [-3.14,   -3.14,  -3.14, -2.9361,    -3.14,  -1.57]
    arm_r_upper = [ 3.14,       0,   3.14,  1.0786,     3.14,   1.57]
    head_lower = [-0.2317, -0.35]
    head_upper = [0.6951, 0.35]
    gripper_lower = [0.0, 0.0, 0.0, 0.0]
    gripper_upper = [1.1, 1.0, 1.1, 1.0]
    base_lower = [-1.0, -1.0, -3.14]
    base_upper = [1.0, 1.0, 3.14]
    lift_lower = [-5.0]
    lift_upper = [0.0]

    keyframes = dict(
        rest=Keyframe(
            pose=sapien.Pose(),
            qpos=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # fmt: skip
        )
    )

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="head_camera",
                pose=sapien.Pose(p=[0.055, 0, -0.01], q=[1, 0, 0, 0]),
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["head_link2"],
            ),
            CameraConfig(
                uid="left_arm_wrist_camera",
                pose=sapien.Pose(p=[0.105, 0.0, -0.08], q=[0.7071,0,0.7071,0]),
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["arm_l_link7"],
            ),
            CameraConfig(
                uid="right_arm_wrist_camera",
                pose=sapien.Pose(p=[0.105, 0.0, -0.08], q=[0.7071,0,0.7071,0]),
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["arm_r_link7"],
            )
        ]
    
    @property
    def _controller_configs(self):
        arm_stiffness = 280
        arm_damping = 20
        arm_force = 400

        gripper_stiffness = 80
        gripper_damping = 2
        gripper_force = 100

        lift_stiffness = 50000
        lift_damping = 30
        lift_force = 3000

        base_controller = PDBaseVelControllerConfig(
            joint_names=self.base_joints,
            lower=self.base_lower,
            upper=self.base_upper,
            friction=10,
            damping=10,
            normalize_action=False,
            drive_mode="force",
        )

        arm_l = PDJointPosControllerConfig(
            joint_names=self.arm_l_joints,
            lower=self.arm_l_lower,
            upper=self.arm_l_upper,
            stiffness=arm_stiffness,
            damping=arm_damping,
            force_limit=arm_force,
            normalize_action=False,
        )
        hand_l_pan = PDJointPosControllerConfig(
            joint_names = ["arm_l_joint7"],
            lower = -1.8201,
            upper = 1.5804,
            stiffness=arm_stiffness,
            damping=0.1,
            force_limit=arm_force,
            normalize_action=False,
        )
        arm_r = PDJointPosControllerConfig(
            joint_names=self.arm_r_joints,
            lower=self.arm_r_lower,
            upper=self.arm_r_upper,
            stiffness=arm_stiffness,
            damping=arm_damping,
            force_limit=arm_force,
            normalize_action=False,
        )
        
        hand_r_pan = PDJointPosControllerConfig(
            joint_names = ["arm_r_joint7"],
            lower = -1.5804,
            upper = 1.8201,
            stiffness=arm_stiffness,
            damping=0.1,
            force_limit=arm_force,
            normalize_action=False,
        )

        gripper_l_1 = PDJointPosMimicControllerConfig(
            joint_names=["gripper_l_joint1", "gripper_l_joint2"],
            lower=[0.0, 0.0],
            upper=[1.1, 1.1],
            stiffness=gripper_stiffness,
            damping=gripper_damping,
            use_delta=False,
            force_limit=gripper_force,
            mimic={
                "gripper_l_joint1" : {"joint": "gripper_l_joint2", "multiplier": 1.0}
            }
        )

        gripper_l_2 = PDJointPosMimicControllerConfig(
            joint_names=["gripper_l_joint3", "gripper_l_joint4"],
            lower=[0.0, 0.0],
            upper=[1.1, 1.1],
            stiffness=gripper_stiffness,
            damping=gripper_damping,
            use_delta=False,
            force_limit=gripper_force,
            mimic={
                "gripper_l_joint3" : {"joint": "gripper_l_joint4", "multiplier": 1.0}
            }
        )


        gripper_r_1 = PDJointPosMimicControllerConfig(
            joint_names=["gripper_r_joint1", "gripper_r_joint2"],
            lower=[0.0, 0.0],
            upper=[1.1, 1.1],
            stiffness=gripper_stiffness,
            damping=gripper_damping,
            use_delta=False,
            force_limit=gripper_force,
            mimic={
                "gripper_r_joint1" : {"joint": "gripper_r_joint2", "multiplier": 1.0}
            }
        )
        gripper_r_2 = PDJointPosMimicControllerConfig(
            joint_names=["gripper_r_joint3", "gripper_r_joint4"],
            lower=[0.0, 0.0],
            upper=[1.1, 1.1],
            stiffness=gripper_stiffness,
            damping=gripper_damping,
            use_delta=False,
            force_limit=gripper_force,
            mimic={
                "gripper_r_joint3" : {"joint": "gripper_r_joint4", "multiplier": 1.0}
            }
        )

        lift = PDJointPosControllerConfig(
            joint_names=self.lift_joint,
            lower=self.lift_lower,
            upper=self.lift_upper,
            stiffness=lift_stiffness,
            friction=10000,
            damping=lift_damping,
            # force_limit=lift_force,
            normalize_action=False,
            drive_mode="force",
        )

        head = PDJointPosControllerConfig(
            joint_names=self.head_joints,
            lower=self.head_lower,
            upper=self.head_upper,
            normalize_action=False,
            stiffness=10,
            damping=10,
        )

        return deepcopy_dict({
            "pd_joint_pos": {
                "arm_l": arm_l,
                "hand_l_pan": hand_l_pan,
                "arm_r": arm_r,
                "hand_r_pan": hand_r_pan,
                "gripper_l_1": gripper_l_1,
                "gripper_l_2": gripper_l_2,
                "gripper_r_1": gripper_r_1,
                "gripper_r_2": gripper_r_2,
                "lift": lift,
                "head": head,
                "base": base_controller,
                "balance_passive_force": False,
            }
        })
    
    def _after_init(self):
        self.base_link: Link = self.robot.links_map["base_link"]
        self.base_link.set_collision_group_bit(group=2, bit_idx=30, bit=1)
        self.base_link.set_collision_group_bit(group=2, bit_idx=2, bit=1)
        self.base_link.set_collision_group_bit(group=2, bit_idx=3, bit=1)

        for link in ["base_link", "head_link1", "head_link2", "arm_base_link", "arm_r_link1", "arm_l_link1", "arm_l_link2", "arm_r_link2"]:
            self.robot.links_map[link].set_collision_group_bit(group=2, bit_idx=1, bit=1)


        for link in [f"arm_l_link{i}" for i in range(1, 8)]:
            self.robot.links_map[link].set_collision_group_bit(group=2, bit_idx=2, bit=1)


        self.camera_l_link = self.robot.links_map["camera_l_link"]
        self.camera_l_link.set_collision_group_bit(group=2, bit_idx=2, bit=1)
        self.camera_l_link.set_collision_group_bit(group=2, bit_idx=4, bit=1)

        for link in [f"arm_r_link{i}" for i in range(1, 8)]:
            self.robot.links_map[link].set_collision_group_bit(group=2, bit_idx=3, bit=1)
            
        self.camera_r_link = self.robot.links_map["camera_r_link"]
        self.camera_r_link.set_collision_group_bit(group=2, bit_idx=3, bit=1)
        self.camera_r_link.set_collision_group_bit(group=2, bit_idx=5, bit=1)


        for link in ["gripper_l_rh_p12_rn_base", "gripper_l_rh_p12_rn_r1", "gripper_l_rh_p12_rn_r2", "gripper_l_rh_p12_rn_l1", "gripper_l_rh_p12_rn_l2"]:
            self.robot.links_map[link].set_collision_group_bit(group=2, bit_idx=2, bit=1)

        
        for link in ["gripper_r_rh_p12_rn_base", "gripper_r_rh_p12_rn_r1", "gripper_r_rh_p12_rn_r2", "gripper_r_rh_p12_rn_l1", "gripper_r_rh_p12_rn_l2"]:
            self.robot.links_map[link].set_collision_group_bit(group=2, bit_idx=3, bit=1)

        self.right_hand_tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), "right_hand_tcp")
        self.left_hand_tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), "left_hand_tcp")


    def _load_scene(self, options: dict):
        self.ground.set_collision_group_bit(group=2, bit_idx=30, bit=1) 

    @property
    def right_hand_tcp_pose(self) -> sapien.Pose:
        return self.right_hand_tcp.pose

    @property
    def left_hand_tcp_pose(self) -> sapien.Pose:
        return self.left_hand_tcp.pose
    
    @property
    def right_hand_tcp_pos(self) -> np.ndarray:
        return self.right_hand_tcp.pose.p

    @property
    def left_hand_tcp_pos(self) -> np.ndarray:
        return self.left_hand_tcp.pose.p
