import os
import sapien
import numpy as np
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig

current_file = os.path.abspath(__file__)

# 프로젝트 루트 경로: ai_worker.py -> robotis_ai_worker -> robots -> mani_skill_core -> src -> HANDY
project_root = os.path.abspath(os.path.join(current_file, "../../../../.."))


@register_agent()
class AIWorker(BaseAgent):
    uid = "ai_worker"

    # MJCF 파일의 상대경로를 project_root 기준으로 설정
    mjcf_path = os.path.join(
        project_root,
        "third_party",
        "robotis_mujoco_menagerie",
        "robotis_ffw",
        "ffw_bg2.xml"
    )

    fix_root_link = False

    arm_l_joints = [f"arm_l_joint{i}" for i in range(1, 8)]
    arm_r_joints = [f"arm_r_joint{i}" for i in range(1, 8)]
    gripper_l_joints = [f"gripper_l_joint{i}" for i in range(1, 5)]
    gripper_r_joints = [f"gripper_r_joint{i}" for i in range(1, 5)]
    head_joints = ["head_joint1", "head_joint2"]
    base_joints = ["base_x_joint", "base_y_joint", "base_yaw_joint"]
    lift_joint = ["lift_joint"]

    # 각 조인트별 range (MJCF 정의 기반)
    arm_l_lower = [-3.14, 0, -6.28, -1.86, -1.57, -1.57, -1.046]
    arm_l_upper = [3.14, 3.14, 6.28, 1.86, 1.57, 1.57, 1.5804]
    arm_r_lower = [-3.14, -3.14, -1.57, -1.86, -1.57, -1.57, -1.5804]
    arm_r_upper = [3.14, 0, 1.57, 1.86, 1.57, 1.57, 1.8201]
    head_lower = [-0.2317, -0.35]
    head_upper = [0.6951, 0.35]
    gripper_lower = [0.0, 0.0, 0.0, 0.0]
    gripper_upper = [1.1, 1.0, 1.1, 1.0]
    base_lower = [-1.0, -1.0, -3.14]
    base_upper = [1.0, 1.0, 3.14]
    lift_lower = [0.0]
    lift_upper = [0.25]

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
        arm_stiffness = 1000
        arm_damping = 100
        arm_force = 200

        gripper_stiffness = 1000
        gripper_damping = 50
        gripper_force = 50

        lift_stiffness = 3000
        lift_damping = 300
        lift_force = 1000

        base_controller = PDBaseVelControllerConfig(
            joint_names=self.base_joints,
            lower=self.base_lower,
            upper=self.base_upper,
            damping=800,
            force_limit=500,
        )

        arm_l = PDJointPosControllerConfig(
            joint_names=self.arm_l_joints,
            lower=self.arm_l_lower,
            upper=self.arm_l_upper,
            stiffness=arm_stiffness,
            damping=arm_damping,
            force_limit=arm_force,
        )
        arm_r = PDJointPosControllerConfig(
            joint_names=self.arm_r_joints,
            lower=self.arm_r_lower,
            upper=self.arm_r_upper,
            stiffness=arm_stiffness,
            damping=arm_damping,
            force_limit=arm_force,
        )

        gripper_l_1 = PDJointPosMimicControllerConfig(
            joint_names=["gripper_l_joint1", "gripper_l_joint2"],
            lower=[0.0, 0.0],
            upper=[1.1, 1.0],
            stiffness=gripper_stiffness,
            damping=gripper_damping,
            force_limit=gripper_force,
        )

        gripper_l_2 = PDJointPosMimicControllerConfig(
            joint_names=["gripper_l_joint3", "gripper_l_joint4"],
            lower=[0.0, 0.0],
            upper=[1.1, 1.0],
            stiffness=gripper_stiffness,
            damping=gripper_damping,
            force_limit=gripper_force,
        )


        gripper_r_1 = PDJointPosMimicControllerConfig(
            joint_names=["gripper_r_joint1", "gripper_r_joint2"],
            lower=[0.0, 0.0],
            upper=[1.1, 1.0],
            stiffness=gripper_stiffness,
            damping=gripper_damping,
            force_limit=gripper_force,
        )
        gripper_r_2 = PDJointPosMimicControllerConfig(
            joint_names=["gripper_r_joint3", "gripper_r_joint4"],
            lower=[0.0, 0.0],
            upper=[1.1, 1.0],
            stiffness=gripper_stiffness,
            damping=gripper_damping,
            force_limit=gripper_force,
        )

        lift = PDJointPosControllerConfig(
            joint_names=self.lift_joint,
            lower=self.lift_lower,
            upper=self.lift_upper,
            stiffness=lift_stiffness,
            damping=lift_damping,
            force_limit=lift_force,
        )

        head = PDJointPosControllerConfig(
            joint_names=self.head_joints,
            lower=self.head_lower,
            upper=self.head_upper,
            stiffness=300,
            damping=50,
            force_limit=100,
        )

        return deepcopy_dict({
            "pd_joint_pos": {
                "arm_l": arm_l,
                "arm_r": arm_r,
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