import gymnasium as gym
import time
import numpy as np
import torch
import cv2
from mani_skill.utils import common
from mani_skill.envs.sapien_env import BaseEnv
import ai_worker_custom
import sapien.core as sapien
import sapien.utils.viewer
from transforms3d.quaternions import axangle2quat
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers.base_controller import DictController
from mani_skill.agents.controllers.base_controller import CombinedController
from mani_skill.agents.controllers.pd_joint_pos import PDJointPosController


def make_axis_gizmo(scene: sapien.Scene, scale=0.08, name="tcp_axis"):
    L = float(scale)        # 축 길이
    r = L * 0.02            # 반지름
    tip_r = r * 2           # 끝 점 구 반지름

    red   = (1.0, 0.0, 0.0)
    green = (0.0, 1.0, 0.0)
    blue  = (0.0, 0.0, 1.0)

    builder = scene.create_actor_builder()  # 시각 전용(충돌 X)

    # X (빨강) - 캡슐 기본 방향은 x축
    builder.add_capsule_visual(
        pose=sapien.Pose([L/2, 0, 0]),
        radius=r, half_length=L/2,
        material=red
    )
    builder.add_sphere_visual(
        pose=sapien.Pose([L, 0, 0]),
        radius=tip_r,
        material=red
    )

    # Y (초록) - z축 +90° 회전해서 x->y 정렬
    q_y = axangle2quat([0, 0, 1], np.pi/2)  # wxyz
    builder.add_capsule_visual(
        pose=sapien.Pose([0, L/2, 0], q=q_y),
        radius=r, half_length=L/2,
        material=green
    )
    builder.add_sphere_visual(
        pose=sapien.Pose([0, L, 0]),
        radius=tip_r,
        material=green
    )

    # Z (파랑) - y축 -90° 회전해서 x->z 정렬
    q_z = axangle2quat([0, 1, 0], -np.pi/2)
    builder.add_capsule_visual(
        pose=sapien.Pose([0, 0, L/2], q=q_z),
        radius=r, half_length=L/2,
        material=blue
    )
    builder.add_sphere_visual(
        pose=sapien.Pose([0, 0, L]),
        radius=tip_r,
        material=blue
    )

    return builder.build_kinematic(name=name)  # KinematicActor 반환

def main():
    # === 환경 설정 ===
    env = gym.make(
        "Empty-v1",
        obs_mode="none",
        reward_mode="none",
        render_mode="human",
        robot_uids="ffw",
        control_mode="pd_joint_pos",
        sim_backend="auto",
        sim_config=dict(sim_freq=100, control_freq=50),
    )

    env.reset(seed=0)
    env: BaseEnv = env.unwrapped
    controller : CombinedController = env.agent.controller
    agent : BaseAgent = env.agent 
    viewer = env.render_human()
    env.agent.robot.set_qpos(env.agent.robot.qpos * 0)

    left_arm_controller = controller.controllers["arm_l"]
    left_hand_pan_controller = controller.controllers["hand_l_pan"]
    right_arm_controller = controller.controllers["arm_r"]
    right_hand_pan_controller = controller.controllers["hand_r_pan"]
    gripper_l1_controller = controller.controllers["gripper_l_1"]
    gripper_r1_controller = controller.controllers["gripper_r_1"]
    gripper_l2_controller = controller.controllers["gripper_l_2"]
    gripper_r2_controller = controller.controllers["gripper_r_2"]
    lift_controller = controller.controllers["lift"]
    head_controller = controller.controllers["head"]
    base_controller = controller.controllers["base"]

    left_arm_qpos = left_arm_controller.qpos.cpu()[0].detach().numpy()
    left_hand_pan_qpos = left_hand_pan_controller.qpos.cpu()[0].detach().numpy()
    right_arm_qpos = right_arm_controller.qpos.cpu()[0].detach().numpy()
    right_hand_pan_qpos = right_hand_pan_controller.qpos.cpu()[0].detach().numpy()
    gripper_l1_qpos = gripper_l1_controller.qpos.cpu()[0].detach().numpy()
    gripper_r1_qpos = gripper_r1_controller.qpos.cpu()[0].detach().numpy()
    gripper_l2_qpos = gripper_l2_controller.qpos.cpu()[0].detach().numpy()
    gripper_r2_qpos = gripper_r2_controller.qpos.cpu()[0].detach().numpy()
    lift_qpos = lift_controller.qpos.cpu()[0].detach().numpy()
    head_qpos = head_controller.qpos.cpu()[0].detach().numpy()
    base_qpos = base_controller.qpos.cpu()[0].detach().numpy()


    all_qpos = {
        "arm_l": torch.tensor(left_arm_qpos, dtype=torch.float32),
        "hand_l_pan": torch.tensor(left_hand_pan_qpos, dtype=torch.float32),
        "arm_r": torch.tensor(right_arm_qpos, dtype=torch.float32),
        "hand_r_pan": torch.tensor(right_hand_pan_qpos, dtype=torch.float32),
        "gripper_l_1": torch.tensor([gripper_l1_qpos[0]], dtype=torch.float32),
        "gripper_r_1": torch.tensor([gripper_r1_qpos[0]], dtype=torch.float32),
        "gripper_l_2": torch.tensor([gripper_l2_qpos[0]], dtype=torch.float32),
        "gripper_r_2": torch.tensor([gripper_r2_qpos[0]], dtype=torch.float32),
        "lift": torch.tensor(lift_qpos, dtype=torch.float32),
        "head": torch.tensor(head_qpos, dtype=torch.float32),
        "base": torch.tensor(base_qpos, dtype=torch.float32)
    }

    for plugin in viewer.plugins:
        if isinstance(plugin, sapien.utils.viewer.viewer.TransformWindow):
            transform_window = plugin

    transform_window.enabled = True

    ln = {link.name: link for link in env.agent.robot.get_links()}
    l_hand = ln["left_hand_tcp"]
    r_hand = ln["right_hand_tcp"]
    l_elbow = ln["left_elbow_tcp"]
    r_elbow = ln["right_elbow_tcp"]

    l_hand_axis = make_axis_gizmo(env.scene, scale=0.3, name="left_hand_tcp_axis")
    r_hand_axis = make_axis_gizmo(env.scene, scale=0.3, name="right_hand_tcp_axis")
    l_elbow_axis = make_axis_gizmo(env.scene, scale=0.3, name="left_elbow_tcp_axis")
    r_elbow_axis = make_axis_gizmo(env.scene, scale=0.3, name="right_elbow_tcp_axis")

    flag = True
    time_now = time.time()
    while True:

        # # # 3초마다 목표 변경
        left_arm_qpos = left_arm_controller.qpos.cpu()[0].detach().numpy()
        left_hand_pan_qpos = left_hand_pan_controller.qpos.cpu()[0].detach().numpy()
        right_arm_qpos = right_arm_controller.qpos.cpu()[0].detach().numpy()
        right_hand_pan_qpos = right_hand_pan_controller.qpos.cpu()[0].detach().numpy()
        gripper_l1_controller_qpos = gripper_l1_controller.qpos.cpu()[0].detach().numpy()
        gripper_r1_controller_qpos = gripper_r1_controller.qpos.cpu()[0].detach().numpy()
        gripper_l2_controller_qpos = gripper_l2_controller.qpos.cpu()[0].detach().numpy()
        gripper_r2_controller_qpos = gripper_r2_controller.qpos.cpu()[0].detach().numpy()
        lift_qpos = lift_controller.qpos.cpu()[0].detach().numpy()
        head_qpos = head_controller.qpos.cpu()[0].detach().numpy()
        base_qpos = base_controller.qpos.cpu()[0].detach().numpy()
        
        if time.time() - time_now > 3:
            time_now = time.time()
            if flag:
                left_arm_qpos = [ 1.0, -0.3 , 0.0, -1.5, 0.0, 0.0]
                left_hand_pan_qpos = [1.0]
                head_qpos = [0.0, 0.0]
                lift_qpos = [0.0]
                flag = False
            else:
                left_arm_qpos = [0.2, 0.1, 0.0, 0.0, 0.0, 0.0]
                left_hand_pan_qpos = [-1.0]
                head_qpos = [0.0, 0.0]
                lift_qpos = [0.0]
                flag = True
            all_qpos.update({
                "arm_l": torch.tensor(left_arm_qpos, dtype=torch.float32),
                "hand_l_pan": torch.tensor(left_hand_pan_qpos, dtype=torch.float32),
                "head": torch.tensor(head_qpos, dtype=torch.float32),
                "lift": torch.tensor(lift_qpos, dtype=torch.float32),
            })
            # print(" all_qpos : ", all_qpos)
            
        env.step(controller.from_action_dict(all_qpos))
        env.render_human()

        ### 충돌보기
        ## contacts =env.scene.get_contacts()
        ## for c in contacts:
        ##     print(" contact : ", c)

        # for link in env.agent.robot.get_links():
        #     print(link.get_name(), link.pose)
        r_hand_tcp_pos = r_hand.pose
        l_hand_tcp_pos = l_hand.pose
        r_elbow_tcp_pos = r_elbow.pose
        l_elbow_tcp_pos = l_elbow.pose
        l_hand_axis.set_pose(l_hand_tcp_pos)
        r_hand_axis.set_pose(r_hand_tcp_pos)
        l_elbow_axis.set_pose(l_elbow_tcp_pos)
        r_elbow_axis.set_pose(r_elbow_tcp_pos)

        env.render()
        # print(" base_controller.qpos : ", base_controller.qpos.cpu()[0].detach().numpy())
        # print(" base_controller.qvel : ", base_controller.qvel.cpu()[0].detach().numpy())
        # === 실시간 상태 출력 === 
    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
