import gymnasium as gym
import time
import numpy as np
import torch
import cv2
from mani_skill.utils import common
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill_core.robots.robotis_ai_worker import ai_worker 
from mani_skill.agents.controllers.base_controller import DictController
from mani_skill.agents.controllers.base_controller import CombinedController
from mani_skill.agents.controllers.pd_joint_pos import PDJointPosController

def main():
    # === 환경 설정 ===
    env = gym.make(
        "Empty-v1",
        obs_mode="none",
        reward_mode="none",
        render_mode="human",
        robot_uids="ai_worker",
        control_mode="pd_joint_pos",
        sim_backend="cpu",
        sim_config=dict(sim_freq=100, control_freq=20),
    )

    env.reset(seed=0)
    env: BaseEnv = env.unwrapped
    env.agent.robot.set_qpos(env.agent.robot.qpos * 0)

    viewer = env.render()
    robot = env.agent
    
    controller = env.agent.controller
    print(type(controller))
    print(type(robot))
    # print(f"{env.action_space.shape[0]}")
    # print(f"관측 공간: {env.observation_space}")
    # print(f"행동 공간: {env.action_space}")
    # print(f"{env.action_space}")
    ctrl = env.agent.controller  # CombinedController 객체
    offset = 0
    for name, sub_ctrl in ctrl.controllers.items():
        dof = sub_ctrl.action_space.shape[0]
        print(f"{name}: {sub_ctrl.joints} (indices {offset} ~ {offset + dof - 1})")
        offset += dof

    print(env.agent.controllers["pd_joint_pos"])
    print(len(env.agent.controllers["pd_joint_pos"].joints))   # print(f"{env.robot.}")
    print(f"제어 가능한 조인트 수: {env.agent.robot.get_qpos()[0].size()}")
    # print(f"{len(robot.get_links())} 링크 수")
    # print(f"{len(robot.get_active_joints())} 활성화된 조인트 수")
    # print(f"{robot.get_root_pose()}")
    # print(f"{len(robot.get_joints())} 전체 조인트 수")
    # print(f"{robot.find_joint_by_name('arm_l_joint1').index}")
    # print(f"{robot.find_joint_by_name('arm_l_joint2').index}")
    # print(f"{robot.find_joint_by_name('arm_l_joint3').index}")
    # print(f"{robot.find_joint_by_name('arm_l_joint4').index}")
    # print(f"{robot.find_joint_by_name('arm_l_joint5').index}")
    # print(f"{robot.find_joint_by_name('arm_l_joint6').index}")
    # print(f"{robot.find_joint_by_name('arm_l_joint7').index}")
    # print(f"{robot.find_joint_by_name('arm_r_joint1').index}")
    # print(f"{robot.find_joint_by_name('arm_r_joint2').index}")
    # print(f"{robot.find_joint_by_name('arm_r_joint3').index}")
    # print(f"{robot.find_joint_by_name('arm_r_joint4').index}")
    # print(f"{robot.find_joint_by_name('arm_r_joint5').index}")
    # print(f"{robot.find_joint_by_name('arm_r_joint6').index}")
    # print(f"{robot.find_joint_by_name('arm_r_joint7').index}")
    # print(f"헤드 조인트: {env.agent.head_joints}")
    # print(f"베이스 조인트: {env.agent.base_joints}")

    # qpos = env.agent.robot.get_qpos()
    # print("qpos shape:", qpos.shape)  # (1, 8)
    # controlled_joints = env.agent.arm_joint_names[:qpos.shape[1]]
    # print(f"{controlled_joints} 조인트 이름:")
    # print("제어 대상 조인트:")
    # for i, name in enumerate(controlled_joints):
    #     print(f"{i}: {name}")

    left_arm_controller = controller.controllers["arm_l"]
    print(f"왼팔 조인트 이름: {left_arm_controller}")
    right_arm_controller = controller.controllers["arm_r"]
    print(f"오른팔 조인트 이름: {right_arm_controller}")
    gripper_l1_controller = controller.controllers["gripper_l_1"]
    print(f"왼손 그리퍼 조인트 이름: {gripper_l1_controller}")
    gripper_r1_controller = controller.controllers["gripper_r_1"]
    print(f"오른손 그리퍼 조인트 이름: {gripper_r1_controller}")
    gripper_l2_controller = controller.controllers["gripper_l_2"]
    print(f"왼손 그리퍼2 조인트 이름: {gripper_l2_controller}")
    gripper_r2_controller = controller.controllers["gripper_r_2"]
    print(f"오른손 그리퍼2 조인트 이름: {gripper_r2_controller}")
    lift_controller = controller.controllers["lift"]
    print(f"리프트 조인트 이름: {lift_controller}")
    head_controller = controller.controllers["head"]
    print(f"헤드 조인트 이름: {head_controller}")
    base_controller = controller.controllers["base"]
    print(f"베이스 조인트 이름: {base_controller}")
    print(head_controller._start_qpos)
    left_arm_qpos = left_arm_controller.qpos.cpu()[0].detach().numpy()
    right_arm_qpos = right_arm_controller.qpos.cpu()[0].detach().numpy()
    gripper_l1_controller_qpos = gripper_l1_controller.qpos.cpu()[0].detach().numpy()
    gripper_r1_controller_qpos = gripper_r1_controller.qpos.cpu()[0].detach().numpy()
    gripper_l2_controller_qpos = gripper_l2_controller.qpos.cpu()[0].detach().numpy()
    gripper_r2_controller_qpos = gripper_r2_controller.qpos.cpu()[0].detach().numpy()
    lift_qpos = lift_controller.qpos.cpu()[0].detach().numpy()
    head_qpos = head_controller.qpos.cpu()[0].detach().numpy()
    base_qpos = base_controller.qpos.cpu()[0].detach().numpy()
    left_arm_qpos[0] = 1.1
    all_qpos = {
        "arm_l": torch.tensor(left_arm_qpos, dtype=torch.float32),
        "arm_r": torch.tensor(right_arm_qpos, dtype=torch.float32),
        "gripper_l_1": torch.tensor(gripper_l1_controller_qpos[0], dtype=torch.float32),
        "gripper_r_1": torch.tensor(gripper_r1_controller_qpos[0], dtype=torch.float32),
        "gripper_l_2": torch.tensor(gripper_l2_controller_qpos[0], dtype=torch.float32),
        "gripper_r_2": torch.tensor(gripper_r2_controller_qpos[0], dtype=torch.float32),
        "lift": torch.tensor(lift_qpos, dtype=torch.float32),
        "head": torch.tensor(head_qpos, dtype=torch.float32),
        "base": torch.tensor(base_qpos, dtype=torch.float32)
    }
    # print(all_qpos)
    # print(f"왼팔 qpos: {left_arm_qpos}")
    # print(type(left_arm_qpos[0][0]))
    # left_arm_qpos[0][0] = 0.1
    # right_arm_qpos[0][0] = -0.1
    # left_arm_qpos = torch.tensor(left_arm_qpos, dtype=torch.float32)
    # right_arm_qpos = torch.tensor(right_arm_qpos, dtype=torch.float32)
    
    # left_arm_controller.set_action(left_arm_qpos)
    # print(f"왼팔 qpos shape: {left_arm_qpos}")
    # print(f"오른팔 qpos shape: {right_arm_qpos}")
    # controller.set_action()
    # right_arm_controller.set_action(right_arm_qpos)

    qpos = torch.tensor(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                                0.0,
                                0.0, 0.0,
                                0.0, 0.0, 0.0]), dtype=torch.float32).unsqueeze(0)

    print(f"{env.agent.get_controller_state()}")

    # left_arm_controller.from_qpos(left_arm_qpos)
    

    # qpos = common.to_tensor(left_arm_qpos, left_arm_controller.device)
    # print(f"qpos shape: {qpos}")  # (1, 28)
    # obs, reward, terminated, truncated, info = env.step(None)
    # print(f"관측값: {obs}")
    # print(f"보상: {reward}")
    # print(f"종료 여부: {terminated}, 잘림 여부: {truncated}")
    # print(f"추가 정보: {info}")

    # left_arm_controller.get_state()
    # print(f"{robot.find_joint_by_name('arm_l_joint1').get_pose_in_child()}")
    # print(f"{robot.find_joint_by_name('arm_l_joint1').get_drive_target()}")
    # print(f"{robot.find_joint_by_name('arm_l_joint1').get_parent_link().get_name()}")
    # print(f"{robot.find_joint_by_name('arm_l_joint1').get_drive_mode()}")
    # robot.find_joint_by_name("arm_l_joint1").set_drive_target(qpos)

    # robot.set_pose()
    # action = controller.from_qpos(qpos)
    # action_dict = controller.to_action_dict(action)
    # # print(f"초기 액션: {action_dict}")

    # # action_dict["arm_l"][0] += 0.5
    # new_action = controller.from_action_dict(action_dict)
    # env.step(new_action)
    # print(f"{action_dict['arm_l'].shape} {action_dict['arm_l'][0]}")
    # print(type(action_dict['arm_l']))
    # === 조인트 명령 전달 ===
    
    # action = controller.from_qpos(qpos[0])  # (N,) -> (1, N)
    # env.step(action)
    flag = True
    time_now = time.time()
    # env.step(controller.from_action_dict(all_qpos))
    while True:
        # env.step(env.agent.robot.set_qpos(qpos))
        # # === 조인트 위치 업데이트 (예시) ===
        # if time.time() - time_now > 3:
        #     time_now = time.time()
    # #     qpos = robot.find_joint_by_name("arm_l_joint1").get_drive_target().detach().clone().cpu().numpy()
    # #     #     ## 3초에 한번씩 action을 업데이트
        if time.time() - time_now > 10:
            time_now = time.time()
            if flag:
                left_arm_qpos[0] = -1.1
                left_arm_qpos[1] = 1.0
                head_qpos[0] = 0.0
                head_qpos[1] = 0.0
                flag = False
            else:
                left_arm_qpos[0] = 1.1
                left_arm_qpos[1] = 0.0
                head_qpos[0] = 0.0
                head_qpos[1] = 0.0
                flag = True
            all_qpos = {
                "arm_l": torch.tensor(left_arm_qpos, dtype=torch.float32),
                "arm_r": torch.tensor(right_arm_qpos, dtype=torch.float32),
                "gripper_l_1": torch.tensor(gripper_l1_controller_qpos[0], dtype=torch.float32),
                "gripper_r_1": torch.tensor(gripper_r1_controller_qpos[0], dtype=torch.float32),
                "gripper_l_2": torch.tensor(gripper_l2_controller_qpos[0], dtype=torch.float32),
                "gripper_r_2": torch.tensor(gripper_r2_controller_qpos[0], dtype=torch.float32),
                "lift": torch.tensor(lift_qpos, dtype=torch.float32),
                "head": torch.tensor(head_qpos, dtype=torch.float32),
                "base": torch.tensor(base_qpos, dtype=torch.float32)
            }

        env.step(controller.from_action_dict(all_qpos))
        
        env.render()

    # === 실시간 상태 출력 === 
    
    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



# [✓] 전체 조인트 수: 29
# [✓] 제어 가능한 조인트 수: 28
#   ▶ 제어 가능한 조인트 이름:
#     • base_x_joint
#     • base_y_joint
#     • base_yaw_joint
#     • lift_joint
#     • head_joint1
#     • arm_l_joint1
#     • arm_r_joint1
#     • head_joint2
#     • arm_l_joint2
#     • arm_r_joint2
#     • arm_l_joint3
#     • arm_r_joint3
#     • arm_l_joint4
#     • arm_r_joint4
#     • arm_l_joint5
#     • arm_r_joint5
#     • arm_l_joint6
#     • arm_r_joint6
#     • arm_l_joint7
#     • arm_r_joint7
#     • gripper_l_joint1
#     • gripper_l_joint3
#     • gripper_r_joint1
#     • gripper_r_joint3
#     • gripper_l_joint2
#     • gripper_l_joint4
#     • gripper_r_joint2
#     • gripper_r_joint4

# [✓] 전체 링크 수: 29
#   ▶ 링크 이름:
#     • dummy_root_0
#     • base_x
#     • base_y
#     • base_yaw
#     • arm_base_link
#     • head_link1
#     • arm_l_link1
#     • arm_r_link1
#     • head_link2
#     • arm_l_link2
#     • arm_r_link2
#     • arm_l_link3
#     • arm_r_link3
#     • arm_l_link4
#     • arm_r_link4
#     • arm_l_link5
#     • arm_r_link5
#     • arm_l_link6
#     • arm_r_link6
#     • arm_l_link7
#     • arm_r_link7
#     • gripper_l_rh_p12_rn_r1
#     • gripper_l_rh_p12_rn_l1
#     • gripper_r_rh_p12_rn_r1
#     • gripper_r_rh_p12_rn_l1
#     • gripper_l_rh_p12_rn_r2
#     • gripper_l_rh_p12_rn_l2
#     • gripper_r_rh_p12_rn_r2
#     • gripper_r_rh_p12_rn_l2

# [✓] 루트 링크 이름: dummy_root_0

# [✓] 자유도 (DOF): tensor([28])

# [✓] 현재 qpos (조인트 위치, rad 또는 m):
# tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#          0., 0., 0., 0.]])
# [✓] 현재 qvel (조인트 속도):
# tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#          0., 0., 0., 0.]])

# [✓] 루트 Pose:
# Pose(raw_pose=tensor([[0., 0., 0., 1., 0., 0., 0.]]))
# [✓] 루트 선속도:
# tensor([[0., 0., 0.]])
# [✓] 루트 각속도:
# tensor([[0., 0., 0.]])

# [✓] drive_targets (목표 위치):
# tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#          0., 0., 0., 0.]])
# [✓] drive_velocities (목표 속도):
# tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#          0., 0., 0., 0.]])

# [✓] 각 조인트 제한값 [min, max]:
#     • base_x_joint        : [-5.000, 5.000]
#     • base_y_joint        : [-5.000, 5.000]
#     • base_yaw_joint      : [-3.140, 3.140]
#     • lift_joint          : [0.000, 0.250]
#     • head_joint1         : [-0.232, 0.695]
#     • arm_l_joint1        : [-3.140, 3.140]
#     • arm_r_joint1        : [-3.140, 3.140]
#     • head_joint2         : [-0.350, 0.350]
#     • arm_l_joint2        : [0.000, 3.140]
#     • arm_r_joint2        : [-3.140, 0.000]
#     • arm_l_joint3        : [-3.140, 3.140]
#     • arm_r_joint3        : [-3.140, 3.140]
#     • arm_l_joint4        : [-2.936, 1.079]
#     • arm_r_joint4        : [-2.936, 1.079]
#     • arm_l_joint5        : [-3.140, 3.140]
#     • arm_r_joint5        : [-3.140, 3.140]
#     • arm_l_joint6        : [-1.570, 1.570]
#     • arm_r_joint6        : [-1.570, 1.570]
#     • arm_l_joint7        : [-1.820, 1.580]
#     • arm_r_joint7        : [-1.580, 1.820]
#     • gripper_l_joint1    : [0.000, 1.100]
#     • gripper_l_joint3    : [0.000, 1.100]
#     • gripper_r_joint1    : [0.000, 1.100]
#     • gripper_r_joint3    : [0.000, 1.100]
#     • gripper_l_joint2    : [0.000, 1.000]
#     • gripper_l_joint4    : [0.000, 1.000]
#     • gripper_r_joint2    : [0.000, 1.000]
#     • gripper_r_joint4    : [0.000, 1.000]

# [✓] 아티큘레이션 이름: ai_worker
# [✓] 병합된 아티큘레이션인가? False
