import gymnasium as gym
import time
import numpy as np
import torch
import cv2
from mani_skill.agents.controllers.base_controller import DictController
from mani_skill.agents.controllers.base_controller import CombinedController
from mani_skill.agents.controllers.pd_joint_pos import PDJointPosController


from mani_skill.envs.sapien_env import BaseEnv
# ManiSkill 환경 로드
env = gym.make(
    "Empty-v1",  # 빈 환경 (바닥만 있음)
    obs_mode="none",
    reward_mode="none",
    render_mode="human",  # GUI 렌더링
    control_mode="pd_joint_pos",  # 조인트 위치 제어
    robot_uids="panda",  # 로봇 종류 (ai_worker 등 다른 모델도 가능)
)

# 환경 초기화
env.reset()

# 현재 qpos의 shape 확인
qpos = env.agent.robot.get_qpos()
print("qpos shape:", qpos.shape)  # (1, N) 형태

# 조인트 이름과 순서 확인
joint_names = env.agent.robot.joints
print("조인트 순서:")
for i, name in enumerate(joint_names):
    print(f"{i}: {name}")

qpos = env.agent.robot.get_qpos()
print("qpos shape:", qpos.shape)  # (1, 8)

# 전체 조인트에서 앞에서부터 N개가 제어 대상인 경우:
controlled_joints = env.agent.arm_joint_names[:qpos.shape[1]]
print(f"{controlled_joints} 조인트 이름:")
print("제어 대상 조인트:")
for i, name in enumerate(controlled_joints):
    print(f"{i}: {name}")


# 예시 조인트 위치 (panda_joint1~7 + panda_finger_joint1)
# 8개의 조인트 → shape (1, 8)
target_joint_angles = np.array([[  # <-- 2차원으로 만들어야 함
    0.0,     # joint1
    -0.5,    # joint2
    0.0,     # joint3
    -1.5,    # joint4
    0.0,     # joint5
    1.0,     # joint6
    0.5,     # joint7
    0.04     # finger_joint1 (보통 하나만 제어하면 양쪽에 적용됨)
]])

# 실행
while True:
    obs, reward, terminated, truncated, info = env.step(action=target_joint_angles)
    time.sleep(1 / 60)
    env.render()

env.close()
