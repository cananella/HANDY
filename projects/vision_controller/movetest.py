import gymnasium as gym
import time
import numpy as np
from mani_skill.envs.sapien_env import BaseEnv

# AIWorker 등록 필수
from mani_skill_core.robots.robotis_ai_worker import ai_worker  # noqa

def main():
    # 환경 생성
    env = gym.make(
        "Empty-v1",
        obs_mode="none",
        reward_mode="none",
        render_mode="human",
        robot_uids="ai_worker",
        control_mode="pd_joint_pos",
        sim_backend="cpu",  # 혹은 "gpu"
        sim_config=dict(sim_freq=100, control_freq=20),
    )

    # 환경 초기화
    env.reset(seed=42)
    env: BaseEnv = env.unwrapped
    robot = env.agent.robot
    controller = env.agent.controller

    print("▶ 현재 로봇 조인트 이름:", robot.joint_names)
    print("▶ 현재 로봇 qpos shape:", robot.qpos.shape)

    # 모든 조인트를 초기화
    robot.set_qpos(np.zeros_like(robot.qpos))

    # 컨트롤할 타겟 조인트 설정 (예: 왼쪽 팔)
    joint_names = ["arm_l_joint1", "arm_l_joint2"]
    target_positions = [0.5, -1.0]  # 라디안 단위

    # DictController 사용: joint_name -> target pos
    if hasattr(controller, "from_qpos"):
        qpos_dict = dict(zip(joint_names, target_positions))
        print("▶ 적용할 조인트 명령:", qpos_dict)
        action = controller.from_qpos(qpos_dict)
    else:
        raise RuntimeError("컨트롤러가 DictController 타입이 아닙니다")

    # 시뮬레이션 루프 (2초간)
    for i in range(2 * env.sim_freq):
        env.step(action)
        env.render()
        time.sleep(1 / env.sim_freq)

    env.close()


if __name__ == "__main__":
    main()
