import gymnasium as gym
import time
import numpy as np
import torch
import cv2
from mani_skill.utils import common
from mani_skill.envs.sapien_env import BaseEnv
import ai_worker_custom
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
        sim_backend="auto",
        sim_config=dict(sim_freq=100, control_freq=50),
    )

    env.reset(seed=0)
    env: BaseEnv = env.unwrapped
    controller : CombinedController = env.agent.controller
    # for name, sub in controller.controllers.items():
    #     if hasattr(sub, "balance_passive_force"):
    #         sub.balance_passive_force = True

    # print("balance_passive_force:",
    #     getattr(controller, "balance_passive_force", None),
    #     [(n, getattr(c, "balance_passive_force", None)) for n,c in controller.controllers.items()])
    # env.agent.robot.set_qpos(env.agent.robot.qpos * 0)

    
    zero_action_dict = {}
    for name, sub_ctrl in controller.controllers.items():
        shape = sub_ctrl.action_space.shape          # 예: (7,), (1,), (2,) ...
        zero_action_dict[name] = torch.zeros(shape, dtype=torch.float32)

    
    zero = controller.from_action_dict(zero_action_dict)

    
    # try:
    #     for _ in range(50):      # 0.5초 정도 정착
    #         env.step(zero)
    # except Exception:
    #     zero_batched = zero[None, ...]  # (1, dof)
    #     for _ in range(50):
    #         env.step(zero_batched)

    robot = env.agent
    print(type(controller))
    print(type(robot))
    ctrl = env.agent.controller  # CombinedController 객체

    left_arm_controller = controller.controllers["arm_l"]
    right_arm_controller = controller.controllers["arm_r"]
    gripper_l1_controller = controller.controllers["gripper_l_1"]
    gripper_r1_controller = controller.controllers["gripper_r_1"]
    gripper_l2_controller = controller.controllers["gripper_l_2"]
    gripper_r2_controller = controller.controllers["gripper_r_2"]
    lift_controller = controller.controllers["lift"]
    head_controller = controller.controllers["head"]
    base_controller = controller.controllers["base"]
    left_arm_qpos = left_arm_controller.qpos.cpu()[0].detach().numpy()
    right_arm_qpos = right_arm_controller.qpos.cpu()[0].detach().numpy()
    gripper_l1_controller_qpos = gripper_l1_controller.qpos.cpu()[0].detach().numpy()
    gripper_r1_controller_qpos = gripper_r1_controller.qpos.cpu()[0].detach().numpy()
    gripper_l2_controller_qpos = gripper_l2_controller.qpos.cpu()[0].detach().numpy()
    gripper_r2_controller_qpos = gripper_r2_controller.qpos.cpu()[0].detach().numpy()
    lift_qpos = lift_controller.qpos.cpu()[0].detach().numpy()
    head_qpos = head_controller.qpos.cpu()[0].detach().numpy()
    base_qpos = base_controller.qpos.cpu()[0].detach().numpy()
    base_qpos =[0.0, 0.0, 0.0]
    all_qpos = {
        "arm_l": torch.tensor(left_arm_qpos, dtype=torch.float32),
        "arm_r": torch.tensor(right_arm_qpos, dtype=torch.float32),
        "gripper_l_1": torch.tensor([gripper_l1_controller_qpos[0]], dtype=torch.float32),
        "gripper_r_1": torch.tensor([gripper_r1_controller_qpos[0]], dtype=torch.float32),
        "gripper_l_2": torch.tensor([gripper_l2_controller_qpos[0]], dtype=torch.float32),
        "gripper_r_2": torch.tensor([gripper_r2_controller_qpos[0]], dtype=torch.float32),
        "lift": torch.tensor(lift_qpos, dtype=torch.float32),
        "head": torch.tensor(head_qpos, dtype=torch.float32),
        "base": torch.tensor(base_qpos, dtype=torch.float32)
    }

    flag = True
    def is_close(a, b, tol=0.1):
        return np.allclose(a, b, atol=tol)
    time_now = time.time()
    while True:
        current_qpos = left_arm_controller.qpos.cpu()[0].detach().numpy()

        if not is_close(current_qpos, left_arm_qpos, tol=1e-3):
            # 목표 상태에 도달하지 않았으면 제어 명령 전달
            env.step(controller.from_action_dict(all_qpos))
        else:
            # 목표 상태에 도달했으면 idle 상태 유지
            env.step(zero)  # 또는 생략도 가능

        env.render()

        # 3초마다 목표 변경
        if time.time() - time_now > 3:
            time_now = time.time()
            head_controller.reset()
            if flag:
                left_arm_qpos[0] = -1.1
                left_arm_qpos[1] = 1.0
                head_qpos[0] = -0.0
                head_qpos[1] = -0.0
                lift_qpos[0] = 0.25
                flag = False
            else:
                left_arm_qpos[0] = 1.1
                left_arm_qpos[1] = 0.0
                head_qpos[0] = 0.0
                head_qpos[1] = 0.0
                lift_qpos[0] = 0.0
                flag = True
            all_qpos.update({
                "arm_l": torch.tensor(left_arm_qpos, dtype=torch.float32),
                "head": torch.tensor(head_qpos, dtype=torch.float32),
                "lift": torch.tensor(lift_qpos, dtype=torch.float32),
            })


        # === 실시간 상태 출력 === 
    
    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
