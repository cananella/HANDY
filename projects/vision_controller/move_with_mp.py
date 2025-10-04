#!/usr/bin/env python3
"""
Instantiates an empty environment with a floor, places a robot, and controls it using human upper-body pose estimation (shoulder/elbow).
"""

import argparse
import cv2
import numpy as np
import gymnasium as gym
from custom_robot.ai_worker_custom import AIWorker
from mani_skill.agents.controllers.base_controller import DictController
from ai_worker_DLS_controller import AIWorkerDLSController
from mani_skill.envs.sapien_env import BaseEnv
from vision_controller.scripts.pose_joint_estimator_mp import PoseController


def main():

    pose_ctrl = PoseController(init_duration=3.0, alpha=0.5,
                                cam_id=0, width=1280, height=720,
                                enable_plot=False)
    print("Calibration: hold your pose in the boxes...")
    while not pose_ctrl.origin_set:
        frame, _ = pose_ctrl.step()
        if frame is None:
            print("Camera error during calibration.")
            return
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) == 27:
            print("Calibration aborted.")
            return
    cv2.destroyWindow("Calibration")
    print("Calibration done.")

    # capture initial human pose angles and fix shoulder pitch
    _, init_angles = pose_ctrl.step()
    init_sh, init_el, init_w = init_angles['left']
    init_sh_pitch = init_sh[1]
    print(f"Init angles: {init_angles} (shoulder pitch fixed at {init_sh_pitch:.2f})")

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
    viewer = env.render_human()
    env.agent.robot.set_qpos(env.agent.robot.qpos * 0)
    env.render()

    controller = AIWorkerDLSController(env, show_axes=True, axis_scale=0.2)
    first_l_hand = controller.l_hand.pose.p
    first_l_elbow = controller.l_elbow.pose.p
    first_r_hand = controller.r_hand.pose.p
    first_r_elbow = controller.r_elbow.pose.p

    action = controller._read_action_from_controllers()

    # 3) Main control loop
    while True:
        frame, lm_axis = pose_ctrl.step()
        if frame is None:
            break
        cv2.imshow("Human Pose", frame)
        if cv2.waitKey(1) == 27:
            break

        if lm_axis:
            l_shoulder, l_elbow, l_wrist = lm_axis['left']
            r_shoulder, r_elbow, r_wrist = lm_axis['right']
            if len(l_elbow) == 7 and len(r_elbow) == 7 and len(l_wrist) == 7 and len(r_wrist) == 7:
                target_l_hand = first_l_hand + np.array([l_wrist[0], l_wrist[1], l_wrist[2]])
                target_l_elbow = first_l_elbow + np.array([l_elbow[0], l_elbow[1], l_elbow[2]])
                target_r_hand = first_r_hand + np.array([r_wrist[0], r_wrist[1], r_wrist[2]])
                target_r_elbow = first_r_elbow + np.array([r_elbow[0], r_elbow[1], r_elbow[2]])

                action = controller.solve_once(
                    left_hand=target_l_hand, left_elbow=target_l_elbow,
                    right_hand=target_r_hand, right_elbow=target_r_elbow
                )
        env.step(controller.controller.from_action_dict(action))
        controller.draw_axes()
        env.render()

    cv2.destroyAllWindows()
    env.close()

if __name__ == "__main__":
    main()
