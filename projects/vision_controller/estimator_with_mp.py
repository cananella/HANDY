#!/usr/bin/env python3
"""
Instantiates an empty environment with a floor, places a robot, and controls it using human upper-body pose estimation (shoulder/elbow).
"""

import argparse
import cv2
import numpy as np
import gymnasium as gym
from mani_skill_core.robots.robotis_ai_worker import ai_worker
from mani_skill.agents.controllers.base_controller import DictController
from mani_skill.envs.sapien_env import BaseEnv
from vision_controller.scripts.pose_joint_estimator_mp import PoseController


def parse_args():
    parser = argparse.ArgumentParser(
        description="Empty env + human-pose-driven robot control"
    )
    parser.add_argument(
        "-r", "--robot-uid", type=str, default="ai_worker",
        help="ID of the robot to place"
    )
    parser.add_argument(
        "-b", "--sim-backend", type=str, default="auto",
        help="Simulation backend: auto / cpu / gpu"
    )
    parser.add_argument(
        "-c", "--control-mode", type=str, default="pd_joint_pos",
        help="Control mode for the robot"
    )
    parser.add_argument(
        "--sim-freq", type=int, default=100,
        help="Simulation frequency"
    )
    parser.add_argument(
        "--control-freq", type=int, default=20,
        help="Control frequency"
    )
    parser.add_argument(
        "--enable-plot", action="store_true",
        help="Show 3D pose visualization window"
    )
    parser.add_argument(
        "--random-actions", action="store_true",
        help="Sample random actions without pose"
    )
    parser.add_argument(
        "--none-actions", action="store_true",
        help="No actions, manual GUI control"
    )
    parser.add_argument(
        "--zero-actions", action="store_true",
        help="Send zero actions"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Initialize pose estimator and calibration
    pose_ctrl = PoseController(init_duration=3.0, alpha=0.5,
                                cam_id=0, width=1280, height=720,
                                enable_plot=args.enable_plot)
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
    init_sh, init_el = init_angles['left']
    init_sh_pitch = init_sh[1]
    print(f"Init angles: {init_angles} (shoulder pitch fixed at {init_sh_pitch:.2f})")

    # 2) Create and reset environment
    env = gym.make(
        "Empty-v1",
        obs_mode="none",
        reward_mode="none",
        enable_shadow=True,
        control_mode=args.control_mode,
        robot_uids=args.robot_uid,
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
        render_mode="human",
        sim_config=dict(sim_freq=args.sim_freq,
                        control_freq=args.control_freq),
        sim_backend=args.sim_backend,
    )
    obs, _ = env.reset()
    env: BaseEnv = env.unwrapped
    agent = env.agent

    # define fixed indices for left-shoulder and left-elbow in qpos
    shoulder_idx = 1  # fixed from init
    elbow_idx    = 2

    # initial align using fixed shoulder pitch
    qpos_init = agent.robot.qpos.clone().detach().cpu().numpy().flatten()
    qpos_init[shoulder_idx] = init_sh_pitch
    qpos_init[elbow_idx]    = init_el[1]
    init_action = (agent.controller.from_qpos(qpos_init)
                   if isinstance(agent.controller, DictController)
                   else qpos_init)
    env.step(init_action)
    env.render()
    print("Robot aligned to initial pose (shoulder fixed).")

    # 3) Main control loop
    while True:
        frame, angles = pose_ctrl.step()
        if frame is None:
            break
        cv2.imshow("Human Pose", frame)
        if cv2.waitKey(1) == 27:
            break

        # choose action
        if args.random_actions:
            action = env.action_space.sample()
        elif args.none_actions:
            action = None
        elif args.zero_actions:
            action = np.zeros_like(env.action_space.sample())
        elif angles:
            qpos = agent.robot.qpos.clone().detach().cpu().numpy().flatten()
            sh, el = angles['left']
            qpos[shoulder_idx] = init_sh_pitch
            qpos[elbow_idx]    = el[1]
            action = (agent.controller.from_qpos(qpos)
                      if isinstance(agent.controller, DictController)
                      else qpos)
        else:
            action = None

        env.step(action)
        env.render()

    cv2.destroyAllWindows()
    env.close()

if __name__ == "__main__":
    main()
