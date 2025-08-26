import gymnasium as gym
import time
import numpy as np
import torch
import cv2
from mani_skill.utils import common
from mani_skill.envs.sapien_env import BaseEnv
import ai_worker_custom

from mani_skill.utils import common, visualization
from mani_skill.agents.controllers.base_controller import DictController
from mani_skill.agents.controllers.base_controller import CombinedController
from mani_skill.agents.controllers.pd_joint_pos import PDJointPosController

env = gym.make(
    "Empty-v1",
    obs_mode="none",
    reward_mode="none",
    render_mode="human",
    robot_uids="ai_worker",
    control_mode="pd_joint_pos",
    sim_backend="auto",
    sim_config=dict(sim_freq=100, control_freq=20),
)
env.reset(seed=0)
env: BaseEnv = env.unwrapped
print("Selected Robot has the following keyframes to view: ")
print(env.agent.keyframes.keys())
qpos = env.agent.robot.get_qpos()
joint_names = env.agent.robot.get_active_joints()
print(len(qpos[0]))
print(len(joint_names))
# print(type(qpos[0]))
# print(type(joint_names))
# for key in joint_names:
#     print(f"{type(key)}")
qpos[0][4] = 0.4
kf = None
keyframe = "store_true"
if len(env.agent.keyframes) > 0:
    kf_name = None
    if keyframe is not None:
        kf_name = keyframe
        kf = env.agent.keyframes[kf_name]
    else:
        for kf_name, kf in env.agent.keyframes.items():
            # keep the first keyframe we find
            break
    if kf.qpos is not None:
        env.agent.robot.set_qpos(kf.qpos)
        env.agent.controller.reset()
    if kf.qvel is not None:
        env.agent.robot.set_qvel(kf.qvel)
    env.agent.robot.set_pose(kf.pose)
    if kf_name is not None:
        print(f"Viewing keyframe {kf_name}")
if env.gpu_sim_enabled:
    env.scene._gpu_apply_all()
    env.scene.px.gpu_update_articulation_kinematics()
    env.scene._gpu_fetch_all()
print(kf)

flag = True
def is_close(a, b, tol=1e-3):
    return np.allclose(a, b, atol=tol)
time_now = time.time()
while True:
    if True:
        env.render_human()

    if after_reset:
        after_reset = False
        # Re-focus on opencv viewer
        if True:
            renderer.close()
            renderer = visualization.ImageRenderer()
            pass

    if env.viewer.window.key_press("q"):
        break

env.close()
# kf = None
# if len(env.agent.keyframes) > 0:
#     kf_name = None
#     if args.keyframe is not None:
#         kf_name = args.keyframe
#         kf = env.agent.keyframes[kf_name]
#     else:
#         for kf_name, kf in env.agent.keyframes.items():
#             # keep the first keyframe we find
#             break
#     if kf.qpos is not None:
#         env.agent.robot.set_qpos(kf.qpos)
#         env.agent.controller.reset()
#     if kf.qvel is not None:
#         env.agent.robot.set_qvel(kf.qvel)
#     env.agent.robot.set_pose(kf.pose)
#     if kf_name is not None:
#         print(f"Viewing keyframe {kf_name}")
# if env.gpu_sim_enabled:
#     env.scene._gpu_apply_all()
#     env.scene.px.gpu_update_articulation_kinematics()
#     env.scene._gpu_fetch_all()
# viewer = env.render()
# viewer.paused = True
# viewer = env.render()




