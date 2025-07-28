import argparse
from typing import Annotated
import gymnasium as gym
import numpy as np
import sapien.core as sapien
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils import sapien_utils
import tyro
from dataclasses import dataclass

from mani_skill_core.robots.robotis_ai_worker import ai_worker 
from motionplanner_aiworker import AIWorkerMotionPlanningSolver  # ✅ 당신이 구현한 planner

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "Empty-v1"
    obs_mode: str = "none"
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "ai_worker"
    record_dir: str = "demos"
    save_video: bool = False
    viewer_shader: str = "rt-fast"
    video_saving_shader: str = "rt-fast"
    use_mplib: bool = False  # 🔁 추가: mplib 사용 여부 선택

def parse_args() -> Args:
    return tyro.cli(Args)

def compute_inverse_kinematics(env: BaseEnv, target_pose: sapien.Pose, arm: str = "left") -> np.ndarray:
    from scipy.optimize import minimize

    robot = env.agent.robot
    joint_names = env.agent.arm_l_joints if arm == "left" else env.agent.arm_r_joints
    link_name = f"arm_{arm[0]}_link7"
    link = robot.links_map[link_name]

    def fk(q):
        qpos_tensor = robot.get_qpos()
        qpos = qpos_tensor[0].cpu().numpy().copy()
        for i, name in enumerate(joint_names):
            idx = robot.find_joint_by_name(name).index.cpu().numpy()[0]
            qpos[idx] = q[i]
        robot.set_qpos(np.expand_dims(qpos, axis=0))
        pose = link.pose
        return pose

    def loss(q):
        pose = fk(q)
        pos_loss = np.linalg.norm(pose.p - target_pose.p)
        quat_diff = np.abs(np.dot(pose.q, target_pose.q))
        quat_loss = 1.0 - quat_diff**2
        return pos_loss + 0.1 * quat_loss

    qpos_full = robot.get_qpos()[0].cpu().numpy().copy()
    q_init = np.zeros(len(joint_names))
    for i, name in enumerate(joint_names):
        idx = int(robot.find_joint_by_name(name).index.item())
        q_init[i] = qpos_full[idx]

    result = minimize(loss, q_init, method="L-BFGS-B")
    if not result.success:
        return None

    return result.x

def main(args: Args):
    output_dir = f"{args.record_dir}/{args.env_id}/teleop/"
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="none",
        enable_shadow=True,
        robot_uids=args.robot_uid,
        viewer_camera_configs=dict(shader_pack=args.viewer_shader)
    )

    env = RecordEpisode(
        env,
        output_dir=output_dir,
        trajectory_name="trajectory",
        save_video=False,
        info_on_video=False,
        source_type="teleoperation",
        source_desc="teleoperation via gizmo system"
    )

    env.reset(seed=0)
    print("[INFO] Gizmo interface ready. Press 'h' for help.")
    solve(env, use_mplib=args.use_mplib)

def solve(env: BaseEnv, use_mplib=True, debug=False, vis=True):
    assert env.unwrapped.control_mode in ["pd_joint_pos"], "Only pd_joint_pos is supported"
    viewer = env.render_human()

    for plugin in viewer.plugins:
        if isinstance(plugin, sapien.utils.viewer.viewer.TransformWindow):
            transform_window = plugin
            break

    transform_window.enabled = True
    planner = None
    if use_mplib:
        planner = AIWorkerMotionPlanningSolver(env, debug=debug, vis=vis)

    while True:
        env.render_human()

        if viewer.window.key_press("h"):
            print("""
            h: show help
            n: plan & execute to ghost pose
            i: IK & PD control to ghost pose (no planner)
            a: dual-arm IK (mirror left to right)
            q: quit
            up/down/left/right/u/j: move ghost pose (Z/Y/X axes)
            """)
        elif viewer.window.key_press("q"):
            return
        elif viewer.window.key_press("n") and use_mplib:
            target_pose = transform_window._gizmo_pose
            print("[INFO] Planning to:", target_pose)
            result = planner.move_to_pose_with_screw(target_pose, dry_run=True)
            if result != -1 and len(result["position"]) < 150:
                _, reward, _, _, info = planner.follow_path(result)
                print(f"[EXEC] reward: {reward}, info: {info}")
            else:
                print("[WARN] Plan failed or too long.")
        elif viewer.window.key_press("i") and not use_mplib:
            target_pose = transform_window._gizmo_pose
            print("[INFO] Using PD control to:", target_pose)
            q_left = compute_inverse_kinematics(env, target_pose, arm="left")
            if q_left is not None:
                qpos = env.agent.robot.get_qpos()[0].cpu().numpy().copy()
                for i, name in enumerate(env.agent.arm_l_joints):
                    idx = int(env.agent.robot.find_joint_by_name(name).index.item())
                    qpos[idx] = q_left[i]
                for _ in range(20):
                    env.step(qpos[None, :24])
                    env.render_human()
            else:
                print("[WARN] Left-arm IK failed")
        elif viewer.window.key_press("a") and not use_mplib:
            target_pose = transform_window._gizmo_pose
            mirrored_pose = sapien.Pose(
                p=[target_pose.p[0], -target_pose.p[1], target_pose.p[2]],
                q=[target_pose.q[0], -target_pose.q[1], target_pose.q[2], -target_pose.q[3]]
            )
            q_l = compute_inverse_kinematics(env, target_pose, arm="left")
            q_r = compute_inverse_kinematics(env, mirrored_pose, arm="right")
            if q_l is not None and q_r is not None:
                qpos = env.agent.robot.get_qpos()[0].cpu().numpy().copy()
                for i, name in enumerate(env.agent.arm_l_joints):
                    idx = int(env.agent.robot.find_joint_by_name(name).index.item())
                    qpos[idx] = q_l[i]
                for i, name in enumerate(env.agent.arm_r_joints):
                    idx = int(env.agent.robot.find_joint_by_name(name).index.item())
                    qpos[idx] = q_r[i]
                for _ in range(20):
                    env.step(qpos[None, :24])
                    env.render_human()
            else:
                print("[WARN] One or both arm IK failed")
        elif viewer.window.key_press("u"):
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, 0, -0.01])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("j"):
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, 0, 0.01])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("left"):
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, 0.01, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("right"):
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, -0.01, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("up"):
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[-0.01, 0, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("down"):
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0.01, 0, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()

if __name__ == "__main__":
    main(parse_args())
