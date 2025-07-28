import gymnasium as gym
import numpy as np
import sapien.core as sapien
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils import sapien_utils
import mani_skill.trajectory.utils as trajectory_utils

from mani_skill_core.robots.robotis_ai_worker import ai_worker 
import h5py
import json
import os


def solve(env: BaseEnv):
    viewer = env.render_human()
    robot = env.agent.robot
    controller = env.agent.controller
    transform_window = None

    for plugin in viewer.plugins:
        if isinstance(plugin, sapien.utils.viewer.viewer.TransformWindow):
            transform_window = plugin
            break
    assert transform_window is not None, "TransformWindow plugin not found in viewer."

    # 조작 대상 링크 (왼팔 끝 링크)
    arm_link = robot.links_map["arm_l_link7"]
    viewer.select_entity(arm_link._objs[0].entity)

    print("🎮 조작 명령어:\n"
          "  n: 현재 pose로 IK 계획 후 실행\n"
          "  wasd / u / j: gizmo 이동\n"
          "  q: 종료 후 trajectory 저장\n"
          "  c: 현재 에피소드 저장 후 다음으로\n")

    trajectories = []
    actions = []

    while True:
        env.render_human()
        execute_plan = False

        if viewer.window.key_press("q"):
            return "quit", trajectories, actions
        elif viewer.window.key_press("c"):
            return "continue", trajectories, actions
        elif viewer.window.key_press("n"):
            execute_plan = True

        move_map = {
            "w": np.array([-0.01, 0, 0]),
            "s": np.array([+0.01, 0, 0]),
            "a": np.array([0, +0.01, 0]),
            "d": np.array([0, -0.01, 0]),
            "u": np.array([0, 0, +0.01]),
            "j": np.array([0, 0, -0.01]),
        }
        for key, offset in move_map.items():
            if viewer.window.key_press(key):
                pose = transform_window._gizmo_pose
                pose = pose * sapien.Pose(p=offset)
                transform_window.gizmo_matrix = pose.to_transformation_matrix()
                transform_window.update_ghost_objects()

        if execute_plan:
            # IK 기반 위치 추정
            pose = transform_window._gizmo_pose
            print(type(pose), pose)
            tcp_pose = pose.to_transformation_matrix()
            print(f"TCP Pose: {tcp_pose}  type: {type(tcp_pose)}")
            # 좌표계를 로봇 기준으로 변환 (예: base_link 기준)
            tcp_pose[:3, 3] -= robot.pose.p.cpu().numpy()


            # 조인트 추정 (사전 정의된 solver가 없으므로 임의 적용)
            from mani_skill.agents.ik_solver import IKSolver
            ik_solver = IKSolver(robot, active_joint_names=robot.active_joint_names)
            success, qpos = ik_solver.solve(tcp_pose)
            if not success:
                print("⚠️ IK 실패: pose에 도달할 수 없음")
                continue

            print(f"✅ IK 성공: 적용 qpos={qpos}")
            env.agent.controller.set_action(np.array([qpos]))
            env.step(np.array([qpos]))  # action은 shape (1, N)
            trajectories.append(env.get_state_dict())
            actions.append(qpos)


def main():
    env = gym.make(
        "Empty-v1",
        obs_mode="none",
        reward_mode="none",
        render_mode="rgb_array",
        robot_uids="ai_worker",
        control_mode="pd_joint_pos",
        sim_backend="cpu",
        sim_config=dict(sim_freq=100, control_freq=20),
        viewer_camera_configs=dict(shader_pack="rt-fast")
    )
    env = RecordEpisode(
        env,
        output_dir="demos/Empty-v1/teleop/",
        trajectory_name="trajectory",
        save_video=False,
        info_on_video=False,
        source_type="teleoperation",
        source_desc="teleoperation for ai_worker",
    )

    seed = 0
    env.reset(seed=seed)
    while True:
        code, traj, acts = solve(env)
        if code == "quit":
            break
        elif code == "continue":
            seed += 1
            env.reset(seed=seed)

    h5_file_path = env._h5_file.filename
    json_file_path = env._json_path
    print(f"💾 Trajectories saved to {h5_file_path}")

    env.close()
    del env


if __name__ == "__main__":
    main()
