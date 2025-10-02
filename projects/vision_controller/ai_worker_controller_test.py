import gymnasium as gym
import time
import numpy as np
import torch
import cv2
from mani_skill.utils import common
from mani_skill.envs.sapien_env import BaseEnv
import robot.ai_worker_custom
import sapien.core as sapien
import sapien.utils.viewer
from transforms3d.quaternions import axangle2quat, quat2mat
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers.base_controller import DictController
from mani_skill.agents.controllers.base_controller import CombinedController
from mani_skill.agents.controllers.pd_joint_pos import PDJointPosController
from mani_skill.utils.structs import Articulation, ArticulationJoint, Link

def _to_numpy(x):
    """torch.Tensor/list/np.array -> np.ndarray(float)"""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(float)
    except Exception:
        pass
    return np.asarray(x, dtype=float)

def link_T_np(link):
    """링크 4x4 변환행렬을 항상 (4,4) np.ndarray 로 반환"""
    T = _to_numpy(link.pose.to_transformation_matrix())
    # 배치(1,4,4) -> (4,4)
    if T.ndim == 3 and T.shape[0] == 1:
        T = T[0]
    assert T.shape == (4, 4), f"unexpected T shape: {T.shape}"
    return T

def link_R_np(link):
    """링크 회전행렬을 항상 (3,3) np.ndarray 로 반환"""
    T = link_T_np(link)
    R = T[:3, :3]
    assert R.shape == (3,3)
    return R.astype(float)

def link_pos_np(link):
    """링크 위치를 항상 (3,) np.ndarray 로 반환"""
    p = _to_numpy(link.pose.p)
    # 배치(1,3) -> (3,)
    if p.ndim == 2 and p.shape[0] == 1 and p.shape[1] == 3:
        p = p[0]
    p = p.reshape(3)
    return p.astype(float)

def rotmat_from_quat_wxyz(qwxyzw):
    """w,x,y,z -> 3x3 회전행렬"""
    q = np.asarray(qwxyzw, dtype=float).reshape(4)
    return quat2mat(q)  # transforms3d: w,x,y,z 약속

def _as_R3(R):
    R = np.asarray(R, dtype=float)
    if R.ndim == 3 and R.shape[0] == 1:
        R = R[0]
    assert R.shape == (3,3), f"R must be (3,3), got {R.shape}"
    return R

def ori_err_vec(R_cur, R_des):
    """
    오리엔테이션 오차의 3D 회전벡터 (theta*axis).
    여기서 R_err = R_des^T * R_cur (현재를 목표로 회전시키는 오차).
    """
    R_cur = _as_R3(R_cur)
    R_des = _as_R3(R_des)
    R = R_des.T @ R_cur
    tr = np.clip(np.trace(R), -1.0, 3.0)
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-6:
        v = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) * 0.5
        return v
    denom = 2.0 * np.sin(theta)
    axis = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / denom
    return theta * axis


def parse_target_pose_optional_orientation(link, target):
    # (pos, quat_wxyz) or dict
    if isinstance(target, (tuple, list)) and len(target) == 2:
        pos, quat = target
        R = rotmat_from_quat_wxyz(quat)   # (3,3)
        return np.asarray(pos, float).reshape(3), R
    if isinstance(target, dict):
        pos = target.get('pos', None)
        quat = target.get('quat', None)
        assert pos is not None, "target dict must contain 'pos'"
        R = rotmat_from_quat_wxyz(quat) if quat is not None else None
        return np.asarray(pos, float).reshape(3), R
    # pos만 온 경우: 현재 R 유지 (배치 제거된 (3,3))
    pos = np.asarray(target, float).reshape(3)
    R_cur = link_R_np(link)
    return pos, R_cur

def _as_numpy_1d(q):
    """robot.get_qpos() 결과를 (DOF,) numpy 로 평탄화 + 컨텍스트 저장"""
    is_torch = hasattr(q, "detach")
    if is_torch:
        arr = q.detach().cpu().numpy()
    else:
        arr = np.asarray(q)
    # 배치(1, DOF) → (DOF,)
    batch_shape = None
    if arr.ndim == 2 and arr.shape[0] == 1:
        batch_shape = arr.shape  # (1, DOF)
        arr = arr[0].copy()
    elif arr.ndim == 1:
        batch_shape = None
    else:
        # 다른 케이스가 있으면 필요에 맞게 처리
        raise RuntimeError(f"Unexpected qpos shape: {arr.shape}")
    return arr, (is_torch, batch_shape, q)

def _to_original_shape_np(q1d: np.ndarray, ctx):
    """(DOF,) numpy → 원래 set_qpos 가 기대하는 모양/타입으로 되돌리기"""
    is_torch, batch_shape, q_orig = ctx
    arr = q1d
    if batch_shape is not None:
        arr = arr.reshape(batch_shape)  # (1, DOF)
    if is_torch:
        return torch.as_tensor(arr, dtype=q_orig.dtype, device=q_orig.device)
    return arr

def get_q_limits_np(robot):
    """robot.get_qlimits()를 (dof,) numpy 쌍으로 반환"""
    lims = robot.get_qlimits()
    if hasattr(lims, "detach"):
        lims = lims.detach().cpu().numpy()
    else:
        lims = np.asarray(lims)
    if lims.ndim == 3 and lims.shape[0] == 1:
        lims = lims[0]
    assert lims.ndim == 2 and lims.shape[1] == 2, f"unexpected qlimit shape: {lims.shape}"
    return lims[:, 0], lims[:, 1]

# === (A) 로봇 조인트 인덱스 유틸 ===
def build_name_to_qindex(robot: Articulation):
    name_to_idx = {}
    idx = 0
    for j in robot.get_active_joints():
        dof = j.dof
        # 이 조인트가 차지하는 qpos 인덱스 범위
        for k in range(dof):
            name_to_idx[(j.get_name(), k)] = idx
            idx += 1
    return name_to_idx

def pick_group_indices_by_names(robot: Articulation, joint_names: list[str]):
    """ManiSkill 서브컨트롤러의 joint_names(조인트당 1 DoF 가정)를 전체 qpos 인덱스로 변환"""
    # 대부분 1-DoF 조인트라서 (name,0) 으로 매핑
    name_to_idx = build_name_to_qindex(robot)
    indices = []
    for nm in joint_names:
        key = (nm, 0)
        if key not in name_to_idx:
            raise RuntimeError(f"joint '{nm}' not found in robot")
        indices.append(name_to_idx[key])
    return np.array(indices, dtype=int)

def get_group_q(robot, group_idx: np.ndarray):
    q = robot.get_qpos()
    q1d, _ctx = _as_numpy_1d(q)
    return q1d[group_idx]

def set_group_q(robot, group_idx: np.ndarray, qgroup: np.ndarray):
    # 전체 q를 1D로 가져와 그룹 부분만 교체
    q = robot.get_qpos()
    q1d, ctx = _as_numpy_1d(q)
    q1d[group_idx] = qgroup
    q_out = _to_original_shape_np(q1d, ctx)
    robot.set_qpos(q_out)
    # 버전에 따라 아래가 scene 또는 env.scene일 수 있음
    try:
        robot.scene.update_render()
    except Exception:
        pass
    return q_out

def link_pos(link: Link):
    return link.pose.p

def numerical_jacobian_error(err_func, q_group, eps=1e-4):
    """
    err_func(q) -> (m,)  (pos/ori 오차 모두 포함)
    J_err[i] = (err(q+eps*e_i) - err(q)) / eps
    """
    e0 = err_func(q_group)
    m = e0.size
    n = q_group.size
    J = np.zeros((m, n), float)
    for i in range(n):
        dq = np.zeros_like(q_group)
        dq[i] = eps
        e1 = err_func(q_group + dq)
        J[:, i] = (e1 - e0) / eps
    return J

def damped_ls(J, err, lam=1e-3):
    J = np.asarray(J, dtype=float)
    err = np.asarray(err, dtype=float).reshape(-1)   # <<< 항상 (m,)
    JT = J.T
    A = JT @ J + lam * np.eye(J.shape[1])
    b = JT @ err
    return np.linalg.solve(A, b)


def make_axis_gizmo(scene: sapien.Scene, scale=0.08, name="tcp_axis"):
    L = float(scale)        # 축 길이
    r = L * 0.02            # 반지름
    tip_r = r * 2           # 끝 점 구 반지름

    red   = (1.0, 0.0, 0.0)
    green = (0.0, 1.0, 0.0)
    blue  = (0.0, 0.0, 1.0)

    builder = scene.create_actor_builder()  # 시각 전용(충돌 X)

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

# 왼팔 FK 함수: q_group(왼팔만) 넣으면 [hand(xyz), elbow(xyz)] 6D 벡터 반환
def error_vec(robot, q_group, hand_target, elbow_target, arm_idx ,hand, elbow):
    # q 반영
    set_group_q(robot, arm_idx, q_group)

    # 현재 상태 (np 보장)
    ph = link_pos_np(hand)
    Rh = link_R_np(hand)
    pe = link_pos_np(elbow)

    # 목표 파싱
    ph_des, Rh_des = parse_target_pose_optional_orientation(hand, hand_target)
    pe_des = np.asarray(elbow_target, float).reshape(3)

    # 위치 오차
    e_h_pos = ph_des - ph
    e_e_pos = pe_des - pe

    # 오리엔테이션 오차 (선택)
    if Rh_des is not None:
        e_h_ori = ori_err_vec(Rh, Rh_des)  # (3,)
        e = np.hstack([e_h_pos, e_h_ori, e_e_pos])
    else:
        e = np.hstack([e_h_pos, e_e_pos])
    return e.astype(float)

# --- IK 1스텝 (오차기반 DLS) ---
def ik_step(robot, q_group, hand_target, elbow_target, arm_idx, hand, elbow, q_lo, q_hi, lam=1e-3, alpha=0.7):
    err_func = lambda q: error_vec(robot, q, hand_target, elbow_target, arm_idx, hand, elbow)
    e = err_func(q_group).reshape(-1)
    J = numerical_jacobian_error(err_func, q_group)  # (m, n), m=6 or 9
    # DLS: q <- q - alpha * (J^T J + lambda I)^-1 J^T e
    JT = J.T
    A = JT @ J + lam * np.eye(J.shape[1])
    b = JT @ e
    dq = np.linalg.solve(A, b)
    q_next = q_group + (-alpha) * dq
    # 조인트 제한
    q_next = np.minimum(np.maximum(q_next, q_lo), q_hi)
    return q_next, float(np.linalg.norm(e))

def solve_arm_to_targets(robot, q_init, hand_target, elbow_target, arm_idx, hand, elbow, q_lo, q_hi, iters=80, tol=2e-3):
    q = q_init.copy()
    best = (q.copy(), 1e9)
    for _ in range(iters):
        q, e = ik_step(robot, q, hand_target, elbow_target, arm_idx, hand, elbow, q_lo, q_hi, lam=1e-3, alpha=0.7)
        if e < best[1]:
            best = (q.copy(), e)
        if e < tol:
            break
    return best[0], best[1]

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
    robot = env.agent.robot  # shorthand

    # 왼팔 서브컨트롤러의 조인트 이름 → 전체 qpos 인덱스
    left_arm_joint_names = [f"arm_l_joint{i}" for i in range(1, 8)]
    left_arm_idx = pick_group_indices_by_names(robot, left_arm_joint_names)

    right_arm_joint_names = [f"arm_r_joint{i}" for i in range(1, 8)]
    right_arm_idx = pick_group_indices_by_names(robot, right_arm_joint_names)

    # 왼팔 조인트 제한(전체에서 골라옴)
    q_lo_all, q_hi_all = get_q_limits_np(robot)
    q_lo_l = q_lo_all[left_arm_idx]
    q_hi_l = q_hi_all[left_arm_idx]

    q_lo_r = q_lo_all[right_arm_idx]
    q_hi_r = q_hi_all[right_arm_idx]



    l_hand_axis = make_axis_gizmo(env.scene, scale=0.3, name="left_hand_tcp_axis")
    r_hand_axis = make_axis_gizmo(env.scene, scale=0.3, name="right_hand_tcp_axis")
    l_elbow_axis = make_axis_gizmo(env.scene, scale=0.3, name="left_elbow_tcp_axis")
    r_elbow_axis = make_axis_gizmo(env.scene, scale=0.3, name="right_elbow_tcp_axis")

    flag = True
    time_now = time.time()
    while True:

        if time.time() - time_now > 3:
            time_now = time.time()

            l_hand_now = l_hand.pose.p
            l_elbow_now = l_elbow.pose.p
            r_hand_now = r_hand.pose.p
            r_elbow_now = r_elbow.pose.p

            if flag:
                target_l_hand = l_hand_now + np.array([0.10, 0.05, 0.05])   # +x,+y,+z 오프셋
                target_l_elbow = l_elbow_now + np.array([0.00, -0.05, 0.05])
                target_r_hand = {"pos" : r_hand_now + np.array([0.0, 0.0, 0.0]), "quat": [0.7071, 0, 0, -0.707]}  
                target_r_elbow = r_elbow_now
                flag = False
            else:
                target_l_hand = l_hand_now + np.array([-0.10, -0.05, -0.05])
                target_l_elbow = l_elbow_now + np.array([0.00, 0.05, -0.05])
                target_r_hand = {"pos" : r_hand_now + np.array([0.0, 0.0, 0.0]), "quat": [1.0, 0, 0.0, 0]}  
                target_r_elbow = r_elbow_now
                flag = True

            # 2) 현재 왼팔 q를 가져와 IK 초기값으로 사용
            q_l_now = get_group_q(robot, left_arm_idx)
            q_r_now = get_group_q(robot, right_arm_idx)

            # 3) IK로 왼팔 조인트 계산
            q_l_sol, err = solve_arm_to_targets(robot, q_l_now, target_l_hand, target_l_elbow, left_arm_idx, l_hand, l_elbow, q_lo_l, q_hi_l, iters=80, tol=2e-3)
            q_r_sol, err = solve_arm_to_targets(robot, q_r_now, target_r_hand, target_r_elbow, right_arm_idx, r_hand, r_elbow, q_lo_r, q_hi_r, iters=80, tol=2e-3)
            l_arm_sol = q_l_sol[:6]
            l_hand_pan_sol = q_l_sol[6:7]
            r_arm_sol = q_r_sol[:6]
            r_hand_pan_sol = q_r_sol[6:7]

            # 4) all_qpos에 반영 (나머지 그룹은 기존 값 유지)
            all_qpos["arm_l"] = torch.tensor(l_arm_sol, dtype=torch.float32)
            all_qpos["hand_l_pan"] = torch.tensor(l_hand_pan_sol, dtype=torch.float32)
            all_qpos["arm_r"] = torch.tensor(r_arm_sol, dtype=torch.float32)
            all_qpos["hand_r_pan"] = torch.tensor(r_hand_pan_sol, dtype=torch.float32)
            
        env.step(controller.from_action_dict(all_qpos))
        r_hand_tcp_pos = r_hand.pose
        l_hand_tcp_pos = l_hand.pose
        r_elbow_tcp_pos = r_elbow.pose
        l_elbow_tcp_pos = l_elbow.pose
        l_hand_axis.set_pose(l_hand_tcp_pos)
        r_hand_axis.set_pose(r_hand_tcp_pos)
        l_elbow_axis.set_pose(l_elbow_tcp_pos)
        r_elbow_axis.set_pose(r_elbow_tcp_pos)

        env.render()
    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
