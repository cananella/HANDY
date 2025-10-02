# aiworker_ik.py
import time
import numpy as np
import torch
from typing import Optional, Dict, Tuple, Union
import gymnasium as gym
import sapien.core as sapien
from transforms3d.quaternions import axangle2quat, quat2mat

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers.base_controller import CombinedController
from mani_skill.utils.structs import Articulation, Link
import custom_robot.ai_worker_custom 

# ------------------------ 유틸 ------------------------
def _to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(float)
    except Exception:
        pass
    return np.asarray(x, dtype=float)

def link_T_np(link: Link):
    T = _to_numpy(link.pose.to_transformation_matrix())
    if T.ndim == 3 and T.shape[0] == 1:
        T = T[0]
    assert T.shape == (4, 4), f"unexpected T shape: {T.shape}"
    return T

def link_R_np(link: Link):
    R = link_T_np(link)[:3, :3]
    assert R.shape == (3, 3)
    return R.astype(float)

def link_pos_np(link: Link):
    p = _to_numpy(link.pose.p)
    if p.ndim == 2 and p.shape[0] == 1 and p.shape[1] == 3:
        p = p[0]
    return p.reshape(3).astype(float)

def rotmat_from_quat_wxyz(qwxyz):
    q = np.asarray(qwxyz, dtype=float).reshape(4)
    return quat2mat(q)

def _as_R3(R):
    R = np.asarray(R, dtype=float)
    if R.ndim == 3 and R.shape[0] == 1:
        R = R[0]
    assert R.shape == (3, 3)
    return R

def ori_err_vec(R_cur, R_des):
    R_cur = _as_R3(R_cur); R_des = _as_R3(R_des)
    R = R_des.T @ R_cur
    tr = np.clip(np.trace(R), -1.0, 3.0)
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-6:
        v = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) * 0.5
        return v
    axis = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2.0 * np.sin(theta))
    return theta * axis

def parse_target_pose_optional_orientation(link: Link, target):
    # (pos, quat) | dict(pos,quat) | pos-only
    if isinstance(target, (tuple, list)) and len(target) == 2:
        pos, quat = target
        R = rotmat_from_quat_wxyz(quat)
        return np.asarray(pos, float).reshape(3), R
    if isinstance(target, dict):
        pos = target.get("pos", None)
        quat = target.get("quat", None)
        assert pos is not None, "target dict must contain 'pos'"
        R = rotmat_from_quat_wxyz(quat) if quat is not None else None
        return np.asarray(pos, float).reshape(3), R
    pos = np.asarray(target, float).reshape(3)
    return pos, link_R_np(link)

def _as_numpy_1d(q):
    is_torch = hasattr(q, "detach")
    arr = q.detach().cpu().numpy() if is_torch else np.asarray(q)
    batch_shape = None
    if arr.ndim == 2 and arr.shape[0] == 1:
        batch_shape = arr.shape
        arr = arr[0].copy()
    elif arr.ndim != 1:
        raise RuntimeError(f"Unexpected qpos shape: {arr.shape}")
    return arr, (is_torch, batch_shape, q)

def _to_original_shape_np(q1d: np.ndarray, ctx):
    is_torch, batch_shape, q_orig = ctx
    arr = q1d
    if batch_shape is not None:
        arr = arr.reshape(batch_shape)
    if is_torch:
        return torch.as_tensor(arr, dtype=q_orig.dtype, device=q_orig.device)
    return arr

def get_q_limits_np(robot: Articulation):
    lims = robot.get_qlimits()
    lims = lims.detach().cpu().numpy() if hasattr(lims, "detach") else np.asarray(lims)
    if lims.ndim == 3 and lims.shape[0] == 1:
        lims = lims[0]
    assert lims.ndim == 2 and lims.shape[1] == 2
    return lims[:, 0], lims[:, 1]

def build_name_to_qindex(robot: Articulation):
    name_to_idx = {}; idx = 0
    for j in robot.get_active_joints():
        for k in range(j.dof):
            name_to_idx[(j.get_name(), k)] = idx
            idx += 1
    return name_to_idx

def pick_group_indices_by_names(robot: Articulation, joint_names):
    name_to_idx = build_name_to_qindex(robot)
    out = []
    for nm in joint_names:
        key = (nm, 0)
        if key not in name_to_idx:
            raise RuntimeError(f"joint '{nm}' not found")
        out.append(name_to_idx[key])
    return np.array(out, dtype=int)

def get_group_q(robot: Articulation, group_idx: np.ndarray):
    q = robot.get_qpos()
    q1d, _ = _as_numpy_1d(q)
    return q1d[group_idx]

def set_group_q(robot: Articulation, group_idx: np.ndarray, qgroup: np.ndarray):
    q = robot.get_qpos()
    q1d, ctx = _as_numpy_1d(q)
    q1d[group_idx] = qgroup
    q_out = _to_original_shape_np(q1d, ctx)
    robot.set_qpos(q_out)
    try: robot.scene.update_render()
    except Exception: pass
    return q_out

def numerical_jacobian_error(err_func, q_group, eps=1e-4):
    e0 = err_func(q_group); m = e0.size; n = q_group.size
    J = np.zeros((m, n), float)
    for i in range(n):
        dq = np.zeros_like(q_group); dq[i] = eps
        e1 = err_func(q_group + dq)
        J[:, i] = (e1 - e0) / eps
    return J

# ------------------------ IK 핵심 ------------------------
def error_vec(robot: Articulation,
              q_group: np.ndarray,
              hand_target,
              elbow_target,
              arm_idx: np.ndarray,
              hand_link: Link,
              elbow_link: Link,
              w_pos: float = 1.0,
              w_ori: float = 1.0,
              w_elbow: float = 1.0):
    set_group_q(robot, arm_idx, q_group)
    ph = link_pos_np(hand_link)
    Rh = link_R_np(hand_link)
    pe = link_pos_np(elbow_link)

    ph_des, Rh_des = parse_target_pose_optional_orientation(hand_link, hand_target)
    pe_des = np.asarray(elbow_target, float).reshape(3)

    e_h_pos = ph_des - ph
    e_terms = [w_pos * e_h_pos]
    if Rh_des is not None:
        e_h_ori = ori_err_vec(Rh, Rh_des)
        e_terms.append(w_ori * e_h_ori)
    e_e_pos = pe_des - pe
    e_terms.append(w_elbow * e_e_pos)
    return np.hstack(e_terms).astype(float)

def ik_step(robot: Articulation,
            q_group: np.ndarray,
            hand_target,
            elbow_target,
            arm_idx: np.ndarray,
            hand_link: Link,
            elbow_link: Link,
            q_lo: np.ndarray,
            q_hi: np.ndarray,
            lam: float = 1e-3,
            alpha: float = 0.7,
            w_pos: float = 1.0,
            w_ori: float = 1.0,
            w_elbow: float = 1.0):
    err_func = lambda q: error_vec(robot, q, hand_target, elbow_target,
                                   arm_idx, hand_link, elbow_link,
                                   w_pos=w_pos, w_ori=w_ori, w_elbow=w_elbow)
    e = err_func(q_group).reshape(-1)
    J = numerical_jacobian_error(err_func, q_group)
    JT = J.T
    A = JT @ J + lam * np.eye(J.shape[1])
    b = JT @ e
    dq = np.linalg.solve(A, b)
    q_next = np.clip(q_group - alpha * dq, q_lo, q_hi)
    return q_next, float(np.linalg.norm(e))

def solve_arm(robot: Articulation,
              q_init: np.ndarray,
              hand_target,
              elbow_target,
              arm_idx: np.ndarray,
              hand_link: Link,
              elbow_link: Link,
              q_lo: np.ndarray,
              q_hi: np.ndarray,
              iters: int = 80,
              tol: float = 2e-3,
              lam: float = 1e-3,
              alpha: float = 0.7,
              w_pos: float = 1.0,
              w_ori: float = 1.0,
              w_elbow: float = 1.0):
    q = q_init.copy()
    best_q, best_e = q.copy(), 1e9
    for _ in range(iters):
        q, e = ik_step(robot, q, hand_target, elbow_target, arm_idx,
                       hand_link, elbow_link, q_lo, q_hi, lam, alpha,
                       w_pos=w_pos, w_ori=w_ori, w_elbow=w_elbow)
        if e < best_e:
            best_q, best_e = q.copy(), e
        if e < tol:
            break
    return best_q, best_e

def make_axis_gizmo(scene: sapien.Scene, scale=0.08, name="tcp_axis"):
    L = float(scale); r = L * 0.02; tip_r = r * 2
    red, green, blue = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    builder = scene.create_actor_builder()
    # +X
    builder.add_capsule_visual(pose=sapien.Pose([L/2, 0, 0]), radius=r, half_length=L/2, material=red)
    builder.add_sphere_visual(pose=sapien.Pose([L, 0, 0]), radius=tip_r, material=red)
    # +Y
    q_y = axangle2quat([0, 0, 1], np.pi/2)
    builder.add_capsule_visual(pose=sapien.Pose([0, L/2, 0], q=q_y), radius=r, half_length=L/2, material=green)
    builder.add_sphere_visual(pose=sapien.Pose([0, L, 0]), radius=tip_r, material=green)
    # +Z
    q_z = axangle2quat([0, 1, 0], -np.pi/2)
    builder.add_capsule_visual(pose=sapien.Pose([0, 0, L/2], q=q_z), radius=r, half_length=L/2, material=blue)
    builder.add_sphere_visual(pose=sapien.Pose([0, 0, L]), radius=tip_r, material=blue)
    return builder.build_kinematic(name=name)

# ------------------------ 공개 클래스 ------------------------
Target = Union[
    np.ndarray,                      # pos(3,)
    Tuple[np.ndarray, np.ndarray],   # (pos(3,), quat_wxyz(4,))
    Dict[str, np.ndarray],           # {"pos":(3,), "quat":(4,)}
]

class AIWorkerDLSController:
    """
    - env 생성/주입
    - 손목/팔꿈치 목표를 넣어 한 번의 IK로 qpos action_dict 반환
    """
    def __init__(
        self,
        env: Optional[BaseEnv] = None,
        *,
        create_env_kwargs: Optional[dict] = None,
        show_axes: bool = False,
        axis_scale: float = 0.3,
    ):
        if env is None:
            kw = dict(
                id="Empty-v1",
                obs_mode="none",
                reward_mode="none",
                render_mode="human",
                robot_uids="ffw",
                control_mode="pd_joint_pos",
                sim_backend="auto",
                sim_config=dict(sim_freq=100, control_freq=50),
            )
            if create_env_kwargs:
                kw.update(create_env_kwargs)
            _env = gym.make(**kw)
            _env.reset(seed=0)
            self.env: BaseEnv = _env.unwrapped
            self.viewer = self.env.render_human()
        else:
            self.env = env.unwrapped if hasattr(env, "unwrapped") else env
            self.viewer = getattr(self.env, "viewer", None)

        self.agent: BaseAgent = self.env.agent
        self.controller: CombinedController = self.env.agent.controller
        self.robot: Articulation = self.env.agent.robot

        # 조인트 이름(질문 코드 기준)
        self.left_arm_joint_names  = [f"arm_l_joint{i}" for i in range(1, 8)]
        self.right_arm_joint_names = [f"arm_r_joint{i}" for i in range(1, 8)]

        # 링크 핸들
        ln = {lk.name: lk for lk in self.robot.get_links()}
        self.l_hand = ln["left_hand_tcp"]
        self.r_hand = ln["right_hand_tcp"]
        self.l_elbow = ln["left_elbow_tcp"]
        self.r_elbow = ln["right_elbow_tcp"]

        # 인덱스/리밋
        self.left_arm_idx  = pick_group_indices_by_names(self.robot, self.left_arm_joint_names)
        self.right_arm_idx = pick_group_indices_by_names(self.robot, self.right_arm_joint_names)
        q_lo_all, q_hi_all = get_q_limits_np(self.robot)
        self.q_lo_l = q_lo_all[self.left_arm_idx]
        self.q_hi_l = q_hi_all[self.left_arm_idx]
        self.q_lo_r = q_lo_all[self.right_arm_idx]
        self.q_hi_r = q_hi_all[self.right_arm_idx]

        # 초기 qpos 0으로
        self.robot.set_qpos(self.robot.qpos * 0)

        # action dict 템플릿(초기화)
        self.action = self._read_action_from_controllers()

        # 축 기즈모(옵션)
        self.axes = None
        if show_axes:
            s = self.env.scene
            self.axes = dict(
                l_hand = make_axis_gizmo(s, axis_scale, "left_hand_tcp_axis"),
                r_hand = make_axis_gizmo(s, axis_scale, "right_hand_tcp_axis"),
                l_elbow= make_axis_gizmo(s, axis_scale, "left_elbow_tcp_axis"),
                r_elbow= make_axis_gizmo(s, axis_scale, "right_elbow_tcp_axis"),
            )

    def solve_once(
        self,
        left_hand: Target,
        left_elbow: np.ndarray,
        right_hand: Target,
        right_elbow: np.ndarray,
        *,
        iters: int = 80,
        tol: float = 2e-3,
        lam: float = 1e-3,
        alpha: float = 0.7,
        w_pos: float = 1.0,
        w_ori: float = 1.0,
        w_elbow: float = 1.0,
        split_pan: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        반환: controller.from_action_dict 에 바로 넣을 수 있는 dict(torch.Tensor)
        - split_pan=True 이면 7-DoF 중 마지막 1 DoF를 hand_pan으로 분리
        """
        q_l_now = get_group_q(self.robot, self.left_arm_idx)
        q_r_now = get_group_q(self.robot, self.right_arm_idx)

        q_l_sol, _ = solve_arm(self.robot, q_l_now, left_hand, left_elbow,
                               self.left_arm_idx, self.l_hand, self.l_elbow,
                               self.q_lo_l, self.q_hi_l,
                               iters=iters, tol=tol, lam=lam, alpha=alpha,
                               w_pos=w_pos, w_ori=w_ori, w_elbow=w_elbow)
        q_r_sol, _ = solve_arm(self.robot, q_r_now, right_hand, right_elbow,
                               self.right_arm_idx, self.r_hand, self.r_elbow,
                               self.q_lo_r, self.q_hi_r,
                               iters=iters, tol=tol, lam=lam, alpha=alpha,
                               w_pos=w_pos, w_ori=w_ori, w_elbow=w_elbow)

        if split_pan:
            l_arm, l_pan = q_l_sol[:6], q_l_sol[6:7]
            r_arm, r_pan = q_r_sol[:6], q_r_sol[6:7]
        else:
            l_arm, l_pan = q_l_sol, np.array([], dtype=float)
            r_arm, r_pan = q_r_sol, np.array([], dtype=float)

        out = self._read_action_from_controllers()  # 다른 그룹 값 유지
        out["arm_l"] = torch.tensor(l_arm, dtype=torch.float32)
        out["hand_l_pan"] = torch.tensor(l_pan, dtype=torch.float32) if l_pan.size else out["hand_l_pan"]
        out["arm_r"] = torch.tensor(r_arm, dtype=torch.float32)
        out["hand_r_pan"] = torch.tensor(r_pan, dtype=torch.float32) if r_pan.size else out["hand_r_pan"]
        return out

    def step_env(self, action_dict: Dict[str, torch.Tensor], render: bool = False):
        self.env.step(self.controller.from_action_dict(action_dict))
        if self.axes:
            self.axes["l_hand"].set_pose(self.l_hand.pose)
            self.axes["r_hand"].set_pose(self.r_hand.pose)
            self.axes["l_elbow"].set_pose(self.l_elbow.pose)
            self.axes["r_elbow"].set_pose(self.r_elbow.pose)
        if render:
            self.env.render()
    def draw_axes(self):
        if self.axes is None:
            return
        self.axes["l_hand"].set_pose(self.l_hand.pose)
        self.axes["r_hand"].set_pose(self.r_hand.pose)
        self.axes["l_elbow"].set_pose(self.l_elbow.pose)
        self.axes["r_elbow"].set_pose(self.r_elbow.pose)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

    def _read_action_from_controllers(self) -> Dict[str, torch.Tensor]:
        ctrl = self.controller.controllers
        def read(name):
            q = ctrl[name].qpos.cpu()[0].detach().numpy()
            return torch.tensor(q, dtype=torch.float32)
        # 개별 1-DoF 그리퍼는 길이 1 텐서로
        def read1(name):
            q = ctrl[name].qpos.cpu()[0].detach().numpy()
            return torch.tensor([q[0]], dtype=torch.float32)

        action = {
            "arm_l": read("arm_l"),
            "hand_l_pan": read("hand_l_pan"),
            "arm_r": read("arm_r"),
            "hand_r_pan": read("hand_r_pan"),
            "gripper_l_1": read1("gripper_l_1"),
            "gripper_r_1": read1("gripper_r_1"),
            "gripper_l_2": read1("gripper_l_2"),
            "gripper_r_2": read1("gripper_r_2"),
            "lift": read("lift"),
            "head": read("head"),
            "base": read("base"),
        }
        return action


if __name__ == "__main__":
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

    flag = True
    time_now = time.time()
    action = controller._read_action_from_controllers()
    while True:
        l_hand_now = controller.l_hand.pose.p
        l_elbow_now = controller.l_elbow.pose.p
        r_hand_now = controller.r_hand.pose.p
        r_elbow_now = controller.r_elbow.pose.p

        if time.time() - time_now > 3:
            time_now = time.time()
            if flag:
                flag = False
                target_l_hand = l_hand_now + np.array([0.10, 0.05, 0.05])
                target_l_elbow = l_elbow_now + np.array([0.00, -0.05, 0.05])

                target_r_hand = {"pos": first_r_hand, "quat": [1.0, 0.0, 0.0, 0.0]}
                target_r_elbow = r_elbow_now
            
            else:
                flag = True
                target_l_hand = {"pos": first_l_hand, "quat": [1.0, 0.0, 0.0, 0.0]}
                target_l_elbow = first_l_elbow

                target_r_hand = {"pos": first_r_hand, "quat": [0.707, 0.0, 0.0, -0.707]}
                target_r_elbow = first_r_elbow

        
            action = controller.solve_once(
                left_hand=target_l_hand,
                left_elbow=target_l_elbow,
                right_hand=target_r_hand,
                right_elbow=target_r_elbow)

        env.step(controller.controller.from_action_dict(action))
        controller.draw_axes()
        env.render()
        

# 2) env 스텝 (컨트롤러에 바로 들어갈 형식)
    controller.step_env(action, render=True)