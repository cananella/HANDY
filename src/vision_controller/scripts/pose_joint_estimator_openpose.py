import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
from sys import platform
import os

from openpose import pyopenpose as op

params = {
    "model_folder": os.path.abspath("./models"),
    "model_pose": "BODY_25",
    "net_resolution": "-1x160",
    "scale_number": 1,
    "num_gpu_start": 0,
    "num_gpu": 1,
    "disable_multi_thread": True,
}
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# 시각화 세팅
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.ion()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
FRAME_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

origin_set = False
origin_point = None
init_start_time = None
INIT_DURATION = 3.0

# 지수 이동 평균 필터
alpha = 0.5
filtered_joints = {}

def ema_filter(joint_id, current):
    if joint_id not in filtered_joints:
        filtered_joints[joint_id] = current
    else:
        filtered_joints[joint_id] = alpha * current + (1 - alpha) * filtered_joints[joint_id]
    return filtered_joints[joint_id]

def get_joint_vector(joints, src_id, tgt_id):
    vec = joints[tgt_id] - joints[src_id]
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-6 else np.zeros(3)

def get_rotation_axis_angle(v1, v2):
    v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
    axis = np.cross(v1, v2)
    norm = np.linalg.norm(axis)
    axis = axis / norm if norm >= 1e-6 else np.array([1, 0, 0])
    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return axis, angle

def axis_angle_to_quaternion(axis, angle):
    half = angle / 2
    return np.array([np.cos(half), *(axis * np.sin(half))])

def quaternion_to_euler(q):
    w, x, y, z = q
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return roll, pitch, yaw

# OpenPose에서 사용되는 상반신 관련 keypoint 인덱스 (BODY_25 기준)
UPPER_BODY_IDS = {
    1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist',
    5: 'LShoulder', 6: 'LElbow', 7: 'LWrist'
}

def plot_pose(joints):
    ax.clear()
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0.5, -0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.set_title("Upper Body Pose")
    for (i, j) in [(1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7)]:
        if i in joints and j in joints:
            pi, pj = joints[i], joints[j]
            ax.plot([pi[0], pj[0]], [pi[1], pj[1]], [pi[2], pj[2]], 'k')
    for pt in joints.values():
        ax.scatter(pt[0], pt[1], pt[2], c='r')
    plt.draw()
    plt.pause(0.01)

def to_custom_coords(pt, frame_w, frame_h):
    x = (pt[0] - frame_w / 2) / frame_w  
    y = (pt[1] - frame_h / 2) / frame_h  
    z = 0  

    # 좌표계 변경: +X=앞, +Y=왼, +Z=위
    return np.array([z, -x, -y])  

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    
    datum = op.Datum()
    datum.cvInputData = frame

    datums_ptr = op.VectorDatum()
    datums_ptr.append(datum)

    opWrapper.emplaceAndPop(datums_ptr)
    show_frame = datum.cvOutputData
    pose_keypoints = datum.poseKeypoints
    if pose_keypoints is not None and len(pose_keypoints.shape) == 3:
        person = pose_keypoints[0]
        joints = {}
        for idx, name in UPPER_BODY_IDS.items():
            x, y, conf = person[idx]
            if conf > 0.3:
                joints[idx] = ema_filter(idx, to_custom_coords([x, y], FRAME_W, FRAME_H))

        if len(joints) >= 5:
            plot_pose(joints)
            for idx, coord in joints.items():
                name = UPPER_BODY_IDS[idx]
                px = int(person[idx][0])
                py = int(person[idx][1])
                cv2.putText(show_frame, f"{name}: ({coord[0]:.2f}, {coord[1]:.2f})", (px+10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    cv2.imshow("OpenPose Upper Body Tracking", show_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
