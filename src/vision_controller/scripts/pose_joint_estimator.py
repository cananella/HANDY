import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import time

# Mediapipe 초기화 (정밀도 향상을 위한 confidence 설정)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# 카메라 해상도 설정 (1280x720)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
FRAME_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 3D 시각화 설정
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.ion()

# 초기화 변수
origin_set = False
origin_point = None
init_start_time = None
INIT_DURATION = 3.0  # seconds
# 뼈 길이 저장 변수
L1 = None  # 어깨-팔꿈치
L2 = None  # 팔꿈치-손목

# 지수 이동 평균 필터 설정
alpha = 0.5  # 0 < alpha < 1, 높을수록 최신 값에 민감
filtered_joints = {}

# ========================
# 📌 좌표 변환 & 회전 계산
# ========================
def get_3d_point(landmarks, idx):
    lm = landmarks[idx]
    return np.array([lm.z, -lm.x, -lm.y])  # +X 정면, +Y 왼쪽, +Z 위

# 지수 이동 평균 적용 함수
def ema_filter(joint_id, current):
    if joint_id not in filtered_joints:
        filtered_joints[joint_id] = current
    else:
        filtered_joints[joint_id] = alpha * current + (1 - alpha) * filtered_joints[joint_id]
    return filtered_joints[joint_id]

# 벡터 정규화 함수
def get_joint_vector(joints, src_id, tgt_id):
    vec = joints[tgt_id] - joints[src_id]
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-6 else np.zeros(3)

# 회전 축과 각도 계산
def get_rotation_axis_angle(v1, v2):
    v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
    axis = np.cross(v1, v2)
    norm = np.linalg.norm(axis)
    axis = axis / norm if norm >= 1e-6 else np.array([1, 0, 0])
    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return axis, angle

# 쿼터니언 변환
def axis_angle_to_quaternion(axis, angle):
    half = angle / 2
    return np.array([np.cos(half), *(axis * np.sin(half))])

# 쿼터니언 -> Euler (roll, pitch, yaw) 변환 함수
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

# ========================
# 📦 박스 클래스
# ========================
class DraggableBox:
    def __init__(self, name, x, y, w, h, joint_ids):
        self.name, self.rect, self.joint_ids = name, [x, y, w, h], joint_ids
        self.drag = self.resizing = False
        self.offset = (0, 0)
        self.corner_size = 15
        self.inside = False
    def draw(self, frame):
        if origin_set: return
        x, y, w, h = self.rect
        color = (0, 255, 0) if self.inside else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, self.name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.rectangle(frame, (x + w - self.corner_size, y + h - self.corner_size), (x + w, y + h), (255,255,255), -1)
    def is_inside(self, px, py):
        x, y, w, h = self.rect
        return x <= px <= x + w and y <= py <= y + h
    def is_on_corner(self, px, py):
        x, y, w, h = self.rect
        return x + w - self.corner_size <= px <= x + w and y + h - self.corner_size <= py <= y + h
    def contains_landmark(self, landmarks, width, height):
        for i in self.joint_ids:
            pt=landmarks[i]; px,py=int(pt.x*width),int(pt.y*height)
            if not self.is_inside(px,py): self.inside=False; return False
        self.inside=True; return True

boxes = [
    DraggableBox("Shoulders", FRAME_W//2 - 180, FRAME_H//2 - 230, 360, 120, [11,12]),
    DraggableBox("Right Elbow", FRAME_W//2 + 105, FRAME_H//2 - 20, 100, 100, [13]),
    DraggableBox("Right Wrist", FRAME_W//2 + 105, FRAME_H//2 + 190, 100, 100, [15]),
    DraggableBox("Left Elbow", FRAME_W//2 - 205, FRAME_H//2 - 20, 100, 100, [14]),
    DraggableBox("Left Wrist", FRAME_W//2 - 205, FRAME_H//2 + 190, 100, 100, [16]),
]
cv2.namedWindow("Pose Estimation + Boxes")
cv2.setMouseCallback("Pose Estimation + Boxes", lambda *args: None)

def draw_axes(ax, center, scale=0.05):
    for vec, c in zip([np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])], ['r','g','b']):
        ax.quiver(center[0], center[1], center[2], vec[0], vec[1], vec[2], length=scale, color=c)
def plot_pose(points):
    ax.clear(); ax.set_xlim(-0.5,0.5); ax.set_ylim(0.5,-0.5); ax.set_zlim(-0.5,0.5)
    ax.set_title("Upper Body Pose"); ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    for (i,j) in [(11,13),(13,15),(12,14),(14,16),(11,12),(11,23),(12,24),(23,24)]:
        if i in points and j in points: pi,pj=points[i],points[j]; ax.plot([pi[0],pj[0]],[pi[1],pj[1]],[pi[2],pj[2]],'k')
    for pt in points.values(): draw_axes(ax,pt)
    plt.draw(); plt.pause(0.01)

# 메인 루프
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame,1)
    img_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    res = pose.process(img_rgb)
    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        center = sum((get_3d_point(lm,i) for i in [11,12,23,24]))/4
        if not origin_set:
            inside = all(b.contains_landmark(lm,FRAME_W,FRAME_H) for b in boxes)
            for b in boxes: b.draw(frame)
            if inside:
                if init_start_time is None: init_start_time=time.time()
                elapsed=time.time()-init_start_time
                cv2.putText(frame,f"Hold still: {elapsed:.1f}s/{INIT_DURATION:.1f}s",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
                if elapsed>=INIT_DURATION:
                    origin_point=center
                    L1=np.linalg.norm(get_3d_point(lm,13)-get_3d_point(lm,11))
                    L2=np.linalg.norm(get_3d_point(lm,15)-get_3d_point(lm,13))
                    origin_set=True
            else: init_start_time=None
        else:
            # joints 계산 및 필터
            joints = {i: ema_filter(i, get_3d_point(lm,i) - origin_point) for i in [11,12,13,14,15,16,23,24]}
            # 뼈 길이 제약
            if L1 and L2:
                for src,tgt,L in [(11,13,L1),(13,15,L2)]:
                    d = joints[tgt] - joints[src]; n=np.linalg.norm(d)
                    if n>1e-6: joints[tgt] = joints[src] + d/n * L
            plot_pose(joints)
            # 좌표 및 RPY 텍스트
            # 좌표계산
            r = {}
            # Left arm segments
            v1 = get_joint_vector(joints,11,13); q1=axis_angle_to_quaternion(*get_rotation_axis_angle(np.array([1,0,0]),v1)); r1=quaternion_to_euler(q1)
            v2 = get_joint_vector(joints,13,15); q2=axis_angle_to_quaternion(*get_rotation_axis_angle(np.array([1,0,0]),v2)); r2=quaternion_to_euler(q2)
            # Right arm segments
            v3 = get_joint_vector(joints,12,14); q3=axis_angle_to_quaternion(*get_rotation_axis_angle(np.array([1,0,0]),v3)); r3=quaternion_to_euler(q3)
            v4 = get_joint_vector(joints,14,16); q4=axis_angle_to_quaternion(*get_rotation_axis_angle(np.array([1,0,0]),v4)); r4=quaternion_to_euler(q4)
            # 각 포인트에 텍스트 그리기
            pts_info = [
                (11, 'LShoulder', joints[11], r1),
                (13, 'LElbow', joints[13], r2),
                (15, 'LWrist', joints[15], r2),
                (12, 'RShoulder', joints[12], r3),
                (14, 'RElbow', joints[14], r4),
                (16, 'RWrist', joints[16], r4),
            ]
            for idx,name,coord,euler in pts_info:
                x3,y3,z3 = coord
                roll,pitch,yaw = np.degrees(euler)
                px,py = int(lm[idx].x*FRAME_W), int(lm[idx].y*FRAME_H)
                cv2.putText(frame,f"{name} ({x3:.2f},{y3:.2f},{z3:.2f})",(px+5,py-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
                cv2.putText(frame,f"RPY({roll:.1f},{pitch:.1f},{yaw:.1f})",(px+5,py+10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
        mp_draw.draw_landmarks(frame,res.pose_landmarks,mp_pose.POSE_CONNECTIONS)
    cv2.imshow("Pose Estimation + Boxes",frame)
    if cv2.waitKey(1)==27: break
cap.release()
cv2.destroyAllWindows()
