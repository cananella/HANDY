import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import time


class DraggableBox:
    def __init__(self, name, x, y, w, h, joint_ids, corner_size=15):
        self.name = name
        self.rect = [x, y, w, h]
        self.joint_ids = joint_ids
        self.corner_size = corner_size
        self.inside = False

    def draw(self, frame, origin_set):
        if origin_set:
            return
        x, y, w, h = self.rect
        color = (0, 255, 0) if self.inside else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, self.name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # resize handle
        cv2.rectangle(
            frame,
            (x + w - self.corner_size, y + h - self.corner_size),
            (x + w, y + h),
            (255, 255, 255),
            -1
        )

    def contains(self, landmarks, width, height):
        for idx in self.joint_ids:
            lm = landmarks[idx]
            px, py = int(lm.x * width), int(lm.y * height)
            x, y, w, h = self.rect
            if not (x <= px <= x + w and y <= py <= y + h):
                self.inside = False
                return False
        self.inside = True
        return True


class PoseController:
    def __init__(self,
                 init_duration=3.0,
                 alpha=0.5,
                 cam_id=0,
                 width=1280,
                 height=720,
                 enable_plot=True):
        # 플롯 사용 여부
        self.enable_plot = enable_plot

        # Mediapipe 초기화
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # 카메라 설정
        self.cap = cv2.VideoCapture(cam_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.W, self.H = width, height

        # 3D 시각화 설정 (옵션)
        if self.enable_plot:
            self.fig = plt.figure(figsize=(2, 2))
            self.ax = self.fig.add_subplot(111, projection='3d')
            plt.ion()

        # 초기화 변수
        self.origin_set = False
        self.origin_point = None
        self.init_start = None
        self.INIT_DURATION = init_duration
        self.L1 = None
        self.L2 = None

        # EMA 필터
        self.alpha = alpha
        self.filtered = {}

        # 박스 영역
        cx, cy = width // 2, height // 2
        self.boxes = [
            DraggableBox("Shoulders", cx - 180, cy - 230, 360, 120, [11, 12]),
            DraggableBox("R Elbow", cx + 150, cy - 20, 100, 100, [13]),
            DraggableBox("R Wrist", cx + 150, cy + 190, 100, 100, [15]),
            DraggableBox("L Elbow", cx - 250, cy - 20, 100, 100, [14]),
            DraggableBox("L Wrist", cx - 250, cy + 190, 100, 100, [16]),
        ]

        cv2.namedWindow("Pose")
        cv2.setMouseCallback("Pose", lambda *args: None)

    # ──────────── Helpers ─────────────────────
    def _get_3d(self, lm, idx):
        p = lm[idx]
        return np.array([-p.z, -p.x, -p.y])  # +X front, +Y left, +Z up

    def _ema(self, idx, val):
        if idx not in self.filtered:
            self.filtered[idx] = val
        else:
            self.filtered[idx] = self.alpha * val + (1 - self.alpha) * self.filtered[idx]
        return self.filtered[idx]

    def _norm_vec(self, src, dst):
        v = dst - src
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else np.zeros(3)

    def _rot_axis_angle(self, v1, v2):
        u1, u2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
        axis = np.cross(u1, u2)
        na = np.linalg.norm(axis)
        axis = axis / na if na > 1e-6 else np.array([1, 0, 0])
        angle = np.arccos(np.clip(np.dot(u1, u2), -1, 1))
        return axis, angle

    def _axis_angle_to_quat(self, axis, ang):
        h = ang / 2
        return np.array([np.cos(h), *(axis * np.sin(h))])

    def _quat_to_euler(self, q):
        w, x, y, z = q
        t0 = 2 * (w * x + y * z)
        t1 = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(t0, t1)
        t2 = 2 * (w * y - z * x)
        t2 = np.clip(t2, -1, 1)
        pitch = np.arcsin(t2)
        t3 = 2 * (w * z + x * y)
        t4 = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        return roll, pitch, yaw

    def _draw_axes(self, center, scale=0.05):
        for vec, c in zip([np.array([1, 0, 0]),
                           np.array([0, 1, 0]),
                           np.array([0, 0, 1])],
                          ['r', 'g', 'b']):
            self.ax.quiver(center[0], center[1], center[2],
                           vec[0], vec[1], vec[2],
                           length=scale, color=c)

    def _plot3d(self, joints):
        self.ax.clear()
        self.ax.set_xlim(-0.5, 0.5)
        self.ax.set_ylim(0.5, -0.5)
        self.ax.set_zlim(-0.5, 0.5)
        edges = [(11, 13), (13, 15), (12, 14), (14, 16),
                 (11, 12), (11, 23), (12, 24), (23, 24)]
        for i, j in edges:
            if i in joints and j in joints:
                p1, p2 = joints[i], joints[j]
                self.ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]], 'k'
                )
        for pt in joints.values():
            self._draw_axes(pt)
        plt.draw()
        plt.pause(0.01)

    # ──────────── Main Loop ────────────────────
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.pose.process(rgb)

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                center = sum((self._get_3d(lm, i) for i in [11, 12, 23, 24])) / 4

                if not self.origin_set:
                    inside = all(b.contains(lm, self.W, self.H) for b in self.boxes)
                    for b in self.boxes:
                        b.draw(frame, self.origin_set)

                    if inside:
                        if self.init_start is None:
                            self.init_start = time.time()
                        elapsed = time.time() - self.init_start
                        cv2.putText(
                            frame,
                            f"Hold still: {elapsed:.1f}/{self.INIT_DURATION:.1f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )
                        if elapsed >= self.INIT_DURATION:
                            self.origin_point = center
                            self.L1 = np.linalg.norm(self._get_3d(lm, 13) - self._get_3d(lm, 11))
                            self.L2 = np.linalg.norm(self._get_3d(lm, 15) - self._get_3d(lm, 13))
                            self.origin_set = True
                    else:
                        self.init_start = None

                else:
                    joints = {
                        i: self._ema(i, self._get_3d(lm, i) - self.origin_point)
                        for i in [11, 12, 13, 14, 15, 16, 23, 24]
                    }
                    if self.L1 and self.L2:
                        for s, t, L in [(11, 13, self.L1), (13, 15, self.L2)]:
                            d = joints[t] - joints[s]
                            n = np.linalg.norm(d)
                            if n > 1e-6:
                                joints[t] = joints[s] + d / n * L

                    if self.enable_plot:
                        self._plot3d(joints)

                    segments = [(11, 13), (13, 15), (12, 14), (14, 16)]
                    names = ["LShoulder", "LElbow", "LWrist",
                             "RShoulder", "RElbow", "RWrist"]
                    idxs = [11, 13, 15, 12, 14, 16]
                    angles = []
                    for s, t in segments:
                        v = self._norm_vec(joints[s], joints[t])
                        axis, ang = self._rot_axis_angle(np.array([1, 0, 0]), v)
                        q = self._axis_angle_to_quat(axis, ang)
                        angles.append(self._quat_to_euler(q))

                    for name, idx_pt, euler in zip(names, idxs, angles):
                        coord = joints[idx_pt]
                        roll, pitch, yaw = np.degrees(euler)
                        px, py = int(lm[idx_pt].x * self.W), int(lm[idx_pt].y * self.H)
                        cv2.putText(
                            frame,
                            f"{name} ({coord[0]:.2f},{coord[1]:.2f},{coord[2]:.2f})",
                            (px + 5, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 255),
                            1
                        )
                        cv2.putText(
                            frame,
                            f"RPY({roll:.1f},{pitch:.1f},{yaw:.1f})",
                            (px + 5, py + 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 255),
                            1
                        )

                self.mp_draw.draw_landmarks(
                    frame,
                    res.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS
                )

            cv2.imshow("Pose", frame)
            if cv2.waitKey(1) == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def step(self):
        """
        Capture one frame, update pose estimation, and return the annotated image
        and arm angles once calibration is complete.
        Returns:
            frame (np.ndarray) or None: mirrored BGR image with overlays
            angles (dict or None): {'left': [(roll,pitch,yaw), (roll,pitch,yaw)],
                                     'right': [(roll,pitch,yaw), (roll,pitch,yaw)]}
                                     None until calibration done or no landmarks.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        angles_dict = None

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            center = sum((self._get_3d(lm, i) for i in [11, 12, 23, 24])) / 4

            # Calibration phase: user holds pose inside boxes
            if not self.origin_set:
                inside = all(b.contains(lm, self.W, self.H) for b in self.boxes)
                for b in self.boxes:
                    b.draw(frame, self.origin_set)
                if inside:
                    if self.init_start is None:
                        self.init_start = time.time()
                    elapsed = time.time() - self.init_start
                    cv2.putText(frame,
                                f"Hold still: {elapsed:.1f}/{self.INIT_DURATION:.1f}",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2)
                    if elapsed >= self.INIT_DURATION:
                        self.origin_point = center
                        self.L1 = np.linalg.norm(self._get_3d(lm, 13) - self._get_3d(lm, 11))
                        self.L2 = np.linalg.norm(self._get_3d(lm, 15) - self._get_3d(lm, 13))
                        self.origin_set = True
                else:
                    self.init_start = None
            else:
                # Compute filtered joint positions
                joints = {i: self._ema(i, self._get_3d(lm, i) - self.origin_point)
                          for i in [11, 12, 13, 14, 15, 16, 23, 24]}
                # Enforce segment lengths
                if self.L1 and self.L2:
                    for s, t, L in [(11, 13, self.L1), (13, 15, self.L2)]:
                        d = joints[t] - joints[s]
                        n = np.linalg.norm(d)
                        if n > 1e-6:
                            joints[t] = joints[s] + d / n * L
                # 3D plot
                if self.enable_plot:
                    self._plot3d(joints)
                # Compute arm angles
                segments = {'left': [(11, 13), (13, 15)],
                            'right': [(12, 14), (14, 16)]}
                angles_dict = {'left': [], 'right': []}
                for side, segs in segments.items():
                    for s, t in segs:
                        v = self._norm_vec(joints[s], joints[t])
                        axis, ang = self._rot_axis_angle(np.array([1, 0, 0]), v)
                        q = self._axis_angle_to_quat(axis, ang)
                        angles_dict[side].append(self._quat_to_euler(q))
                # Overlay text
                names = {'left': ['LShoulder', 'LElbow'],
                         'right': ['RShoulder', 'RElbow']}
                idxs = {'left': [11, 13], 'right': [12, 14]}
                for side in ['left', 'right']:
                    for name, idx_pt, euler in zip(names[side], idxs[side], angles_dict[side]):
                        coord = joints[idx_pt]
                        roll, pitch, yaw = np.degrees(euler)
                        px, py = int(lm[idx_pt].x * self.W), int(lm[idx_pt].y * self.H)
                        cv2.putText(frame,
                                    f"{name} ({coord[0]:.2f},{coord[1]:.2f},{coord[2]:.2f})",
                                    (px + 5, py - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4,
                                    (255, 255, 255),
                                    1)
                        cv2.putText(frame,
                                    f"RPY({roll:.1f},{pitch:.1f},{yaw:.1f})",
                                    (px + 5, py + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4,
                                    (255, 255, 255),
                                    1)
            # Draw landmarks
            self.mp_draw.draw_landmarks(
                frame,
                res.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS
            )
        return frame, angles_dict


if __name__ == "__main__":
    controller = PoseController(enable_plot=True)
    controller.run()
