import cv2
import mediapipe as mp
import numpy as np
import time


class BehaviorAnalyzer:
    def __init__(
        self,
        pose_every_n=3,
        velocity_window=10,
        min_crop_size=80,
        pose_model_complexity=0,
    ):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=pose_model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4,
        )
        self.position_history = {}
        self.pose_every_n = pose_every_n
        self.velocity_window = velocity_window
        self.min_crop_size = min_crop_size
        self.pose_last_run = {}
        self.pose_cache = {}

    def analyze_person(self, frame, person_bbox, track_id, frame_index=None, run_pose=True):
        score = 0
        behaviors = []

        x1, y1, x2, y2 = [int(v) for v in person_bbox]
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            return 0, []

        bbox_w = x2 - x1
        bbox_h = y2 - y1

        velocity = self._calculate_velocity(person_bbox, track_id)
        if velocity > 2.0:
            score += 4
            behaviors.append(f"Fast approach ({velocity:.1f}m/s)")
        elif velocity > 1.0:
            score += 2
            behaviors.append(f"Moderate speed ({velocity:.1f}m/s)")

        if self._is_approaching(track_id):
            score += 3
            behaviors.append("Approaching")

        if not run_pose or bbox_w < self.min_crop_size or bbox_h < self.min_crop_size:
            return min(score, 10), behaviors

        should_run_pose = True
        if frame_index is not None:
            last_run = self.pose_last_run.get(track_id, -10**9)
            should_run_pose = (frame_index - last_run) >= self.pose_every_n

        if not should_run_pose and track_id in self.pose_cache:
            pose_score, pose_behaviors = self.pose_cache[track_id]
            score += pose_score
            behaviors.extend(pose_behaviors)
            return min(score, 10), behaviors

        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return min(score, 10), behaviors

        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_crop)

        pose_score = 0
        pose_behaviors = []

        if results.pose_landmarks:
            if self._has_raised_arms(results.pose_landmarks):
                pose_score += 5
                pose_behaviors.append("Raised arms")

            if self._is_lunging_forward(results.pose_landmarks):
                pose_score += 4
                pose_behaviors.append("Lunging forward")

            if self._is_running(results.pose_landmarks):
                pose_score += 3
                pose_behaviors.append("Running")

        self.pose_cache[track_id] = (pose_score, pose_behaviors)
        if frame_index is not None:
            self.pose_last_run[track_id] = frame_index

        score += pose_score
        behaviors.extend(pose_behaviors)
        return min(score, 10), behaviors

    def _has_raised_arms(self, landmarks):
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        return (left_wrist.y < left_shoulder.y) and (right_wrist.y < right_shoulder.y)

    def _is_lunging_forward(self, landmarks):
        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        hip_center_y = (left_hip.y + right_hip.y) / 2
        return nose.y < hip_center_y - 0.2

    def _is_running(self, landmarks):
        left_knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        return abs(left_knee.y - right_knee.y) > 0.15

    def _calculate_velocity(self, bbox, track_id):
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        bbox_height = max(1, bbox[3] - bbox[1])

        if track_id not in self.position_history:
            self.position_history[track_id] = []

        now = time.time()
        self.position_history[track_id].append(
            {"center": (center_x, center_y), "height": bbox_height, "t": now}
        )
        self.position_history[track_id] = self.position_history[track_id][-self.velocity_window:]

        if len(self.position_history[track_id]) < 4:
            return 0.0

        recent = self.position_history[track_id][-4:]
        start_pos = recent[0]["center"]
        end_pos = recent[-1]["center"]
        dt = max(1e-3, recent[-1]["t"] - recent[0]["t"])

        pixel_distance = np.sqrt(
            (end_pos[0] - start_pos[0]) ** 2 + (end_pos[1] - start_pos[1]) ** 2
        )
        avg_height = np.mean([p["height"] for p in recent])
        meters_per_pixel = 1.7 / avg_height if avg_height > 0 else 0.01
        distance_meters = pixel_distance * meters_per_pixel
        return distance_meters / dt

    def _is_approaching(self, track_id):
        if track_id not in self.position_history or len(self.position_history[track_id]) < 3:
            return False
        recent = self.position_history[track_id][-3:]
        first_height = recent[0]["height"]
        last_height = recent[-1]["height"]
        if first_height <= 0:
            return False
        growth = (last_height - first_height) / first_height
        return growth > 0.10
