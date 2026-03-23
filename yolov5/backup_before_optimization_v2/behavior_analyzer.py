import cv2
import mediapipe as mp
import numpy as np


class BehaviorAnalyzer:
    """Analyze body language and behavior for threats"""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.position_history = {}

    def analyze_person(self, frame, person_bbox, track_id):
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

        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return 0, []

        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_crop)

        if results.pose_landmarks:
            if self._has_raised_arms(results.pose_landmarks):
                score += 5
                behaviors.append("Raised arms")

            if self._is_lunging_forward(results.pose_landmarks):
                score += 4
                behaviors.append("Lunging forward")

            if self._is_running(results.pose_landmarks):
                score += 3
                behaviors.append("Running")

        velocity = self._calculate_velocity(person_bbox, track_id)
        if velocity > 2.0:
            score += 4
            behaviors.append(f"Fast approach ({velocity:.1f}m/s)")
        elif velocity > 1.0:
            score += 2
            behaviors.append(f"Moderate speed ({velocity:.1f}m/s)")

        if self._is_approaching(person_bbox, track_id):
            score += 3
            behaviors.append("Approaching")

        return min(score, 10), behaviors

    def _has_raised_arms(self, landmarks):
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]

        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]

        left_raised = left_wrist.y < left_shoulder.y
        right_raised = right_wrist.y < right_shoulder.y

        return left_raised and right_raised

    def _is_lunging_forward(self, landmarks):
        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]

        hip_center_y = (left_hip.y + right_hip.y) / 2
        forward_lean = nose.y < hip_center_y - 0.2

        return forward_lean

    def _is_running(self, landmarks):
        left_knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]

        knee_diff = abs(left_knee.y - right_knee.y)
        return knee_diff > 0.15

    def _calculate_velocity(self, bbox, track_id):
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        bbox_height = bbox[3] - bbox[1]

        if track_id not in self.position_history:
            self.position_history[track_id] = []

        self.position_history[track_id].append({
            "center": (center_x, center_y),
            "height": bbox_height
        })

        self.position_history[track_id] = self.position_history[track_id][-10:]

        if len(self.position_history[track_id]) < 5:
            return 0.0

        recent = self.position_history[track_id][-5:]
        start_pos = recent[0]["center"]
        end_pos = recent[-1]["center"]

        pixel_distance = np.sqrt(
            (end_pos[0] - start_pos[0]) ** 2 +
            (end_pos[1] - start_pos[1]) ** 2
        )

        avg_height = np.mean([p["height"] for p in recent])
        meters_per_pixel = 1.7 / avg_height if avg_height > 0 else 0.01

        distance_meters = pixel_distance * meters_per_pixel
        velocity = distance_meters / 0.5
        return velocity

    def _is_approaching(self, bbox, track_id):
        if track_id not in self.position_history:
            return False

        if len(self.position_history[track_id]) < 3:
            return False

        recent = self.position_history[track_id][-3:]
        first_height = recent[0]["height"]
        last_height = recent[-1]["height"]

        if first_height <= 0:
            return False

        growth = (last_height - first_height) / first_height
        return growth > 0.10

