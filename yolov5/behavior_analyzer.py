import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import time
import urllib.request
import os

# Landmark indices (same values as the old PoseLandmark enum)
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_lite.task")


class BehaviorAnalyzer:
    def __init__(
        self,
        pose_every_n=3,
        velocity_window=10,
        min_crop_size=80,
        pose_model_complexity=0,  # kept for API compatibility, unused
    ):
        if not os.path.exists(_MODEL_PATH):
            print("[BehaviorAnalyzer] Downloading pose model (~5 MB)...")
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
            print("[BehaviorAnalyzer] Download complete.")

        base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.4,
        )
        self.detector = mp_vision.PoseLandmarker.create_from_options(options)
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
            pose_score, pose_behaviors, pose_landmarks = self.pose_cache[track_id]
            score += pose_score
            behaviors.extend(pose_behaviors)
            return min(score, 10), behaviors

        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return min(score, 10), behaviors

        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
        results = self.detector.detect(mp_image)

        pose_score = 0
        pose_behaviors = []

        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]  # first detected pose

            if self._has_raised_arms(landmarks):
                pose_score += 5
                pose_behaviors.append("Raised arms")

            if self._is_lunging_forward(landmarks):
                pose_score += 4
                pose_behaviors.append("Lunging forward")

            if self._is_running(landmarks):
                pose_score += 3
                pose_behaviors.append("Running")

        self.pose_cache[track_id] = (pose_score, pose_behaviors, landmarks if results.pose_landmarks else None)
        if frame_index is not None:
            self.pose_last_run[track_id] = frame_index

        score += pose_score
        behaviors.extend(pose_behaviors)
        return min(score, 10), behaviors

    def get_pose_landmarks(self, track_id):
        """Get the latest pose landmarks for a track_id if available"""
        if track_id in self.pose_cache:
            _, _, landmarks = self.pose_cache[track_id]
            return landmarks
        return None

    def _has_raised_arms(self, landmarks):
        return (
            landmarks[LEFT_WRIST].y < landmarks[LEFT_SHOULDER].y
            and landmarks[RIGHT_WRIST].y < landmarks[RIGHT_SHOULDER].y
        )

    def _is_lunging_forward(self, landmarks):
        nose = landmarks[NOSE]
        hip_center_y = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2
        return nose.y < hip_center_y - 0.2

    def _is_running(self, landmarks):
        return abs(landmarks[LEFT_KNEE].y - landmarks[RIGHT_KNEE].y) > 0.15

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
