import cv2
import math
import os
import urllib.request
from collections import deque

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_BASE_DIR, "models", "aggression")

_MODELS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "pose_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
}


def _ensure_models():
    os.makedirs(_MODEL_DIR, exist_ok=True)
    for filename, url in _MODELS.items():
        path = os.path.join(_MODEL_DIR, filename)
        if not os.path.exists(path):
            print(f"[AggressionAnalyzer] Downloading {filename}...")
            urllib.request.urlretrieve(url, path)
            print(f"[AggressionAnalyzer] {filename} ready.")


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def avg(values):
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def euclidean(p1, p2):
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))


class AggressionAnalyzer:
    def __init__(
        self,
        face_model_path=None,
        hand_model_path=None,
        pose_model_path=None,
        smooth_history=5
    ):
        _ensure_models()
        face_model_path = face_model_path or os.path.join(_MODEL_DIR, "face_landmarker.task")
        hand_model_path = hand_model_path or os.path.join(_MODEL_DIR, "hand_landmarker.task")
        pose_model_path = pose_model_path or os.path.join(_MODEL_DIR, "pose_landmarker.task")

        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = vision.RunningMode

        self.face_landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=face_model_path),
                running_mode=VisionRunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=True,
            )
        )

        self.hand_landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=hand_model_path),
                running_mode=VisionRunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

        self.pose_landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=pose_model_path),
                running_mode=VisionRunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False,
            )
        )

        self.face_hist = deque(maxlen=smooth_history)
        self.hand_hist = deque(maxlen=smooth_history)
        self.pose_hist = deque(maxlen=smooth_history)
        self.motion_hist = deque(maxlen=smooth_history)
        self.final_hist = deque(maxlen=smooth_history)

        self.left_hand_center_hist = deque(maxlen=smooth_history)
        self.right_hand_center_hist = deque(maxlen=smooth_history)

        self.prev_torso_center = None
        self.prev_timestamp_ms = None

    def _smooth(self, hist, value):
        hist.append(float(value))
        return avg(hist)

    def _hand_center(self, pts):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _compute_speed(self, hist, dt_sec):
        if len(hist) < 2 or dt_sec <= 1e-6:
            return 0.0
        return euclidean(hist[-1], hist[-2]) / dt_sec

    def analyze(self, person_bgr, timestamp_ms):
        if person_bgr is None or person_bgr.size == 0:
            return {
                "face_score": 0.0,
                "hand_score": 0.0,
                "pose_score": 0.0,
                "motion_score": 0.0,
                "aggression_score": 0.0,
                "state": "CALM",
                "debug": {}
            }

        if self.prev_timestamp_ms is None:
            dt_sec = 1.0 / 30.0
        else:
            dt_sec = max(1e-6, (timestamp_ms - self.prev_timestamp_ms) / 1000.0)
        self.prev_timestamp_ms = timestamp_ms

        rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        face_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        face_score_raw = 0.0
        hand_score_raw = 0.0
        pose_score_raw = 0.0
        motion_score_raw = 0.0

        brow_down = 0.0
        eye_squint = 0.0
        smile = 0.0
        lip_press = 0.0
        nose_sneer = 0.0
        angry_face_score = 0.0

        fist_count = 0
        has_fist = 0.0
        left_hand_speed_norm = 0.0
        right_hand_speed_norm = 0.0
        hand_movement_score = 0.0

        forward_lean = 0.0
        torso_center = None

        fist_angry_bonus_raw = 0.0
        fist_motion_bonus_raw = 0.0
        angry_motion_bonus_raw = 0.0

        WATCH_THRESHOLD = 0.20
        AGGRESSIVE_THRESHOLD = 0.50

        # ---- FACE ----
        if face_result.face_blendshapes:
            shapes = {x.category_name: float(x.score) for x in face_result.face_blendshapes[0]}

            brow_down = 0.5 * (
                shapes.get("browDownLeft", 0.0) +
                shapes.get("browDownRight", 0.0)
            )
            eye_squint = 0.5 * (
                shapes.get("eyeSquintLeft", 0.0) +
                shapes.get("eyeSquintRight", 0.0)
            )
            smile = 0.5 * (
                shapes.get("mouthSmileLeft", 0.0) +
                shapes.get("mouthSmileRight", 0.0)
            )
            lip_press = 0.5 * (
                shapes.get("mouthPressLeft", 0.0) +
                shapes.get("mouthPressRight", 0.0)
            )
            nose_sneer = 0.5 * (
                shapes.get("noseSneerLeft", 0.0) +
                shapes.get("noseSneerRight", 0.0)
            )

            angry_face_score = clamp(
                0.28 * brow_down +
                0.18 * eye_squint +
                0.24 * lip_press +
                0.12 * nose_sneer -
                0.18 * smile
            )

            face_score_raw = angry_face_score

        # ---- HANDS ----
        if hand_result.hand_landmarks:
            h, w = person_bgr.shape[:2]

            for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
                pts = [(lm.x * w, lm.y * h) for lm in hand_landmarks]

                wrist = pts[0]
                middle_mcp = pts[9]
                palm_size = euclidean(wrist, middle_mcp) + 1e-6
                palm_center = (
                    (wrist[0] + middle_mcp[0]) / 2.0,
                    (wrist[1] + middle_mcp[1]) / 2.0
                )

                curled = 0
                for idx in [8, 12, 16, 20]:
                    d = euclidean(pts[idx], palm_center) / palm_size
                    if d < 1.35:
                        curled += 1

                is_fist = curled >= 3
                if is_fist:
                    fist_count += 1

                hand_center = self._hand_center(pts)

                handedness_label = "Unknown"
                if i < len(hand_result.handedness) and len(hand_result.handedness[i]) > 0:
                    handedness_label = hand_result.handedness[i][0].category_name

                if handedness_label == "Left":
                    self.left_hand_center_hist.append(hand_center)
                elif handedness_label == "Right":
                    self.right_hand_center_hist.append(hand_center)
                else:
                    if len(self.left_hand_center_hist) <= len(self.right_hand_center_hist):
                        self.left_hand_center_hist.append(hand_center)
                    else:
                        self.right_hand_center_hist.append(hand_center)

            has_fist = 1.0 if fist_count >= 1 else 0.0

            left_hand_speed = self._compute_speed(self.left_hand_center_hist, dt_sec)
            right_hand_speed = self._compute_speed(self.right_hand_center_hist, dt_sec)

            left_hand_speed_norm = clamp(left_hand_speed / 900.0)
            right_hand_speed_norm = clamp(right_hand_speed / 900.0)
            hand_movement_score = max(left_hand_speed_norm, right_hand_speed_norm)

            # fist alone should be enough for WATCH, but not AGGRESSIVE
            hand_score_raw = clamp(
                0.50 * has_fist +
                0.50 * hand_movement_score
            )

        # ---- POSE + TORSO MOTION ----
        if pose_result.pose_landmarks:
            lm = pose_result.pose_landmarks[0]
            h, w = person_bgr.shape[:2]

            nose = (lm[0].x * w, lm[0].y * h)
            l_sh = (lm[11].x * w, lm[11].y * h)
            r_sh = (lm[12].x * w, lm[12].y * h)
            l_hip = (lm[23].x * w, lm[23].y * h)
            r_hip = (lm[24].x * w, lm[24].y * h)

            shoulder_center = (
                (l_sh[0] + r_sh[0]) / 2.0,
                (l_sh[1] + r_sh[1]) / 2.0
            )
            hip_center = (
                (l_hip[0] + r_hip[0]) / 2.0,
                (l_hip[1] + r_hip[1]) / 2.0
            )

            shoulder_width = euclidean(l_sh, r_sh) + 1e-6
            forward_lean = clamp(abs(nose[0] - hip_center[0]) / (0.9 * shoulder_width))
            pose_score_raw = forward_lean

            torso_center = (
                (shoulder_center[0] + hip_center[0]) / 2.0,
                (shoulder_center[1] + hip_center[1]) / 2.0
            )

            if self.prev_torso_center is not None:
                torso_speed = euclidean(torso_center, self.prev_torso_center) / dt_sec
                motion_score_raw = clamp(torso_speed / 800.0)

            self.prev_torso_center = torso_center

        face_score = self._smooth(self.face_hist, face_score_raw)
        hand_score = self._smooth(self.hand_hist, hand_score_raw)
        pose_score = self._smooth(self.pose_hist, pose_score_raw)
        motion_score = self._smooth(self.motion_hist, motion_score_raw)

        # ---- COMBO BONUSES ----
        # aggressive is intentionally stricter now

        # fist + angry face
        if has_fist and face_score >= 0.22:
            fist_angry_bonus_raw = clamp(0.30 * face_score)

        # fist + fast hand movement
        if has_fist and hand_movement_score >= 0.28:
            fist_motion_bonus_raw = clamp(0.38 * hand_movement_score)

        # angry face + fast hand movement
        if face_score >= 0.24 and hand_movement_score >= 0.24:
            angry_motion_bonus_raw = clamp(0.22 * min(face_score, hand_movement_score))

        # ---- FINAL SCORE ----
        base_score_raw = clamp(
            0.26 * face_score +
            0.30 * hand_score +
            0.14 * pose_score +
            0.08 * motion_score
        )

        combo_score_raw = clamp(
            fist_angry_bonus_raw +
            fist_motion_bonus_raw +
            angry_motion_bonus_raw
        )

        final_raw = clamp(base_score_raw + combo_score_raw)
        aggression_score = self._smooth(self.final_hist, final_raw)

        if aggression_score >= AGGRESSIVE_THRESHOLD:
            state = "AGGRESSIVE"
        elif aggression_score >= WATCH_THRESHOLD:
            state = "WATCH"
        else:
            state = "CALM"

        return {
            "face_score": face_score,
            "hand_score": hand_score,
            "pose_score": pose_score,
            "motion_score": motion_score,
            "aggression_score": aggression_score,
            "state": state,
            "debug": {
                "face_raw": face_score_raw,
                "hand_raw": hand_score_raw,
                "pose_raw": pose_score_raw,
                "motion_raw": motion_score_raw,
                "base_score_raw": base_score_raw,
                "combo_score_raw": combo_score_raw,
                "fist_angry_bonus_raw": fist_angry_bonus_raw,
                "fist_motion_bonus_raw": fist_motion_bonus_raw,
                "angry_motion_bonus_raw": angry_motion_bonus_raw,
                "brow_down": brow_down,
                "eye_squint": eye_squint,
                "lip_press": lip_press,
                "nose_sneer": nose_sneer,
                "smile": smile,
                "angry_face_score": angry_face_score,
                "fist_count": fist_count,
                "has_fist": has_fist,
                "left_hand_speed_norm": left_hand_speed_norm,
                "right_hand_speed_norm": right_hand_speed_norm,
                "hand_movement_score": hand_movement_score,
                "forward_lean": forward_lean,
                "torso_center": torso_center,
                "watch_threshold": WATCH_THRESHOLD,
                "aggressive_threshold": AGGRESSIVE_THRESHOLD,
            }
        }
