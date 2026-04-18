import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import time
import urllib.request
import os
from collections import defaultdict, deque

# Landmark indices (same values as the old PoseLandmark enum)
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
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
        self.pose_flag_history = defaultdict(lambda: deque(maxlen=5))
        self.motion_history = defaultdict(
            lambda: {
                "torso": deque(maxlen=4),
                "left_wrist": deque(maxlen=4),
                "right_wrist": deque(maxlen=4),
                "left_span": deque(maxlen=4),
                "right_span": deque(maxlen=4),
            }
        )

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
            if track_id in self.pose_cache:
                entry = self.pose_cache[track_id]
                score += entry[0]
                behaviors.extend(entry[1])
            return min(score, 10), behaviors

        should_run_pose = True
        if frame_index is not None:
            last_run = self.pose_last_run.get(track_id, -10**9)
            should_run_pose = (frame_index - last_run) >= self.pose_every_n

        if not should_run_pose and track_id in self.pose_cache:
            entry = self.pose_cache[track_id]
            pose_score, pose_behaviors = entry[0], entry[1]
            score += pose_score
            behaviors.extend(pose_behaviors)
            return min(score, 10), behaviors

        # Expand the crop horizontally so MediaPipe sees arms that extend
        # beyond the YOLO bounding box.  Landmarks are normalised to this
        # expanded crop, so the stored bbox must match what MediaPipe saw.
        arm_pad_x = int(bbox_w * 0.70)   # 70 % of box width on each side
        arm_pad_y = int(bbox_h * 0.10)   # 10 % of box height top/bottom
        ex1 = max(0, x1 - arm_pad_x)
        ey1 = max(0, y1 - arm_pad_y)
        ex2 = min(w - 1, x2 + arm_pad_x)
        ey2 = min(h - 1, y2 + arm_pad_y)
        expanded_bbox = [float(ex1), float(ey1), float(ex2), float(ey2)]

        person_crop = frame[ey1:ey2, ex1:ex2]
        if person_crop.size == 0:
            return min(score, 10), behaviors

        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
        results = self.detector.detect(mp_image)

        pose_score = 0
        pose_behaviors = []
        landmarks = None
        raw_flags = self._empty_pose_flags()
        stable_flags = self._get_stable_flags(track_id)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]  # first detected pose
            raw_flags = self._compute_pose_signals(
                landmarks,
                pose_bbox=expanded_bbox,
                track_id=track_id,
            )
            stable_flags = self._update_pose_flags(track_id, raw_flags)

            if stable_flags["raised_arms"]:
                pose_score += 5
                pose_behaviors.append("Raised arms")
            elif stable_flags["arm_pose"]:
                pose_score += 3
                pose_behaviors.append("Arm raised/extended")

            if stable_flags["strike_windup"]:
                pose_score += 4
                pose_behaviors.append("Strike wind-up")

            if stable_flags["attack_motion"]:
                pose_score += 5
                pose_behaviors.append("Punching/slapping")

            if stable_flags["lunge"]:
                pose_score += 4
                pose_behaviors.append("Lunging forward")

            if stable_flags["running"]:
                pose_score += 3
                pose_behaviors.append("Running")
        else:
            stable_flags = self._update_pose_flags(track_id, raw_flags)

        # Store landmarks + the expanded bbox so callers can convert landmark
        # coordinates back to global pixel positions correctly.
        self.pose_cache[track_id] = (
            pose_score,
            pose_behaviors,
            landmarks,
            expanded_bbox,
            stable_flags,
            raw_flags,
        )
        if frame_index is not None:
            self.pose_last_run[track_id] = frame_index

        score += pose_score
        behaviors.extend(pose_behaviors)
        return min(score, 10), behaviors

    def get_pose_landmarks(self, track_id):
        """Get the latest pose landmarks for a track_id if available."""
        if track_id in self.pose_cache:
            return self.pose_cache[track_id][2]
        return None

    def get_pose_bbox(self, track_id):
        """Return the expanded crop bbox [x1,y1,x2,y2] used when pose was last computed.
        Landmark coordinates are normalised to this bbox, not the original YOLO box."""
        entry = self.pose_cache.get(track_id)
        if entry is not None and len(entry) >= 4:
            return entry[3]
        return None

    def get_pose_flags(self, track_id, stabilized=True):
        """Return the latest boolean pose flags for a track."""
        entry = self.pose_cache.get(track_id)
        if entry is None:
            return self._empty_pose_flags()
        idx = 4 if stabilized else 5
        if len(entry) > idx:
            return dict(entry[idx])
        return self._empty_pose_flags()

    def get_pose_signals(self, track_id, stabilized=True):
        """Alias for get_pose_flags() with a behavior-oriented name."""
        return self.get_pose_flags(track_id, stabilized=stabilized)

    def _empty_pose_flags(self):
        return {
            "raised_arms": False,
            "single_arm_high": False,
            "arm_pose": False,
            "forward_arm": False,
            "wrist_fast": False,
            "swinging_arm": False,
            "torso_lean": False,
            "crouch": False,
            "head_lowered": False,
            "strike_windup": False,
            "punch_detected": False,
            "attack_motion": False,
            "lunge": False,
            "running": False,
        }

    def _update_pose_flags(self, track_id, raw_flags):
        history = self.pose_flag_history[track_id]
        history.append(dict(raw_flags))
        if not history:
            return self._empty_pose_flags()

        stable = {}
        window = list(history)
        # quick_signals fire on a single positive frame (fast transients).
        # punch_detected / attack_motion are NOT quick — they must appear in
        # at least 2 of the last 5 frames so that a single noisy frame cannot
        # trigger a THREAT.  A real punch sustains the signal for 3–5 frames.
        quick_signals = {"single_arm_high", "wrist_fast", "swinging_arm", "strike_windup"}
        for key in self._empty_pose_flags():
            positives = sum(1 for item in window if item.get(key, False))
            if key in quick_signals:
                required = 1
            else:
                required = 2 if len(window) >= 3 else 1
            stable[key] = positives >= required
        return stable

    def _get_stable_flags(self, track_id):
        entry = self.pose_cache.get(track_id)
        if entry is not None and len(entry) >= 5:
            return dict(entry[4])
        return self._empty_pose_flags()

    def _visibility_ok(self, landmarks, indices, threshold=0.45):
        if any(idx >= len(landmarks) for idx in indices):
            return False
        return min(float(getattr(landmarks[idx], "visibility", 1.0)) for idx in indices) >= threshold

    def _landmark_to_global(self, landmark, pose_bbox):
        x1, y1, x2, y2 = pose_bbox
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        return (float(x1) + landmark.x * w, float(y1) + landmark.y * h)

    def _hip_center_y(self, landmarks, fallback_offset=0.32):
        hip_indices = [LEFT_HIP, RIGHT_HIP]
        if self._visibility_ok(landmarks, hip_indices, threshold=0.15):
            return (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2
        shoulder_y = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2
        return shoulder_y + fallback_offset

    def _has_raised_arms(self, landmarks):
        if not self._visibility_ok(
            landmarks,
            [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST],
        ):
            return False
        return (
            landmarks[LEFT_WRIST].y < landmarks[LEFT_SHOULDER].y
            and landmarks[RIGHT_WRIST].y < landmarks[RIGHT_SHOULDER].y
        )

    def _compute_pose_signals(self, landmarks, pose_bbox=None, track_id=None):
        shoulder_center_x = (landmarks[LEFT_SHOULDER].x + landmarks[RIGHT_SHOULDER].x) / 2
        shoulder_center_y = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2
        hip_center_x = (landmarks[LEFT_HIP].x + landmarks[RIGHT_HIP].x) / 2
        hip_center_y = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2
        knee_center_y = (landmarks[LEFT_KNEE].y + landmarks[RIGHT_KNEE].y) / 2
        nose = landmarks[NOSE]

        torso_height = hip_center_y - shoulder_center_y
        thigh_height = knee_center_y - hip_center_y
        head_gap = shoulder_center_y - nose.y
        torso_shift = abs(shoulder_center_x - hip_center_x)
        head_shift = abs(nose.x - hip_center_x)

        raised_arms = self._has_raised_arms(landmarks)
        single_arm_high = self._has_single_arm_high(landmarks)
        arm_pose = self._has_threatening_arm_pose(landmarks)
        forward_arm = self._has_forward_arm_extension(landmarks)
        wrist_fast = False
        swinging_arm = False
        punch_detected = False
        if pose_bbox is not None and track_id is not None:
            wrist_fast, swinging_arm, punch_detected = self._compute_attack_dynamics(
                track_id, landmarks, pose_bbox
            )
        torso_lean = torso_shift > 0.045 or head_shift > 0.08
        crouch = thigh_height > 0.0 and thigh_height < 0.24
        head_lowered = head_gap < 0.14
        strike_windup = arm_pose and (forward_arm or torso_lean)

        # attack_motion: require a genuine strike, not just raised/extended arms.
        #   punch_detected  — fast wrist + arm actively extending (straight punch / jab)
        #   swinging_arm    — fast wrist moving INWARD (hook / slap); outward raises blocked
        # forward_arm, single_arm_high, and strike_windup alone no longer suffice because
        # they all fire for a simple T-pose or arm raise.
        attack_motion = punch_detected or (wrist_fast and swinging_arm)

        lunge = crouch and torso_lean and (forward_arm or head_lowered or attack_motion)

        return {
            "raised_arms": raised_arms,
            "single_arm_high": single_arm_high,
            "arm_pose": arm_pose,
            "forward_arm": forward_arm,
            "wrist_fast": wrist_fast,
            "swinging_arm": swinging_arm,
            "torso_lean": torso_lean,
            "crouch": crouch,
            "head_lowered": head_lowered,
            "strike_windup": strike_windup,
            "punch_detected": punch_detected,
            "attack_motion": attack_motion,
            "lunge": lunge,
            "running": self._is_running(landmarks),
        }

    def _has_single_arm_high(self, landmarks):
        required = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST]
        if not self._visibility_ok(landmarks, required):
            return False
        return (
            landmarks[LEFT_WRIST].y < landmarks[LEFT_SHOULDER].y - 0.04
            or landmarks[RIGHT_WRIST].y < landmarks[RIGHT_SHOULDER].y - 0.04
        )

    def _has_threatening_arm_pose(self, landmarks):
        required = [
            LEFT_SHOULDER, RIGHT_SHOULDER,
            LEFT_ELBOW, RIGHT_ELBOW,
            LEFT_WRIST, RIGHT_WRIST,
        ]
        if not self._visibility_ok(landmarks, required):
            return False

        shoulder_center_x = (landmarks[LEFT_SHOULDER].x + landmarks[RIGHT_SHOULDER].x) / 2
        hip_center_y = self._hip_center_y(landmarks)

        def arm_active(shoulder_idx, elbow_idx, wrist_idx):
            shoulder = landmarks[shoulder_idx]
            elbow = landmarks[elbow_idx]
            wrist = landmarks[wrist_idx]
            if min(
                float(getattr(shoulder, "visibility", 1.0)),
                float(getattr(elbow, "visibility", 1.0)),
                float(getattr(wrist, "visibility", 1.0)),
            ) < 0.45:
                return False

            raised = wrist.y < shoulder.y + 0.05
            extended = abs(wrist.x - shoulder.x) > 0.16 and wrist.y < hip_center_y + 0.02
            punching_lane = (
                abs(wrist.x - shoulder_center_x) > 0.18
                and abs(wrist.y - elbow.y) < 0.16
            )
            return raised or extended or punching_lane

        return (
            arm_active(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
            or arm_active(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        )

    def _has_forward_arm_extension(self, landmarks):
        required = [
            LEFT_SHOULDER, RIGHT_SHOULDER,
            LEFT_ELBOW, RIGHT_ELBOW,
            LEFT_WRIST, RIGHT_WRIST,
        ]
        if not self._visibility_ok(landmarks, required):
            return False

        def extension(shoulder_idx, elbow_idx, wrist_idx):
            shoulder = landmarks[shoulder_idx]
            elbow = landmarks[elbow_idx]
            wrist = landmarks[wrist_idx]
            arm_len = np.hypot(wrist.x - shoulder.x, wrist.y - shoulder.y)
            forearm_straight = np.hypot(wrist.x - elbow.x, wrist.y - elbow.y)
            lateral_reach = abs(wrist.x - shoulder.x)
            return (
                arm_len > 0.22
                and forearm_straight > 0.10
                and lateral_reach > 0.18
                and abs(wrist.y - shoulder.y) < 0.18
            )

        return (
            extension(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
            or extension(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        )

    def _compute_attack_dynamics(self, track_id, landmarks, pose_bbox):
        history = self.motion_history[track_id]
        torso_hist = history["torso"]

        torso_point = None
        if self._visibility_ok(landmarks, [LEFT_HIP, RIGHT_HIP], threshold=0.15):
            l_hip = self._landmark_to_global(landmarks[LEFT_HIP], pose_bbox)
            r_hip = self._landmark_to_global(landmarks[RIGHT_HIP], pose_bbox)
            torso_point = ((l_hip[0] + r_hip[0]) / 2.0, (l_hip[1] + r_hip[1]) / 2.0)
        elif self._visibility_ok(landmarks, [LEFT_SHOULDER, RIGHT_SHOULDER], threshold=0.30):
            l_sh = self._landmark_to_global(landmarks[LEFT_SHOULDER], pose_bbox)
            r_sh = self._landmark_to_global(landmarks[RIGHT_SHOULDER], pose_bbox)
            torso_point = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)

        torso_dx = torso_dy = 0.0
        if torso_point is not None:
            if torso_hist:
                torso_dx = torso_point[0] - torso_hist[-1][0]
                torso_dy = torso_point[1] - torso_hist[-1][1]
            torso_hist.append(torso_point)

        l_sh = self._landmark_to_global(landmarks[LEFT_SHOULDER], pose_bbox)
        r_sh = self._landmark_to_global(landmarks[RIGHT_SHOULDER], pose_bbox)
        shoulder_width = max(40.0, abs(l_sh[0] - r_sh[0]))
        wrist_speed_thresh    = max(28.0, shoulder_width * 0.35)   # ~35 % shoulder_width / frame
        swing_thresh          = max(22.0, shoulder_width * 0.28)   # horizontal speed for hook/slap
        extension_gain_thresh = max(10.0, shoulder_width * 0.12)   # arm must be actively extending
        extension_len_thresh  = max(80.0, shoulder_width * 1.00)   # arm must be nearly fully extended

        any_fast = False
        any_swing = False
        any_punch = False

        for side, shoulder_idx, wrist_idx in (
            ("left", LEFT_SHOULDER, LEFT_WRIST),
            ("right", RIGHT_SHOULDER, RIGHT_WRIST),
        ):
            if not self._visibility_ok(landmarks, [shoulder_idx, wrist_idx], threshold=0.35):
                continue

            shoulder = self._landmark_to_global(landmarks[shoulder_idx], pose_bbox)
            wrist = self._landmark_to_global(landmarks[wrist_idx], pose_bbox)
            wrist_hist = history[f"{side}_wrist"]
            span_hist = history[f"{side}_span"]

            rel_dx = rel_dy = 0.0
            if wrist_hist:
                rel_dx = (wrist[0] - wrist_hist[-1][0]) - torso_dx
                rel_dy = (wrist[1] - wrist_hist[-1][1]) - torso_dy
            wrist_hist.append(wrist)

            arm_span = float(np.hypot(wrist[0] - shoulder[0], wrist[1] - shoulder[1]))
            extension_gain = 0.0
            if span_hist:
                extension_gain = arm_span - span_hist[-1]
            span_hist.append(arm_span)

            rel_speed = float(np.hypot(rel_dx, rel_dy))
            wrist_fast = rel_speed >= wrist_speed_thresh

            # Inward motion: right arm punches/slaps by moving LEFT (rel_dx < 0),
            # left arm by moving RIGHT (rel_dx > 0).  Outward arm raises are the
            # opposite sign and are blocked by this check.
            wrist_inward = (
                (side == "right" and rel_dx < 0)
                or (side == "left"  and rel_dx > 0)
            )

            swing_like = (
                abs(rel_dx) >= swing_thresh
                and abs(rel_dx) > abs(rel_dy) * 0.85
                and wrist[1] < shoulder[1] + shoulder_width * 0.9
                and wrist_inward          # must move toward center, not outward
            )

            # punch_like: straight punch / jab
            #   • wrist must be at or near shoulder height (not hanging at waist)
            #     — arm raises from waist put the wrist far below the shoulder
            #   • motion must be mostly horizontal (a punch goes FORWARD, not UP)
            #     — pure vertical lifts have large rel_dy and tiny rel_dx
            punch_like = (
                wrist_fast
                and arm_span >= extension_len_thresh
                and extension_gain >= extension_gain_thresh
                and wrist_inward
                and wrist[1] <= shoulder[1] + shoulder_width * 0.9   # wrist at/near shoulder height
                and abs(rel_dx) >= abs(rel_dy) * 0.5                  # at least 50 % horizontal
            )

            any_fast = any_fast or wrist_fast or punch_like
            any_swing = any_swing or swing_like
            any_punch = any_punch or punch_like

        return any_fast, any_swing, any_punch

    def _is_lunging_forward(self, landmarks):
        return self._compute_pose_signals(landmarks)["lunge"]

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
