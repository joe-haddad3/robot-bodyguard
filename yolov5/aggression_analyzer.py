"""
AggressionAnalyzer — face blendshapes + hand gesture analysis.

Design notes:
  - Pose/torso motion is intentionally REMOVED. BehaviorAnalyzer owns pose.
    This avoids loading two pose models and running duplicate inference.
  - This class is STATEFUL (keeps smoothing history). Create ONE instance
    per tracked person — do NOT share across persons.
  - Caller must pass real wall-clock milliseconds as timestamp_ms so that
    MediaPipe VIDEO mode temporal tracking works correctly.
    Use: int(time.time() * 1000)
"""

import cv2
import math
import os
import urllib.request
from collections import deque

import mediapipe as mp
from mediapipe.tasks.python import vision

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_BASE_DIR, "models", "aggression")

# Only face + hand — pose removed (BehaviorAnalyzer handles it)
_MODELS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
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
    return float(sum(values) / len(values)) if values else 0.0


def euclidean(p1, p2):
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))


class AggressionAnalyzer:
    """
    Per-person aggression analyzer (face expression + hand gestures).

    IMPORTANT: Create one instance per tracked person.
    Sharing a single instance across persons corrupts the smoothing history.
    """

    def __init__(self, face_model_path=None, hand_model_path=None, smooth_history=6):
        _ensure_models()
        face_model_path = face_model_path or os.path.join(_MODEL_DIR, "face_landmarker.task")
        hand_model_path = hand_model_path or os.path.join(_MODEL_DIR, "hand_landmarker.task")

        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = vision.RunningMode

        self.face_landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=face_model_path),
                running_mode=VisionRunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.4,
                min_face_presence_confidence=0.4,
                min_tracking_confidence=0.4,
                output_face_blendshapes=True,
            )
        )

        self.hand_landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=hand_model_path),
                running_mode=VisionRunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.4,
                min_hand_presence_confidence=0.4,
                min_tracking_confidence=0.4,
            )
        )

        # Per-instance history — never shared
        self.face_hist = deque(maxlen=smooth_history)
        self.hand_hist = deque(maxlen=smooth_history)
        self.final_hist = deque(maxlen=smooth_history)
        self.left_hand_center_hist = deque(maxlen=smooth_history)
        self.right_hand_center_hist = deque(maxlen=smooth_history)
        self.prev_timestamp_ms = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def analyze(self, person_bgr, timestamp_ms):
        """
        Analyze a person crop for aggression signals.

        Args:
            person_bgr  : BGR crop of the person bounding box.
            timestamp_ms: int(time.time() * 1000) — real wall-clock ms.
                          NEVER pass a counter; MediaPipe needs real time.

        Returns:
            dict with keys: face_score, hand_score, aggression_score, state, debug
        """
        _empty = {
            "face_score": 0.0,
            "hand_score": 0.0,
            "aggression_score": 0.0,
            "state": "CALM",
            "debug": {},
        }

        if person_bgr is None or person_bgr.size == 0:
            return _empty

        # Compute real elapsed time for speed calculations
        if self.prev_timestamp_ms is None:
            dt_sec = 1.0 / 15.0   # assume ~15 fps for the very first frame
        else:
            dt_sec = max(1e-4, (timestamp_ms - self.prev_timestamp_ms) / 1000.0)
        self.prev_timestamp_ms = timestamp_ms

        rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        face_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        # ---- FACE blendshapes ----
        brow_down = eye_squint = mouth_close = brow_squint_combo = 0.0
        angry_face_score = 0.0
        face_score_raw = 0.0

        if face_result.face_blendshapes:
            shapes = {
                x.category_name: float(x.score)
                for x in face_result.face_blendshapes[0]
            }

            # ── Three anger signals the user defined ────────────────────────
            # 1. Eyebrows frowning (pulling down)
            brow_down  = 0.5 * (shapes.get("browDownLeft",  0.0) + shapes.get("browDownRight", 0.0))
            # 2. Teeth / jaw clenching (lips pressed shut, jaw closed tight)
            mouth_close = shapes.get("mouthClose", 0.0)
            # 3. Eyes narrowing — only meaningful when brows are also frowning
            eye_squint = 0.5 * (shapes.get("eyeSquintLeft", 0.0) + shapes.get("eyeSquintRight", 0.0))

            # Combination bonus: brow frown + eye squint together → clear anger
            brow_squint_combo = brow_down * eye_squint   # product: both must be active

            # Smile is intentionally NOT used — should have no effect on the score
            angry_face_score = clamp(
                0.40 * brow_down        +   # primary: brows frowning alone
                0.30 * mouth_close      +   # secondary: teeth clenched / jaw shut
                0.30 * brow_squint_combo    # amplifier: frown + narrowed eyes
            )
            face_score_raw = angry_face_score

            # Expose these for the debug dict even though some old keys are removed
            brow_down   = brow_down
            eye_squint  = eye_squint
            smile       = 0.0   # kept for debug key parity; not used in scoring
            lip_press   = 0.0
            nose_sneer  = 0.0

        # ---- HANDS ----
        fist_count = 0
        has_fist = 0.0
        left_hand_speed_norm = right_hand_speed_norm = hand_movement_score = 0.0
        hand_score_raw = 0.0

        if hand_result.hand_landmarks:
            h, w = person_bgr.shape[:2]

            for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
                pts = [(lm.x * w, lm.y * h) for lm in hand_landmarks]

                wrist      = pts[0]
                middle_mcp = pts[9]
                palm_size  = euclidean(wrist, middle_mcp) + 1e-6
                palm_center = (
                    (wrist[0] + middle_mcp[0]) / 2.0,
                    (wrist[1] + middle_mcp[1]) / 2.0,
                )

                # Fingertip indices: index=8, middle=12, ring=16, pinky=20
                curled = sum(
                    1 for idx in [8, 12, 16, 20]
                    if euclidean(pts[idx], palm_center) / palm_size < 1.35
                )
                if curled >= 3:
                    fist_count += 1

                hand_center = self._hand_center(pts)

                handedness_label = "Unknown"
                if i < len(hand_result.handedness) and hand_result.handedness[i]:
                    handedness_label = hand_result.handedness[i][0].category_name

                if handedness_label == "Left":
                    self.left_hand_center_hist.append(hand_center)
                elif handedness_label == "Right":
                    self.right_hand_center_hist.append(hand_center)
                else:
                    # Unknown handedness — assign to whichever side has fewer entries
                    if len(self.left_hand_center_hist) <= len(self.right_hand_center_hist):
                        self.left_hand_center_hist.append(hand_center)
                    else:
                        self.right_hand_center_hist.append(hand_center)

            has_fist = 1.0 if fist_count >= 1 else 0.0

            left_speed  = self._compute_speed(self.left_hand_center_hist, dt_sec)
            right_speed = self._compute_speed(self.right_hand_center_hist, dt_sec)

            # 600 px/s = fully aggressive hand movement at typical RPi resolution
            left_hand_speed_norm  = clamp(left_speed  / 600.0)
            right_hand_speed_norm = clamp(right_speed / 600.0)
            hand_movement_score   = max(left_hand_speed_norm, right_hand_speed_norm)

            hand_score_raw = clamp(0.50 * has_fist + 0.50 * hand_movement_score)

        # ---- TEMPORAL SMOOTHING ----
        face_score = self._smooth(self.face_hist, face_score_raw)
        hand_score = self._smooth(self.hand_hist, hand_score_raw)

        # ---- COMBINATION BONUSES ----
        # Individual signals alone are weak; combinations are the real threat indicator.
        fist_angry_bonus  = 0.0
        fist_motion_bonus = 0.0
        angry_motion_bonus = 0.0

        if has_fist and face_score >= 0.22:
            fist_angry_bonus = clamp(0.35 * face_score)         # angry fist

        if has_fist and hand_movement_score >= 0.25:
            fist_motion_bonus = clamp(0.40 * hand_movement_score)  # fast fist swing

        if face_score >= 0.22 and hand_movement_score >= 0.22:
            angry_motion_bonus = clamp(0.25 * min(face_score, hand_movement_score))

        combo_score = clamp(fist_angry_bonus + fist_motion_bonus + angry_motion_bonus)

        # Face and hands weighted equally at base; combos push the score up
        base_score  = clamp(0.42 * face_score + 0.58 * hand_score)
        final_raw   = clamp(base_score + combo_score)
        aggression_score = self._smooth(self.final_hist, final_raw)

        WATCH_THRESHOLD      = 0.20
        AGGRESSIVE_THRESHOLD = 0.50

        if aggression_score >= AGGRESSIVE_THRESHOLD:
            state = "AGGRESSIVE"
        elif aggression_score >= WATCH_THRESHOLD:
            state = "WATCH"
        else:
            state = "CALM"

        return {
            "face_score": face_score,
            "hand_score": hand_score,
            "aggression_score": aggression_score,
            "state": state,
            "debug": {
                "brow_down":            brow_down,
                "eye_squint":           eye_squint,
                "mouth_close":          mouth_close,
                "brow_squint_combo":    brow_squint_combo,
                "angry_face_score":     angry_face_score,
                "fist_count":           fist_count,
                "has_fist":             has_fist,
                "hand_movement_score":  hand_movement_score,
                "left_hand_speed_norm": left_hand_speed_norm,
                "right_hand_speed_norm":right_hand_speed_norm,
                "base_score":           base_score,
                "combo_score":          combo_score,
                "fist_angry_bonus":     fist_angry_bonus,
                "fist_motion_bonus":    fist_motion_bonus,
                "angry_motion_bonus":   angry_motion_bonus,
            },
        }
