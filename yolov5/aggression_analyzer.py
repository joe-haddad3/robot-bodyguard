
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

BASE_DIR = os.path.dirname(os.path.abspath(file_))
_MODEL_DIR = os.path.join(_BASE_DIR, "models", "aggression")

# Only face + hand — pose removed (BehaviorAnalyzer handles it)
_MODELS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
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

    def _init_(self, face_model_path=None, smooth_history=6):
        _ensure_models()
        face_model_path = face_model_path or os.path.join(_MODEL_DIR, "face_landmarker.task")

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

        # Per-instance history — never shared
        self.face_hist = deque(maxlen=smooth_history)
        self.final_hist = deque(maxlen=smooth_history)
        self.prev_timestamp_ms = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _smooth(self, hist, value):
        hist.append(float(value))
        return avg(hist)

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

        self.prev_timestamp_ms = timestamp_ms

        rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        face_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)

        # ---- FACE blendshapes ----
        brow_down = eye_squint = mouth_close = brow_squint_combo = 0.0
        brow_up = eye_wide = alert_combo = 0.0
        nose_sneer = upper_lip_raise = cheek_squint = mouth_frown = snarl_combo = 0.0
        brow_trigger = jaw_trigger = blueprint_trigger = snarl_trigger = alert_trigger = False
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
            # 3b. Alarm / intense look: brows up + eyes widened
            brow_up = max(
                shapes.get("browInnerUp", 0.0),
                0.5 * (
                    shapes.get("browOuterUpLeft", 0.0) + shapes.get("browOuterUpRight", 0.0)
                ),
            )
            eye_wide = 0.5 * (shapes.get("eyeWideLeft", 0.0) + shapes.get("eyeWideRight", 0.0))
            # 4. Snarl / disgust-like anger cues that show up when teeth are bared
            nose_sneer = 0.5 * (shapes.get("noseSneerLeft", 0.0) + shapes.get("noseSneerRight", 0.0))
            upper_lip_raise = 0.5 * (
                shapes.get("mouthUpperUpLeft", 0.0) + shapes.get("mouthUpperUpRight", 0.0)
            )
            cheek_squint = 0.5 * (
                shapes.get("cheekSquintLeft", 0.0) + shapes.get("cheekSquintRight", 0.0)
            )
            mouth_frown = 0.5 * (
                shapes.get("mouthFrownLeft", 0.0) + shapes.get("mouthFrownRight", 0.0)
            )

            # Combination bonus: brow frown + eye squint together → clear anger
            brow_squint_combo = brow_down * eye_squint   # product: both must be active
            snarl_combo = max(nose_sneer, upper_lip_raise) * max(cheek_squint, mouth_frown)
            alert_combo = brow_up * eye_wide

            # Explicit triggers reflect the intended blueprint more directly than
            # the weighted mix alone.
            brow_trigger = brow_down >= 0.18
            jaw_trigger = mouth_close >= 0.22
            blueprint_trigger = (
                brow_down >= 0.16
                and eye_squint >= 0.12
                and mouth_close >= 0.18
            )
            snarl_trigger = (
                (nose_sneer >= 0.12 or upper_lip_raise >= 0.16)
                and (cheek_squint >= 0.10 or mouth_frown >= 0.10)
            )
            alert_trigger = (
                brow_up >= 0.18
                and eye_wide >= 0.16
            )

            # Smile is intentionally NOT used — should have no effect on the score
            weighted_face_score = clamp(
                0.25 * brow_down          +   # classic angry brows
                0.12 * mouth_close        +   # clenched / shut jaw
                0.18 * brow_squint_combo  +   # frown + narrowed eyes
                0.08 * brow_up            +   # alarmed / intense brows
                0.08 * eye_wide           +   # widened eyes / alertness
                0.16 * nose_sneer         +   # snarling nose
                0.15 * upper_lip_raise    +   # teeth / upper lip raised
                0.08 * cheek_squint       +   # narrowed cheek / eye tension
                0.06 * mouth_frown        +   # downturned mouth corners
                0.20 * snarl_combo        +   # bared-teeth snarl pattern
                0.10 * alert_combo            # brows-up / eyes-wide intensity
            )
            trigger_floor = 0.0
            if brow_trigger:
                trigger_floor = max(trigger_floor, 0.38)
            if jaw_trigger:
                trigger_floor = max(trigger_floor, 0.34)
            if snarl_trigger:
                trigger_floor = max(trigger_floor, 0.60)
            if blueprint_trigger:
                trigger_floor = max(trigger_floor, 0.72)
            if alert_trigger:
                trigger_floor = max(trigger_floor, 0.24)

            angry_face_score = max(weighted_face_score, trigger_floor)
            face_score_raw = angry_face_score

        # ---- TEMPORAL SMOOTHING ----
        face_score = self._smooth(self.face_hist, face_score_raw)
        aggression_score = self._smooth(self.final_hist, face_score_raw)

        if face_score >= 0.36:
            state = "AGGRESSIVE"
        elif face_score >= 0.12:
            state = "WATCH"
        else:
            state = "CALM"

        return {
            "face_score": face_score,
            "aggression_score": aggression_score,
            "state": state,
            "debug": {
                "brow_down":         brow_down,
                "eye_squint":        eye_squint,
                "mouth_close":       mouth_close,
                "brow_squint_combo": brow_squint_combo,
                "brow_up":           brow_up,
                "eye_wide":          eye_wide,
                "alert_combo":       alert_combo,
                "nose_sneer":        nose_sneer,
                "upper_lip_raise":   upper_lip_raise,
                "cheek_squint":      cheek_squint,
                "mouth_frown":       mouth_frown,
                "snarl_combo":       snarl_combo,
                "angry_face_score":  angry_face_score,
                "brow_trigger":      brow_trigger,
                "jaw_trigger":       jaw_trigger,
                "blueprint_trigger": blueprint_trigger,
                "snarl_trigger":     snarl_trigger,
                "alert_trigger":     alert_trigger,
            },
        }