"""
EnhancedThreatAnalyzer — combines behavior, aggression, weapons, and
identity into a single per-person threat score.

Key design:
  - One AggressionAnalyzer INSTANCE PER PERSON (keyed by track_id).
    Sharing a single instance was the primary source of wrong scores.
  - AggressionAnalyzer is rate-limited (aggression_every_n frames).
    Last result is cached so Phase 2 always has a score to show.
  - Real wall-clock timestamps are passed to AggressionAnalyzer so
    MediaPipe VIDEO mode works correctly.
  - BehaviorAnalyzer owns all pose inference — AggressionAnalyzer
    handles only face blendshapes + hand gestures.
"""

import math
import time
import numpy as np
from collections import defaultdict, deque

from behavior_analyzer import BehaviorAnalyzer
from aggression_analyzer import AggressionAnalyzer


class EnhancedThreatAnalyzer:

    def __init__(self, config=None):
        if config is None:
            from enhanced_tracking import BodyguardConfig
            config = BodyguardConfig()

        self.config = config
        self.rpi_mode = config.rpi_mode

        # ---- Threat / distance history (smoothing) ----
        self.threat_history = defaultdict(list)
        self.owner_distance_history = defaultdict(list)

        # ---- Per-person AggressionAnalyzer instances ----
        self.aggression_analyzers = {}   # track_id -> AggressionAnalyzer
        self.aggression_cache     = {}   # track_id -> last analyze() result
        self.aggression_last_run  = {}   # track_id -> last frame_index it ran

        # On RPi run aggression every 8 active frames; PC every 4
        self.aggression_every_n = 8 if self.rpi_mode else 4

        # ---- Wrist trajectory history for swing detection ----
        # Stores last N global (x,y) positions per wrist per track.
        self.wrist_history = defaultdict(lambda: {"left": deque(maxlen=5), "right": deque(maxlen=5)})

        # ---- BehaviorAnalyzer (owns pose inference) ----
        pose_every_n   = 20  if self.rpi_mode else 10
        min_crop_size  = 60  if self.rpi_mode else 100

        self.behavior_analyzer = BehaviorAnalyzer(
            pose_every_n=pose_every_n,
            velocity_window=10,
            min_crop_size=min_crop_size,
            pose_model_complexity=0,
        )

        # ---- Scoring weights ----
        self.IDENTITY_WEIGHTS = {
            "owner":   -20,
            "unknown":   3,
        }

        self.WEAPON_WEIGHTS = {
            "gun":          22,   # gun alone pushes past THREAT
            "knife":        18,   # knife alone pushes past THREAT
            "hammer":       12,
            "baseball_bat":  8,
        }

        self.WEAPON_THRESHOLDS = {
            "gun":          0.65,
            "knife":        0.50,
            "hammer":       0.50,
            "baseball_bat": 0.50,
        }

        # Classification thresholds
        self.SUSPICIOUS_THRESHOLD = 8
        self.THREAT_THRESHOLD     = 18

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_aggression_analyzer(self, track_id):
        """Return the per-person AggressionAnalyzer, creating it if new."""
        if track_id not in self.aggression_analyzers:
            self.aggression_analyzers[track_id] = AggressionAnalyzer()
        return self.aggression_analyzers[track_id]

    def _bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _distance(self, a, b):
        return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))

    def _point_near_bbox(self, point, bbox, margin=35):
        x, y = point
        x1, y1, x2, y2 = bbox
        return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)

    def _landmark_to_global(self, landmark, person_bbox):
        x1, y1, x2, y2 = person_bbox
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        return (float(x1) + landmark.x * w, float(y1) + landmark.y * h)

    def _is_attacking_owner(self, person_bbox, owner_bbox, pose_landmarks):
        if owner_bbox is None or not pose_landmarks:
            return False

        # Wrists entering the owner's space ~= swing / strike.
        # Ankles entering the owner's space ~= kick.
        for idx in (15, 16, 27, 28):
            if idx >= len(pose_landmarks):
                continue
            point = self._landmark_to_global(pose_landmarks[idx], person_bbox)
            if self._point_near_bbox(point, owner_bbox, margin=35):
                return True
        return False

    # ------------------------------------------------------------------
    # Track lifecycle
    # ------------------------------------------------------------------

    def cleanup_missing_tracks(self, active_track_ids):
        """Remove state for person tracks that no longer exist."""
        active = set(active_track_ids)

        for d in (
            self.threat_history,
            self.owner_distance_history,
            self.aggression_analyzers,
            self.aggression_cache,
            self.aggression_last_run,
            self.wrist_history,
        ):
            for tid in list(d.keys()):
                if tid not in active:
                    del d[tid]

    # ------------------------------------------------------------------
    # Aggression — runs independently from threat scoring
    # ------------------------------------------------------------------

    def update_aggression(self, frame, track_id, bbox, frame_index, person_moved=True):
        """
        Run aggression analysis (face blendshapes + hand gestures) for one
        person and cache the result.  Called as a SEPARATE step before
        analyze_person() so it has its own rate-limiting cadence.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        x1 = max(0, min(x1, fw - 1)); x2 = max(0, min(x2, fw - 1))
        y1 = max(0, min(y1, fh - 1)); y2 = max(0, min(y2, fh - 1))
        if x2 <= x1 or y2 <= y1:
            return

        has_cache  = track_id in self.aggression_cache
        should_run = (not has_cache) or (
            person_moved and (
                frame_index is None or
                (frame_index - self.aggression_last_run.get(track_id, -9999)) >= self.aggression_every_n
            )
        )
        if not should_run:
            return

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        analyzer     = self._get_aggression_analyzer(track_id)
        timestamp_ms = int(time.time() * 1000)
        result       = analyzer.analyze(crop, timestamp_ms)
        self.aggression_cache[track_id] = result
        if frame_index is not None:
            self.aggression_last_run[track_id] = frame_index

    # ------------------------------------------------------------------
    # Threat helpers
    # ------------------------------------------------------------------

    def _box_to_box_distance(self, box1, box2):
        """Pixel distance between the nearest edges of two bounding boxes.
        Returns 0 if they overlap."""
        x1a, y1a, x2a, y2a = box1
        x1b, y1b, x2b, y2b = box2
        dx = max(0.0, max(x1a, x1b) - min(x2a, x2b))
        dy = max(0.0, max(y1a, y1b) - min(y2a, y2b))
        return math.sqrt(dx * dx + dy * dy)

    def _weapon_inside_owner(self, weapons, owner_bbox):
        """True if any weapon box OVERLAPS (is inside) the owner box (edge-to-edge distance == 0)."""
        if owner_bbox is None or not weapons:
            return False
        for w in weapons:
            wbox = w.get("bbox")
            if wbox is None:
                continue
            if self._box_to_box_distance(wbox, owner_bbox) == 0:
                return True
        return False

    def _weapon_near_owner(self, weapons, owner_bbox, threshold_px=150):
        """True if any weapon box is within threshold_px of the owner box but NOT overlapping."""
        if owner_bbox is None or not weapons:
            return False
        for w in weapons:
            wbox = w.get("bbox")
            if wbox is None:
                continue
            d = self._box_to_box_distance(wbox, owner_bbox)
            if 0 < d < threshold_px:
                return True
        return False

    def _detect_swing_toward_owner(self, track_id, person_bbox, owner_bbox, pose_landmarks,
                                    proximity_px=200):
        """
        Detect a swinging motion directed at the owner.

        Three conditions must all be true:
          1. Wrist speed >= 25 px over 2 frames (fast movement).
          2. Movement direction has cosine similarity >= 0.6 toward owner centre.
          3. Wrist is already within proximity_px of the owner box edge (so a swing
             from across the room does not count — the hand must be near the owner).

        Wrist positions accumulate in wrist_history every call.
        """
        if owner_bbox is None or not pose_landmarks or len(pose_landmarks) < 17:
            return False

        owner_cx = (owner_bbox[0] + owner_bbox[2]) / 2.0
        owner_cy = (owner_bbox[1] + owner_bbox[3]) / 2.0

        hist = self.wrist_history[track_id]
        for side, idx in (("left", 15), ("right", 16)):
            if idx >= len(pose_landmarks):
                continue
            gx, gy = self._landmark_to_global(pose_landmarks[idx], person_bbox)
            hist[side].append((gx, gy))

            positions = hist[side]
            if len(positions) < 3:
                continue

            # Condition 3: wrist must be near the owner box already
            # Use a synthetic 1-px box around the wrist to reuse _box_to_box_distance
            wrist_box = (gx, gy, gx + 1, gy + 1)
            if self._box_to_box_distance(wrist_box, owner_bbox) > proximity_px:
                continue

            # Condition 1: fast displacement over 2 frames
            dx = positions[-1][0] - positions[-3][0]
            dy = positions[-1][1] - positions[-3][1]
            speed = math.sqrt(dx * dx + dy * dy)
            if speed < 25.0:
                continue

            # Condition 2: direction toward owner centre
            to_ox = owner_cx - positions[-1][0]
            to_oy = owner_cy - positions[-1][1]
            dist_to_owner = math.sqrt(to_ox * to_ox + to_oy * to_oy)
            if dist_to_owner < 1.0:
                continue

            cos_sim = (dx * to_ox + dy * to_oy) / (speed * dist_to_owner)
            if cos_sim >= 0.6:
                return True

        return False

    # ------------------------------------------------------------------
    # Main scoring
    # ------------------------------------------------------------------

    def analyze_person(self, frame, person, context, frame_index=None, person_moved=True):
        """
        Threat ruleset (aggression is pre-computed by update_aggression()):

        1. Physical attack: wrist/ankle enters owner space           → THREAT
        2. Swinging motion directed at owner                         → THREAT
        3. Weapon box within 150 px of owner box                     → THREAT
        4. Weapon held but not yet near owner                        → SUSPICIOUS
        5. Angry face (no weapon)                                    → SUSPICIOUS
        6. Everything else                                           → SAFE

        Returns: (level, score, explanation, debug)
        """
        try:
            track_id = person["id"]
            identity = person.get("identity", "unknown")
            bbox     = person["bbox"]
            weapons  = person.get("weapons", [])

            if identity == "owner":
                return "SAFE", 0.0, ["OWNER"], {}

            # ── Weapon presence ──────────────────────────────────────────────
            has_weapon      = False
            strongest_weapon = None
            for w in weapons:
                cls  = w["class_name"]
                conf = float(w["conf"])
                if cls in self.WEAPON_WEIGHTS and conf >= self.WEAPON_THRESHOLDS.get(cls, 0.50):
                    has_weapon      = True
                    strongest_weapon = cls
                    break

            # ── Aggression from cache (updated separately) ───────────────────
            cached           = self.aggression_cache.get(track_id, {})
            aggression_score = float(cached.get("aggression_score", 0.0))
            face_score       = float(cached.get("face_score", 0.0))
            angry            = face_score >= 0.22

            # ── Pose analysis (rate-gated) ────────────────────────────────────
            run_pose = (frame_index is None) or (frame_index % 4 == 0)
            self.behavior_analyzer.analyze_person(
                frame, bbox, track_id,
                frame_index=frame_index, run_pose=run_pose,
            )
            pose_landmarks = self.behavior_analyzer.get_pose_landmarks(track_id)
            owner_bbox     = context.get("owner_bbox")

            # ── Threat conditions ─────────────────────────────────────────────
            #
            # THREAT requires an active, directed gesture near the owner.
            # Mere proximity (overlapping person boxes) is NOT enough.
            #
            # 1. Swing detected: fast wrist movement directed at owner while
            #    already within 200 px of the owner box.
            if self._detect_swing_toward_owner(track_id, bbox, owner_bbox, pose_landmarks):
                return "THREAT", 90.0, ["SWINGING_AT_OWNER"], {
                    "aggression_score": aggression_score, "has_weapon": has_weapon}

            # 2. Weapon box overlaps the owner box (weapon is at the owner).
            if has_weapon and self._weapon_inside_owner(weapons, owner_bbox):
                return "THREAT", 85.0, [f"WEAPON_AT_OWNER:{strongest_weapon}"], {
                    "aggression_score": aggression_score, "has_weapon": True}

            # 3. Angry + weapon near owner → escalate to THREAT.
            if angry and has_weapon and self._weapon_near_owner(weapons, owner_bbox, threshold_px=150):
                return "THREAT", 80.0, [f"ANGRY_ARMED_NEAR_OWNER:{strongest_weapon}"], {
                    "aggression_score": aggression_score, "has_weapon": True}

            # 4. Weapon held close to owner (within 150 px edge-to-edge) but
            #    not yet overlapping and not angry — approaching but not confirmed hostile.
            if has_weapon and self._weapon_near_owner(weapons, owner_bbox, threshold_px=150):
                return "SUSPICIOUS", 65.0, [f"ARMED_NEAR_OWNER:{strongest_weapon}"], {
                    "aggression_score": aggression_score, "has_weapon": True}

            # 5. Weapon present but far from owner.
            if has_weapon:
                return "SUSPICIOUS", 55.0, [f"ARMED_UNKNOWN:{strongest_weapon}"], {
                    "aggression_score": aggression_score, "has_weapon": True}

            # 6. Angry face with no weapon.
            if angry:
                return "SUSPICIOUS", 40.0, ["ANGRY_UNKNOWN"], {
                    "aggression_score": aggression_score, "has_weapon": False}

            return "SAFE", 0.0, ["SAFE"], {
                "aggression_score": aggression_score, "has_weapon": False}

        except Exception as e:
            print(f"[EnhancedThreatAnalyzer] analyze_person error: {e}")
            return "SAFE", 0.0, ["ERROR"], {}

    def get_pose_landmarks(self, track_id):
        """Get pose landmarks for a track_id from the behavior analyzer"""
        return self.behavior_analyzer.get_pose_landmarks(track_id)
