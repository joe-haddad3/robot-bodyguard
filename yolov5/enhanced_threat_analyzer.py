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

import time
import numpy as np
from collections import defaultdict

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
        ):
            for tid in list(d.keys()):
                if tid not in active:
                    del d[tid]

    # ------------------------------------------------------------------
    # Main scoring
    # ------------------------------------------------------------------

    def analyze_person(self, frame, person, context, frame_index=None, person_moved=True):
        """
        Compute threat level for one person using a strict ruleset:

        1. Unknown + weapon + close to owner => THREAT
        2. Unknown + weapon + far from owner => SUSPICIOUS
        3. Unknown + angry => SUSPICIOUS
        4. Unknown swings at / kicks owner => THREAT

        Returns:
            (level, avg_score, explanation, debug)
            level: "SAFE" | "SUSPICIOUS" | "THREAT"
        """
        try:
            explanation = []
            debug = {
                "identity_score": 0.0,
                "weapon_score": 0.0,
                "aggression_score": 0.0,
                "aggression_points": 0.0,
                "behavior_score": 0.0,
                "robot_distance_score": 0.0,
                "owner_proximity_score": 0.0,
                "has_weapon": False,
                "close_to_owner": False,
                "angry": False,
                "attacking_owner": False,
            }

            track_id = person["id"]
            identity = person.get("identity", "unknown")
            bbox     = person["bbox"]

            # Owner short-circuits immediately.
            if identity == "owner":
                explanation.append("OWNER")
                return "SAFE", 0.0, explanation, debug

            # ---- WEAPON PRESENCE ----
            has_weapon = False
            strongest_weapon = None
            for w in person.get("weapons", []):
                cls  = w["class_name"]
                conf = float(w["conf"])
                if cls in self.WEAPON_WEIGHTS and conf >= self.WEAPON_THRESHOLDS.get(cls, 0.50):
                    has_weapon = True
                    strongest_weapon = cls
                    break

            debug["has_weapon"] = has_weapon
            debug["weapon_score"] = 1.0 if has_weapon else 0.0

            # ---- FACE / AGGRESSION CACHE ----
            x1, y1, x2, y2 = [int(v) for v in bbox]
            fh, fw = frame.shape[:2]
            x1 = max(0, min(x1, fw - 1))
            x2 = max(0, min(x2, fw - 1))
            y1 = max(0, min(y1, fh - 1))
            y2 = max(0, min(y2, fh - 1))

            aggression_score = 0.0
            face_score = 0.0

            if x2 > x1 and y2 > y1:
                has_cache  = track_id in self.aggression_cache
                should_run = (not has_cache) or (
                    person_moved and (
                        frame_index is None or
                        (frame_index - self.aggression_last_run.get(track_id, -9999)) >= self.aggression_every_n
                    )
                )

                if should_run:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        analyzer     = self._get_aggression_analyzer(track_id)
                        timestamp_ms = int(time.time() * 1000)
                        result       = analyzer.analyze(crop, timestamp_ms)
                        self.aggression_cache[track_id] = result
                        if frame_index is not None:
                            self.aggression_last_run[track_id] = frame_index

                cached = self.aggression_cache.get(track_id, {})
                aggression_score = float(cached.get("aggression_score", 0.0))
                face_score = float(cached.get("face_score", 0.0))

            debug["aggression_score"] = aggression_score

            # "angry" should come from the face/anger signal, not the old weighted score.
            angry = face_score >= 0.22
            debug["angry"] = angry

            # ---- POSE / ATTACK CHECK ----
            run_pose = True
            if frame_index is not None:
                run_pose = (frame_index % 4 == 0)

            _behavior_score, _behaviors = self.behavior_analyzer.analyze_person(
                frame,
                bbox,
                track_id,
                frame_index=frame_index,
                run_pose=run_pose,
            )
            owner_bbox = context.get("owner_bbox", None)
            close_to_owner = False
            if owner_bbox is not None:
                person_center = self._bbox_center(bbox)
                owner_center = self._bbox_center(owner_bbox)
                owner_dist = self._distance(person_center, owner_center)
                self.owner_distance_history[track_id].append(owner_dist)
                self.owner_distance_history[track_id] = self.owner_distance_history[track_id][-5:]
                close_to_owner = owner_dist < 220.0
            debug["close_to_owner"] = close_to_owner
            debug["owner_proximity_score"] = 1.0 if close_to_owner else 0.0

            pose_landmarks = self.behavior_analyzer.get_pose_landmarks(track_id)
            attacking_owner = self._is_attacking_owner(bbox, owner_bbox, pose_landmarks)
            debug["attacking_owner"] = attacking_owner

            if attacking_owner:
                explanation.append("ATTACKING_OWNER")
                return "THREAT", 100.0, explanation, debug

            if has_weapon:
                if close_to_owner:
                    explanation.append(f"ARMED_NEAR_OWNER:{strongest_weapon}")
                    return "THREAT", 90.0, explanation, debug
                explanation.append(f"ARMED_UNKNOWN:{strongest_weapon}")
                return "SUSPICIOUS", 60.0, explanation, debug

            if angry:
                explanation.append("ANGRY_UNKNOWN")
                return "SUSPICIOUS", 40.0, explanation, debug

            return "SAFE", 0.0, explanation, debug

        except Exception as e:
            print(f"[EnhancedThreatAnalyzer] Error in analyze_person: {e}")
            return "SAFE", 0.0, ["ANALYSIS_ERROR"], {"error": str(e)}

    def get_pose_landmarks(self, track_id):
        """Get pose landmarks for a track_id from the behavior analyzer"""
        return self.behavior_analyzer.get_pose_landmarks(track_id)
