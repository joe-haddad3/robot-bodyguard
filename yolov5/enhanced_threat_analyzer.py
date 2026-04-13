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
        Compute threat level for one person.

        Returns:
            (level, avg_score, explanation, debug)
            level: "SAFE" | "SUSPICIOUS" | "THREAT"
        """
        try:
            score       = 0.0
            explanation = []
            debug       = {}

            track_id = person["id"]
            identity = person.get("identity", "unknown")
            bbox     = person["bbox"]

            # ---- IDENTITY ----
            identity_score = float(self.IDENTITY_WEIGHTS.get(identity, 3))
            score += identity_score
            debug["identity_score"] = identity_score

            # Owner short-circuits immediately — no further analysis needed
            if identity == "owner":
                explanation.append("OWNER")
                return "SAFE", score, explanation, debug

            # ---- WEAPON SCORE ----
            weapon_score = 0.0
            for w in person.get("weapons", []):
                cls  = w["class_name"]
                conf = float(w["conf"])
                if cls in self.WEAPON_WEIGHTS and conf >= self.WEAPON_THRESHOLDS.get(cls, 0.50):
                    weapon_score += float(self.WEAPON_WEIGHTS[cls])

            score += weapon_score
            debug["weapon_score"] = weapon_score

            # ---- AGGRESSION SCORE (rate-limited, per-person instance) ----
            x1, y1, x2, y2 = [int(v) for v in bbox]
            fh, fw = frame.shape[:2]
            x1 = max(0, min(x1, fw - 1))
            x2 = max(0, min(x2, fw - 1))
            y1 = max(0, min(y1, fh - 1))
            y2 = max(0, min(y2, fh - 1))

            aggression_score  = 0.0
            aggression_points = 0.0

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
                aggression_score  = float(cached.get("aggression_score", 0.0))
                aggression_points = aggression_score * 20.0
                score += aggression_points

            debug["aggression_score"]  = aggression_score
            debug["aggression_points"] = aggression_points

            # ---- BEHAVIOR SCORE (pose — BehaviorAnalyzer owns this) ----
            run_pose = True
            if frame_index is not None:
                run_pose = (frame_index % 4 == 0)

            behavior_score, behaviors = self.behavior_analyzer.analyze_person(
                frame,
                bbox,
                track_id,
                frame_index=frame_index,
                run_pose=run_pose,
            )
            behavior_score = float(behavior_score)
            score += behavior_score
            debug["behavior_score"] = behavior_score
            explanation.extend(behaviors)

            # ---- DISTANCE TO ROBOT ----
            bbox_height       = bbox[3] - bbox[1]
            robot_dist_score  = 0.0

            if bbox_height > 400:
                robot_dist_score = 5.0
            elif bbox_height > 300:
                robot_dist_score = 3.0
            elif bbox_height > 200:
                robot_dist_score = 1.0

            score += robot_dist_score
            debug["robot_distance_score"] = robot_dist_score

            # ---- OWNER PROXIMITY ----
            owner_bbox            = context.get("owner_bbox", None)
            owner_proximity_score = 0.0

            if owner_bbox is not None:
                person_center = self._bbox_center(bbox)
                owner_center  = self._bbox_center(owner_bbox)
                dist          = self._distance(person_center, owner_center)

                if dist < 120:
                    owner_proximity_score = 10.0
                elif dist < 220:
                    owner_proximity_score = 6.0
                elif dist < 320:
                    owner_proximity_score = 3.0

                self.owner_distance_history[track_id].append(dist)
                self.owner_distance_history[track_id] = self.owner_distance_history[track_id][-5:]

                hist = self.owner_distance_history[track_id]
                if len(hist) >= 3:
                    if hist[0] - hist[-1] > 120:
                        owner_proximity_score += 8.0
                    elif hist[0] - hist[-1] > 60:
                        owner_proximity_score += 4.0

            score += owner_proximity_score
            debug["owner_proximity_score"] = owner_proximity_score

            # ---- FINAL SMOOTHING (last 10 frames) ----
            self.threat_history[track_id].append(score)
            self.threat_history[track_id] = self.threat_history[track_id][-10:]
            avg_score = float(np.mean(self.threat_history[track_id]))

            if avg_score >= self.THREAT_THRESHOLD:
                level = "THREAT"
            elif avg_score >= self.SUSPICIOUS_THRESHOLD:
                level = "SUSPICIOUS"
            else:
                level = "SAFE"

            return level, avg_score, explanation, debug

        except Exception as e:
            print(f"[EnhancedThreatAnalyzer] Error in analyze_person: {e}")
            return "SAFE", 0.0, ["ANALYSIS_ERROR"], {"error": str(e)}

    def get_pose_landmarks(self, track_id):
        """Get pose landmarks for a track_id from the behavior analyzer"""
        return self.behavior_analyzer.get_pose_landmarks(track_id)

