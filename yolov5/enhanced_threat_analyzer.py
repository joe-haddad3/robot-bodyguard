import numpy as np
from collections import defaultdict

from behavior_analyzer import BehaviorAnalyzer
from aggression_analyzer import AggressionAnalyzer


class EnhancedThreatAnalyzer:
    def __init__(self, rpi_mode=False):
        self.threat_history = defaultdict(list)
        self.owner_distance_history = defaultdict(list)

        pose_every_n = 20 if rpi_mode else 12
        min_crop_size = 60 if rpi_mode else 100

        self.behavior_analyzer = BehaviorAnalyzer(
            pose_every_n=pose_every_n,
            velocity_window=10,
            min_crop_size=min_crop_size,
            pose_model_complexity=0,
        )

        self.rpi_mode = rpi_mode
        # One AggressionAnalyzer per tracked person — each has its own
        # history deques and MediaPipe VIDEO-mode state, so they never mix.
        self.aggression_analyzers = {}
        self.aggression_ts_counter = 0

        self.IDENTITY_WEIGHTS = {
            "owner": -20,
            "unknown": 3,
        }

        self.WEAPON_WEIGHTS = {
            "gun": 22,           # gun alone must reach THREAT
            "knife": 18,         # knife alone must reach THREAT
            "hammer": 12,
            "baseball_bat": 8,
        }

        self.WEAPON_THRESHOLDS = {
            "gun": 0.65,         # slightly easier to trigger (was 0.75)
            "knife": 0.50,       # match entry threshold (was 0.4 — too loose)
            "hammer": 0.50,
            "baseball_bat": 0.50,
        }

        self.SUSPICIOUS_THRESHOLD = 8    # catch approaching / raised arms (was 10)
        self.THREAT_THRESHOLD = 18       # weapon alone pushes past this (was 20)

    def cleanup_missing_tracks(self, active_track_ids):
        active_track_ids = set(active_track_ids)

        for tid in list(self.threat_history.keys()):
            if tid not in active_track_ids:
                del self.threat_history[tid]

        for tid in list(self.owner_distance_history.keys()):
            if tid not in active_track_ids:
                del self.owner_distance_history[tid]

        for tid in list(self.aggression_analyzers.keys()):
            if tid not in active_track_ids:
                del self.aggression_analyzers[tid]

    def _bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _distance(self, a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def _next_aggression_timestamp(self):
        self.aggression_ts_counter += 1
        return self.aggression_ts_counter

    def analyze_person(self, frame, person, context, frame_index=None):
        score = 0.0
        explanation = []
        debug = {}

        track_id = person["id"]
        identity = person.get("identity", "unknown")
        bbox = person["bbox"]

        # ---------------- IDENTITY ----------------
        identity_score = float(self.IDENTITY_WEIGHTS.get(identity, 3))
        score += identity_score
        debug["identity_score"] = identity_score

        if identity == "owner":
            explanation.append("OWNER")
            return "SAFE", score, explanation, debug

        # ---------------- WEAPON SCORE ----------------
        weapon_score = 0.0
        for w in person.get("weapons", []):
            cls = w["class_name"]
            conf = float(w["conf"])

            if cls in self.WEAPON_WEIGHTS and conf >= self.WEAPON_THRESHOLDS[cls]:
                weapon_score += float(self.WEAPON_WEIGHTS[cls])

        score += weapon_score
        debug["weapon_score"] = weapon_score

        # ---------------- AGGRESSION SCORE ----------------
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        aggression_score = 0.0
        aggression_points = 0.0

        # Smart skip: if RPi mode and person has been consistently safe with no weapons,
        # skip the expensive 3-model MediaPipe aggression analysis
        prev_scores = self.threat_history.get(track_id, [])
        skip_aggression = (
            self.rpi_mode
            and len(prev_scores) >= 4
            and float(sum(prev_scores[-4:]) / 4) < 6.0
            and weapon_score == 0
        )

        if not skip_aggression and x2 > x1 and y2 > y1:
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                # Get (or create) this person's private AggressionAnalyzer
                if track_id not in self.aggression_analyzers:
                    self.aggression_analyzers[track_id] = AggressionAnalyzer()
                analyzer = self.aggression_analyzers[track_id]
                timestamp_ms = self._next_aggression_timestamp()
                aggression_result = analyzer.analyze(crop, timestamp_ms)
                aggression_score = float(aggression_result["aggression_score"])
                aggression_points = aggression_score * 20.0
                score += aggression_points

        debug["aggression_score"] = aggression_score
        debug["aggression_points"] = aggression_points

        # ---------------- BEHAVIOR SCORE ----------------
        behavior_score, behaviors = self.behavior_analyzer.analyze_person(
            frame,
            bbox,
            track_id,
            frame_index=frame_index,
            run_pose=(frame_index % 4 == 0) if frame_index is not None else True,
        )
        behavior_score = float(behavior_score)
        score += behavior_score
        debug["behavior_score"] = behavior_score
        explanation.extend(behaviors)

        # ---------------- DISTANCE TO ROBOT ----------------
        bbox_height = bbox[3] - bbox[1]
        robot_dist_score = 0.0

        if bbox_height > 400:
            robot_dist_score = 5.0
        elif bbox_height > 300:
            robot_dist_score = 3.0
        elif bbox_height > 200:
            robot_dist_score = 1.0

        score += robot_dist_score
        debug["robot_distance_score"] = robot_dist_score

        # ---------------- OWNER PROXIMITY ----------------
        owner_bbox = context.get("owner_bbox", None)
        owner_proximity_score = 0.0

        if owner_bbox is not None:
            person_center = self._bbox_center(bbox)
            owner_center = self._bbox_center(owner_bbox)
            dist = self._distance(person_center, owner_center)

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
                old_dist = hist[0]
                new_dist = hist[-1]

                if old_dist - new_dist > 120:
                    owner_proximity_score += 8.0
                elif old_dist - new_dist > 60:
                    owner_proximity_score += 4.0

        score += owner_proximity_score
        debug["owner_proximity_score"] = owner_proximity_score

        # ---------------- FINAL SMOOTHING ----------------
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
