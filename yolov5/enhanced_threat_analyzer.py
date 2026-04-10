import numpy as np
from collections import defaultdict, deque

from behavior_analyzer import BehaviorAnalyzer
from aggression_analyzer import AggressionAnalyzer
from mask_detector import MaskDetector


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

        # Pre-create a pool of AggressionAnalyzers at startup so MediaPipe
        # models load now (not mid-detection causing a freeze).
        # Pool size = max simultaneous people before a new one is created on-demand.
        _pool_size = 2 if rpi_mode else 4
        print(f"[EnhancedThreatAnalyzer] Pre-warming {_pool_size} AggressionAnalyzers...")
        self._aggression_pool = deque(AggressionAnalyzer() for _ in range(_pool_size))
        print("[EnhancedThreatAnalyzer] Ready.")

        # track_id -> AggressionAnalyzer (assigned from pool)
        self.aggression_analyzers = {}
        self.aggression_ts_counter = 0

        # Mask detector — runs every N frames per person to avoid slowdown
        self.mask_detector   = MaskDetector()
        self.mask_cache      = defaultdict(lambda: {"mask": False, "confidence": 0.0, "label": "UNKNOWN"})
        self.mask_frame_cnt  = defaultdict(int)
        self.MASK_EVERY_N    = 10   # check concealment every 10 frames per person
        self.MASK_SCORE      = 15.0 # threat points — concealed face alone → SUSPICIOUS

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
                freed = self.aggression_analyzers.pop(tid)
                freed.reset()
                self._aggression_pool.append(freed)

        for tid in list(self.mask_cache.keys()):
            if tid not in active_track_ids:
                del self.mask_cache[tid]
                self.mask_frame_cnt.pop(tid, None)

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
                # Get (or assign) this person's private AggressionAnalyzer
                if track_id not in self.aggression_analyzers:
                    if self._aggression_pool:
                        self.aggression_analyzers[track_id] = self._aggression_pool.popleft()
                    else:
                        # Pool exhausted — create a new one (rare: >pool_size people at once)
                        self.aggression_analyzers[track_id] = AggressionAnalyzer()
                analyzer = self.aggression_analyzers[track_id]
                timestamp_ms = self._next_aggression_timestamp()
                aggression_result = analyzer.analyze(crop, timestamp_ms)
                aggression_score = float(aggression_result["aggression_score"])
                aggression_points = aggression_score * 8.0
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

        # ---------------- MASK SCORE ----------------
        mask_score = 0.0
        mask_result = self.mask_cache[track_id]

        if self.mask_detector.enabled:
            self.mask_frame_cnt[track_id] += 1
            if self.mask_frame_cnt[track_id] % self.MASK_EVERY_N == 0:
                # Crop face region = top 40% of person box
                H, W = frame.shape[:2]
                fx1 = max(0, x1); fy1 = max(0, y1)
                fx2 = min(W, x2); fy2 = min(H, y1 + int(0.40 * (y2 - y1)))
                if fx2 > fx1 and fy2 > fy1:
                    face_crop = frame[fy1:fy2, fx1:fx2]
                    box_height = y2 - y1
                    mask_result = self.mask_detector.detect(face_crop, person_box_height=box_height)
                    self.mask_cache[track_id] = mask_result

            if mask_result.get("mask", False):
                mask_score = self.MASK_SCORE

        score += mask_score
        debug["mask_score"]  = mask_score
        debug["mask_label"]  = mask_result.get("label", "UNKNOWN")
        debug["mask_conf"]   = mask_result.get("confidence", 0.0)

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
