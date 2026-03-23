import numpy as np
from collections import defaultdict
from behavior_analyzer import BehaviorAnalyzer


class EnhancedThreatAnalyzer:
    """
    Multi-factor threat analysis
    - Identity
    - Context
    - Weapons (multiple)
    - Behavior
    - Proximity to robot/camera
    - Proximity to owner
    - Multiple unknowns
    - Temporal smoothing
    """

    def __init__(self):
        self.threat_history = defaultdict(list)
        self.behavior_analyzer = BehaviorAnalyzer()
        self.owner_distance_history = defaultdict(list)

        self.IDENTITY_WEIGHTS = {
            "owner": -20,
            "family1": -20,
            "family2": -20,
            "known1": 0,
            "unknown": 3,
            "owner_proxy": -20,
        }

        self.WEAPON_WEIGHTS = {
            "gun": 12,
            "knife": 8,
            "hammer": 6,
            "baseball_bat": 5,
        }

        self.WEAPON_THRESHOLDS = {
            "gun": 0.70,
            "knife": 0.45,
            "hammer": 0.45,
            "baseball_bat": 0.60,
        }

        self.THREAT_THRESHOLD = 25
        self.SUSPICIOUS_THRESHOLD = 15

    def _bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _distance(self, a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def analyze_person(self, frame, person, context):
        score = 0
        explanation = []

        track_id = person["id"]
        identity = person.get("identity", "unknown")
        bbox = person["bbox"]

        # FACTOR 1: IDENTITY
        identity_score = self.IDENTITY_WEIGHTS.get(identity, 3)
        score += identity_score

        if identity_score < 0:
            explanation.append(f"Known: {identity}")
            return "SAFE", score, explanation
        else:
            explanation.append(f"Unknown person (+{identity_score})")

        # FACTOR 2: CONTEXT
        if context.get("mode") == "indoor":
            if not context.get("owner_present", True):
                score += 6
                explanation.append("Intruder context (+6)")

        if context.get("time_of_day") == "night":
            score += 2
            explanation.append("Night time (+2)")

        # FACTOR 3: MULTIPLE WEAPONS
        weapons = person.get("weapons", [])

        counted_weapons = 0
        for w in weapons:
            weapon_class = w.get("class_name")
            weapon_conf = float(w.get("conf", 0.0))

            if weapon_class in self.WEAPON_WEIGHTS:
                min_conf = self.WEAPON_THRESHOLDS.get(weapon_class, 1.0)
                if weapon_conf >= min_conf:
                    wscore = self.WEAPON_WEIGHTS[weapon_class]
                    score += wscore
                    counted_weapons += 1
                    explanation.append(f"{weapon_class} detected (+{wscore})")

        # small extra escalation if multiple weapons are associated to same person
        if counted_weapons >= 2:
            score += 6
            explanation.append("Multiple weapons (+6)")

        # FACTOR 4: BEHAVIOR
        behavior_score, behaviors = self.behavior_analyzer.analyze_person(
            frame, bbox, track_id
        )
        score += behavior_score
        for behavior in behaviors:
            explanation.append(behavior)

        # FACTOR 5: PROXIMITY TO ROBOT / CAMERA
        bbox_height = bbox[3] - bbox[1]

        if bbox_height > 400:
            score += 5
            explanation.append("Very close to robot (+5)")
        elif bbox_height > 300:
            score += 3
            explanation.append("Close to robot (+3)")
        elif bbox_height > 200:
            score += 1
            explanation.append("Moderate distance to robot (+1)")

        # FACTOR 6: PROXIMITY TO OWNER
        owner_bbox = context.get("owner_bbox", None)
        owner_id = context.get("owner_id", None)

        if owner_bbox is not None and owner_id is not None and track_id != owner_id:
            person_center = self._bbox_center(bbox)
            owner_center = self._bbox_center(owner_bbox)

            dist = self._distance(person_center, owner_center)

            self.owner_distance_history[track_id].append(dist)
            self.owner_distance_history[track_id] = self.owner_distance_history[track_id][-5:]

            if dist < 120:
                score += 10
                explanation.append("Very close to owner (+10)")
            elif dist < 220:
                score += 6
                explanation.append("Close to owner (+6)")
            elif dist < 320:
                score += 3
                explanation.append("Near owner (+3)")

            hist = self.owner_distance_history[track_id]
            if len(hist) >= 3:
                old_dist = hist[0]
                new_dist = hist[-1]

                if old_dist - new_dist > 120:
                    score += 8
                    explanation.append("Suddenly closing in on owner (+8)")
                elif old_dist - new_dist > 60:
                    score += 4
                    explanation.append("Approaching owner (+4)")

        # FACTOR 7: MULTIPLE UNKNOWNS
        num_unknowns = context.get("num_unknown_people", 1)
        if num_unknowns > 1:
            extra = (num_unknowns - 1) * 2
            score += extra
            explanation.append(f"Multiple unknowns (+{extra})")

        # TEMPORAL SMOOTHING
        self.threat_history[track_id].append(score)
        self.threat_history[track_id] = self.threat_history[track_id][-10:]
        recent_scores = self.threat_history[track_id]
        avg_score = float(np.mean(recent_scores))

        # FINAL LEVEL
        if avg_score >= self.THREAT_THRESHOLD:
            threat_level = "THREAT"
        elif avg_score >= self.SUSPICIOUS_THRESHOLD:
            threat_level = "SUSPICIOUS"
        else:
            threat_level = "SAFE"

        return threat_level, avg_score, explanation