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

        # ---- Torso (hip midpoint) position history — lunge detection ----
        # Two positions are enough: we only need the frame-to-frame delta.
        self.torso_history = defaultdict(lambda: deque(maxlen=2))

        # ---- Elbow-to-wrist distance history — arm extension detection ----
        self.arm_extension_history = defaultdict(lambda: {"left": deque(maxlen=2), "right": deque(maxlen=2)})

        # ---- Upper-arm point history — upper-arm thrust detection ----
        # 35% shoulder + 65% elbow: stable, always inside the YOLO crop,
        # more informative than shoulder alone during a strike.
        self.upper_arm_history = defaultdict(lambda: {"left": deque(maxlen=3), "right": deque(maxlen=3)})

        # ---- Contact streak — sustained contact detection ----
        # Counts consecutive frames where an arm segment intersects the owner's
        # face/torso zone.  Catches grabs, shoves, and hand placements.
        self.contact_streak = defaultdict(lambda: {"left": 0, "right": 0})

        # ---- Facing orientation history — 3-frame temporal smoothing ----
        # Stores raw True/False/None votes per frame; wrapper computes majority.
        self.facing_history = defaultdict(lambda: deque(maxlen=3))

        # ---- BehaviorAnalyzer (owns pose inference) ----
        pose_every_n   = 10  if self.rpi_mode else 1
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

    # ------------------------------------------------------------------
    # Track lifecycle
    # ------------------------------------------------------------------

    def cleanup_missing_tracks(self, active_track_ids):
        """Remove state for person tracks that no longer exist."""
        active = set(active_track_ids)

        # Explicitly close MediaPipe tasks before dropping the AggressionAnalyzer
        # reference so C++ resources are released immediately rather than waiting
        # for Python's GC.  Without this, rapid track-ID churn (track lost +
        # re-acquired under blur) creates many short-lived model instances that
        # accumulate in memory within a single session.
        for tid in list(self.aggression_analyzers.keys()):
            if tid not in active:
                analyzer = self.aggression_analyzers.pop(tid)
                try:
                    analyzer.face_landmarker.close()
                    analyzer.hand_landmarker.close()
                except Exception:
                    pass

        for d in (
            self.threat_history,
            self.owner_distance_history,
            self.aggression_cache,
            self.aggression_last_run,
            self.wrist_history,
            self.torso_history,
            self.arm_extension_history,
            self.upper_arm_history,
            self.contact_streak,
            self.facing_history,
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

    def _box_iou(self, box1, box2):
        """Intersection-over-union for two axis-aligned boxes."""
        x1a, y1a, x2a, y2a = [float(v) for v in box1]
        x1b, y1b, x2b, y2b = [float(v) for v in box2]
        inter_w = max(0.0, min(x2a, x2b) - max(x1a, x1b))
        inter_h = max(0.0, min(y2a, y2b) - max(y1a, y1b))
        inter = inter_w * inter_h
        area_a = max(0.0, x2a - x1a) * max(0.0, y2a - y1a)
        area_b = max(0.0, x2b - x1b) * max(0.0, y2b - y1b)
        union = area_a + area_b - inter
        if union <= 0.0:
            return 0.0
        return inter / union

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

    def _is_facing_owner_raw(self, pose_landmarks, person_bbox, owner_bbox):
        """
        Single-frame facing estimate combining nose, ears, and shoulders.

        Uses:
          - Shoulder ORDER (lm11.x vs lm12.x): front-facing vs back-facing.
            Anatomical left (lm11) appears image-right for a front-facing person
            so  lm11.x > lm12.x  means shoulder_diff > 0  means front-facing.
          - Nose + ear midpoint offset from shoulder midpoint: left-right turn.
            Ears (lm7, lm8) are averaged with the nose when visible, giving a
            more stable head-yaw signal than nose alone.

        Returns:
          True  — confident the person faces the owner
          False — confident the person does NOT face the owner
          None  — pose quality too weak to determine (caller should skip check)
        """
        if pose_landmarks is None or len(pose_landmarks) < 13:
            return None

        nose_vis = getattr(pose_landmarks[0],  'visibility', 1.0)
        lsh_vis  = getattr(pose_landmarks[11], 'visibility', 1.0)
        rsh_vis  = getattr(pose_landmarks[12], 'visibility', 1.0)

        # Shoulders must be visible for any orientation reading to be reliable
        if lsh_vis < 0.4 or rsh_vis < 0.4:
            return None

        nose = self._landmark_to_global(pose_landmarks[0],  person_bbox)
        l_sh = self._landmark_to_global(pose_landmarks[11], person_bbox)
        r_sh = self._landmark_to_global(pose_landmarks[12], person_bbox)

        owner_cx  = (float(owner_bbox[0]) + float(owner_bbox[2])) / 2.0
        person_cx = (float(person_bbox[0]) + float(person_bbox[2])) / 2.0
        to_owner_x = owner_cx - person_cx

        # shoulder_diff > 0 → front-facing;  < 0 → back-facing
        shoulder_diff = l_sh[0] - r_sh[0]
        sh_mid_x      = (l_sh[0] + r_sh[0]) / 2.0

        # Head offset: blend nose with ear midpoint when ears are visible
        nose_offset = nose[0] - sh_mid_x
        head_offset = nose_offset
        if len(pose_landmarks) > 8:
            lear_vis = getattr(pose_landmarks[7], 'visibility', 0.0)
            rear_vis = getattr(pose_landmarks[8], 'visibility', 0.0)
            if lear_vis >= 0.4 and rear_vis >= 0.4:
                l_ear = self._landmark_to_global(pose_landmarks[7], person_bbox)
                r_ear = self._landmark_to_global(pose_landmarks[8], person_bbox)
                ear_mid_x   = (l_ear[0] + r_ear[0]) / 2.0
                ear_offset  = ear_mid_x - sh_mid_x
                head_offset = 0.5 * nose_offset + 0.5 * ear_offset  # equal weight

        # Case 1: clearly back-facing AND nose not reliably visible → unknown
        if shoulder_diff < -15 and nose_vis < 0.5:
            return None

        # Case 2: meaningful lateral head/body turn → head offset decides
        if abs(head_offset) > 10:
            return head_offset * to_owner_x > 0

        # Case 3: near-frontal, front-facing → True (arm can reach to either side)
        if shoulder_diff > 10:
            return True

        # Case 4: clearly back-facing, head centered → False
        if shoulder_diff < -15:
            return False

        # Ambiguous geometry but visible shoulders → True (permissive)
        return True

    def _is_facing_owner(self, track_id, pose_landmarks, person_bbox, owner_bbox):
        """
        Temporally smoothed facing check.  Stores the raw per-frame vote in
        self.facing_history[track_id] (maxlen=3) and returns the majority.

        Returns:
          True  — majority of recent frames say facing
          False — majority say not facing
          None  — majority of recent frames were unknown (weak pose quality)
        """
        raw  = self._is_facing_owner_raw(pose_landmarks, person_bbox, owner_bbox)
        hist = self.facing_history[track_id]
        hist.append(raw)

        if len(hist) < 2:
            return raw   # not enough history yet; use the single raw result

        true_count  = sum(1 for v in hist if v is True)
        false_count = sum(1 for v in hist if v is False)
        none_count  = sum(1 for v in hist if v is None)
        total       = len(hist)

        # If more than half the recent frames lacked reliable pose, mark unknown
        if none_count > total / 2:
            return None

        # Strict majority: ties do NOT resolve to True.
        # A tie means the evidence is genuinely ambiguous — do not approve a strike.
        return true_count > false_count

    def _detect_arm_entering_owner(
        self,
        track_id,
        person_bbox,
        owner_bbox,
        pose_landmarks,
        min_speed_px=40.0,
        owner_box_margin=10,
        overlap_iou_thresh=0.05,
        overlap_fast_speed_thresh=80.0,
        debug_out=None,
    ):
        """
        Detect a targeted arm strike.  All conditions must hold:
          1. Person is facing the owner (temporally smoothed, None → skip).
          2. One wrist is inside the owner's bounding box (+ owner_box_margin px).
          3. That wrist's speed >= min_speed_px.
          4. The wrist is closing on the owner's center vs the previous frame.
             (Replaces the dot-product direction check, which fails once the wrist
             reaches or overshoots the owner's center.)
          5. Torso speed < 75% of wrist speed (body not surging forward).
          6. The striking arm is not clearly retracting (>3 px decrease in
             elbow→wrist distance over last two frames).

        Lunge guard (pre-pass): both wrists fast AND same direction (cos > 0.70)
        → whole-body movement → reject before per-wrist checks.

        debug_out: optional dict populated with facing/torso/wrist/lunge values
                   for the HUD overlay.
        """
        _dbg = debug_out if debug_out is not None else {}

        if owner_bbox is None or not pose_landmarks or len(pose_landmarks) < 17:
            return False

        # Landmarks are normalised to the expanded crop that BehaviorAnalyzer used
        # (wider than the YOLO box so arms extending sideways are visible).
        # Use pose_bbox for all wrist/elbow/hip landmark-to-global conversions in
        # this method; keep person_bbox for the facing check (head/shoulders are
        # reliably inside the YOLO box and the facing logic is tuned to it).
        pose_bbox = self.behavior_analyzer.get_pose_bbox(track_id) or person_bbox

        # ── Facing check (smoothed; None = unknown = skip) ───────────────────
        facing = self._is_facing_owner(track_id, pose_landmarks, person_bbox, owner_bbox)
        _dbg['facing'] = facing
        if facing is not True:
            return False

        body_iou = self._box_iou(person_bbox, owner_bbox)
        boxes_overlapping = body_iou > overlap_iou_thresh
        _dbg['body_iou'] = round(body_iou, 3)

        ox1, oy1, ox2, oy2 = [float(v) for v in owner_bbox]
        owner_cx = (ox1 + ox2) / 2.0
        owner_cy = (oy1 + oy2) / 2.0
        m = float(owner_box_margin)

        def _in_owner(gx, gy):
            return (ox1 - m) <= gx <= (ox2 + m) and (oy1 - m) <= gy <= (oy2 + m)

        # ── Torso speed (hip midpoint frame-to-frame delta) ───────────────────
        torso_speed = 0.0
        if len(pose_landmarks) >= 25:
            lhip_vis = getattr(pose_landmarks[23], 'visibility', 0.0)
            rhip_vis = getattr(pose_landmarks[24], 'visibility', 0.0)
            if lhip_vis >= 0.4 and rhip_vis >= 0.4:
                l_hip = self._landmark_to_global(pose_landmarks[23], pose_bbox)
                r_hip = self._landmark_to_global(pose_landmarks[24], pose_bbox)
                hip_mid = ((l_hip[0] + r_hip[0]) / 2.0, (l_hip[1] + r_hip[1]) / 2.0)
                t_hist  = self.torso_history[track_id]
                t_hist.append(hip_mid)
                if len(t_hist) >= 2:
                    tdx = t_hist[-1][0] - t_hist[-2][0]
                    tdy = t_hist[-1][1] - t_hist[-2][1]
                    torso_speed = math.sqrt(tdx * tdx + tdy * tdy)
        _dbg['torso_speed'] = round(torso_speed, 1)

        # ── Elbow-to-wrist extension per side ────────────────────────────────
        # arm_ext[side] = True  if arm is currently extending (punch motion)
        #               = False if arm is retracting or stationary
        #               = None  if neither elbow nor shoulder gives a usable signal
        arm_ext = {}
        arm_ext_source = {}
        shoulder_map = {"left": 11, "right": 12}
        for side, elbow_idx, wrist_idx in (("left", 13, 15), ("right", 14, 16)):
            if elbow_idx >= len(pose_landmarks) or wrist_idx >= len(pose_landmarks):
                continue
            e_vis = getattr(pose_landmarks[elbow_idx], 'visibility', 0.0)
            w_vis = getattr(pose_landmarks[wrist_idx], 'visibility', 0.0)
            if e_vis >= 0.4 and w_vis >= 0.4:
                elbow_g = self._landmark_to_global(pose_landmarks[elbow_idx], pose_bbox)
                wrist_g = self._landmark_to_global(pose_landmarks[wrist_idx], pose_bbox)
                dist    = math.sqrt((wrist_g[0] - elbow_g[0]) ** 2 +
                                    (wrist_g[1] - elbow_g[1]) ** 2)
                e_hist  = self.arm_extension_history[track_id][side]
                e_hist.append(dist)
                if len(e_hist) >= 2:
                    arm_ext[side] = e_hist[-1] > e_hist[-2]
                    arm_ext_source[side] = "elbow"
                continue

            shoulder_idx = shoulder_map[side]
            if shoulder_idx >= len(pose_landmarks):
                continue
            s_vis = getattr(pose_landmarks[shoulder_idx], 'visibility', 0.0)
            if s_vis >= 0.4 and w_vis >= 0.4:
                shoulder_g = self._landmark_to_global(pose_landmarks[shoulder_idx], pose_bbox)
                wrist_g    = self._landmark_to_global(pose_landmarks[wrist_idx], pose_bbox)
                dist       = math.sqrt((wrist_g[0] - shoulder_g[0]) ** 2 +
                                       (wrist_g[1] - shoulder_g[1]) ** 2)
                e_hist = self.arm_extension_history[track_id][side]
                e_hist.append(dist)
                if len(e_hist) >= 2:
                    arm_ext[side] = e_hist[-1] > e_hist[-2]
                    arm_ext_source[side] = "shoulder_fallback"

        _dbg['arm_extending'] = arm_ext
        _dbg['arm_ext_source'] = arm_ext_source

        # ── Pass 1: update wrist history and compute per-wrist speeds ─────────
        wrist_data = {}
        for side, wrist_idx in (("left", 15), ("right", 16)):
            if wrist_idx >= len(pose_landmarks):
                continue
            wx, wy = self._landmark_to_global(pose_landmarks[wrist_idx], pose_bbox)
            self.wrist_history[track_id][side].append((wx, wy))
            positions = self.wrist_history[track_id][side]
            if len(positions) >= 2:
                dx    = positions[-1][0] - positions[-2][0]
                dy    = positions[-1][1] - positions[-2][1]
                speed = math.sqrt(dx * dx + dy * dy)
                wrist_data[side] = {"pos": (wx, wy), "speed": speed, "dx": dx, "dy": dy}
            else:
                wrist_data[side] = {"pos": (wx, wy), "speed": 0.0, "dx": 0.0, "dy": 0.0}

        max_wrist_speed = max((d["speed"] for d in wrist_data.values()), default=0.0)
        _dbg['wrist_speed'] = round(max_wrist_speed, 1)

        # ── Lunge guard: both wrists fast AND same direction ──────────────────
        both_fast_floor = min_speed_px * 0.65
        l_spd = wrist_data.get("left",  {}).get("speed", 0.0)
        r_spd = wrist_data.get("right", {}).get("speed", 0.0)
        lunge_guard_fired = False
        if l_spd >= both_fast_floor and r_spd >= both_fast_floor:
            l_dx  = wrist_data["left"]["dx"];  l_dy  = wrist_data["left"]["dy"]
            r_dx  = wrist_data["right"]["dx"]; r_dy  = wrist_data["right"]["dy"]
            l_mag = math.sqrt(l_dx * l_dx + l_dy * l_dy)
            r_mag = math.sqrt(r_dx * r_dx + r_dy * r_dy)
            if l_mag > 0 and r_mag > 0:
                cos_sim = (l_dx * r_dx + l_dy * r_dy) / (l_mag * r_mag)
                if cos_sim > 0.70:
                    lunge_guard_fired = True
            else:
                lunge_guard_fired = True
        _dbg['lunge_guard'] = lunge_guard_fired
        if lunge_guard_fired:
            return False

        # ── Pass 2: per-wrist strike test ─────────────────────────────────────
        for side, data in wrist_data.items():
            speed = data["speed"]
            if speed < min_speed_px:
                continue
            if boxes_overlapping and speed < overlap_fast_speed_thresh:
                continue

            # Torso guard: body surging at ≥75 % of wrist speed → lunge/rush.
            # Raised from 60 % so a step-in punch (attacker advances while striking)
            # is not blocked — the hips naturally move somewhat in that motion.
            if torso_speed > 0.75 * speed:
                continue

            # Arm-extension guard (soft): only blocks when the elbow→wrist distance
            # clearly decreased (>3 px) over the last two frames — i.e. the arm is
            # actively retracting, not extending.  At pose_model_complexity=0,
            # elbow positions jitter a few pixels per frame, so a strict
            # "not extending" veto would block real punches on noisy landmarks.
            ext_hist = self.arm_extension_history[track_id].get(side)
            if ext_hist is not None and len(ext_hist) >= 2:
                delta = ext_hist[-1] - ext_hist[-2]
                if delta < -3:   # arm clearly retracting → skip
                    continue

            wx, wy = data["pos"]

            # Closing-distance check: is the wrist getting closer to the owner's
            # center compared to the previous frame?
            # This replaces the old dot-product direction check, which broke once
            # the wrist reached or slightly overshot the owner center (dot → 0 or
            # negative even during a valid strike contact).
            positions = self.wrist_history[track_id][side]
            if len(positions) >= 2:
                prev_x, prev_y = positions[-2]
                prev_dist = math.sqrt((prev_x - owner_cx) ** 2 + (prev_y - owner_cy) ** 2)
                cur_dist  = math.sqrt((wx    - owner_cx) ** 2 + (wy    - owner_cy) ** 2)
                if cur_dist >= prev_dist:   # wrist not closing on owner → skip
                    continue
            # If only one history entry exists we can't compute closing distance;
            # fall through and let the _in_owner check decide.

            # Wrist must be inside the owner's bounding box (+ margin)
            if _in_owner(wx, wy):
                return True

        return False

    def _detect_upper_arm_thrust_toward_owner(
        self,
        track_id,
        person_bbox,
        owner_bbox,
        pose_landmarks,
        rel_speed_thresh=22.0,
        closing_thresh=8.0,
        aim_cos_thresh=0.25,
        debug_out=None,
    ):
        """
        Complementary strike detector using the upper-arm point (35% shoulder +
        65% elbow) instead of the wrist.

        The elbow and shoulder stay inside the YOLO bounding box even when the
        arm is fully extended, making this detector robust to the tight-crop
        problem that limits the wrist detector.

        Fires when ALL of the following hold for either arm:
          1. Person is facing owner (reads existing smoothed facing_history —
             does NOT add another vote so it can't double-count).
          2. Relative upper-arm speed (arm velocity minus torso velocity)
             >= rel_speed_thresh px.  Torso subtraction removes whole-body
             lunges and approaching steps.
          3. Upper-arm point is closing on the owner center by >= closing_thresh px.
          4. Shoulder→elbow vector aims toward the owner (cosine >= aim_cos_thresh).
        """
        _dbg = debug_out if debug_out is not None else {}

        if owner_bbox is None or not pose_landmarks or len(pose_landmarks) < 15:
            _dbg['ua_thrust'] = False
            return False

        # ── Read facing from existing smoothed history (no new vote added) ───
        f_hist = self.facing_history[track_id]
        if not f_hist:
            _dbg['ua_thrust'] = False
            return False
        true_count  = sum(1 for v in f_hist if v is True)
        false_count = sum(1 for v in f_hist if v is False)
        none_count  = sum(1 for v in f_hist if v is None)
        if none_count > len(f_hist) / 2 or not (true_count > false_count):
            _dbg['ua_thrust'] = False
            return False

        pose_bbox = self.behavior_analyzer.get_pose_bbox(track_id) or person_bbox

        ox1, oy1, ox2, oy2 = [float(v) for v in owner_bbox]
        owner_cx = (ox1 + ox2) / 2.0
        owner_cy = (oy1 + oy2) / 2.0

        # ── Torso velocity: read from existing torso_history (no new append) ─
        # _detect_arm_entering_owner, which runs first, already appended the
        # current hip midpoint.  If hips were not visible that frame, the history
        # may be empty — fall back to zero torso motion (conservative).
        t_hist = self.torso_history[track_id]
        if len(t_hist) >= 2:
            torso_dx = t_hist[-1][0] - t_hist[-2][0]
            torso_dy = t_hist[-1][1] - t_hist[-2][1]
        else:
            torso_dx = torso_dy = 0.0

        max_rel_speed = 0.0
        triggered = False

        for side, sh_idx, el_idx in (("left", 11, 13), ("right", 12, 14)):
            if sh_idx >= len(pose_landmarks) or el_idx >= len(pose_landmarks):
                continue
            sh_vis = getattr(pose_landmarks[sh_idx], 'visibility', 0.0)
            el_vis = getattr(pose_landmarks[el_idx], 'visibility', 0.0)
            if sh_vis < 0.4 or el_vis < 0.4:
                continue

            sh_g = self._landmark_to_global(pose_landmarks[sh_idx], pose_bbox)
            el_g = self._landmark_to_global(pose_landmarks[el_idx], pose_bbox)

            # Upper-arm point: shoulder-biased toward elbow
            ua_x = 0.35 * sh_g[0] + 0.65 * el_g[0]
            ua_y = 0.35 * sh_g[1] + 0.65 * el_g[1]

            hist = self.upper_arm_history[track_id][side]
            hist.append((ua_x, ua_y))

            if len(hist) < 2:
                continue

            arm_dx = hist[-1][0] - hist[-2][0]
            arm_dy = hist[-1][1] - hist[-2][1]

            # Subtract torso motion — isolates arm-relative movement
            rel_dx = arm_dx - torso_dx
            rel_dy = arm_dy - torso_dy
            rel_speed = math.hypot(rel_dx, rel_dy)
            max_rel_speed = max(max_rel_speed, rel_speed)

            if rel_speed < rel_speed_thresh:
                continue

            # Closing check: upper-arm point must be getting closer to owner
            prev_ua = hist[-2]
            prev_dist = math.hypot(prev_ua[0] - owner_cx, prev_ua[1] - owner_cy)
            cur_dist  = math.hypot(ua_x - owner_cx, ua_y - owner_cy)
            closing   = prev_dist - cur_dist

            if closing < closing_thresh:
                continue

            # Aim check: shoulder→elbow direction must point toward owner
            arm_vec_x = el_g[0] - sh_g[0]
            arm_vec_y = el_g[1] - sh_g[1]
            arm_mag   = math.hypot(arm_vec_x, arm_vec_y)

            to_owner_x = owner_cx - sh_g[0]
            to_owner_y = owner_cy - sh_g[1]
            to_owner_mag = math.hypot(to_owner_x, to_owner_y)

            if arm_mag < 1.0 or to_owner_mag < 1.0:
                continue

            aim_cos = (
                (arm_vec_x * to_owner_x + arm_vec_y * to_owner_y)
                / (arm_mag * to_owner_mag)
            )

            if aim_cos < aim_cos_thresh:
                continue

            triggered = True
            break

        _dbg['ua_thrust']    = triggered
        _dbg['ua_rel_speed'] = round(max_rel_speed, 1)
        return triggered

    def _owner_contact_zones(self, owner_bbox):
        """
        Split the owner bounding box into two contact zones:
          face_zone  — upper 35 % (+ margin): slaps, grabs to head
          torso_zone — next 45 % (35–80 %, + margin): shoves, punches to body
        Returns (face_zone, torso_zone) each as (x1, y1, x2, y2).
        """
        ox1, oy1, ox2, oy2 = [float(v) for v in owner_bbox]
        h = oy2 - oy1
        fm, tm = 12.0, 17.0   # face margin, torso margin (px)
        face_bottom  = oy1 + h * 0.35
        torso_bottom = oy1 + h * 0.80
        face_zone  = (ox1 - fm, oy1 - fm,  ox2 + fm, face_bottom  + fm)
        torso_zone = (ox1 - tm, face_bottom, ox2 + tm, torso_bottom + tm)
        return face_zone, torso_zone

    def _segment_intersects_zone(self, p1, p2, zone):
        """True if line segment p1→p2 intersects the axis-aligned rectangle zone."""
        zx1, zy1, zx2, zy2 = zone

        def in_rect(p):
            return zx1 <= p[0] <= zx2 and zy1 <= p[1] <= zy2

        if in_rect(p1) or in_rect(p2):
            return True

        def cross2d(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        def segs_cross(a1, a2, b1, b2):
            d1 = cross2d(b1, b2, a1); d2 = cross2d(b1, b2, a2)
            d3 = cross2d(a1, a2, b1); d4 = cross2d(a1, a2, b2)
            return (
                ((d1 > 0 > d2) or (d1 < 0 < d2)) and
                ((d3 > 0 > d4) or (d3 < 0 < d4))
            )

        corners = [(zx1, zy1), (zx2, zy1), (zx2, zy2), (zx1, zy2)]
        for i in range(4):
            if segs_cross(p1, p2, corners[i], corners[(i + 1) % 4]):
                return True
        return False

    def _detect_arm_contact_with_owner(
        self,
        track_id,
        person_bbox,
        owner_bbox,
        pose_landmarks,
        fast_rel_speed_thresh=28.0,
        contact_streak_thresh=2,
        overlap_iou_thresh=0.05,
        overlap_fast_speed_thresh=70.0,
        debug_out=None,
    ):
        """
        Universal arm-contact detector using the forearm (elbow→wrist) segment,
        with shoulder→elbow as fallback when the wrist is not visible.

        Two trigger modes:
          FAST CONTACT    — segment inside face/torso zone AND relative arm speed
                            >= fast_rel_speed_thresh AND person facing owner.
                            Catches slaps, hits, sudden reaches.
          SUSTAINED CONTACT — segment stays in zone for >= contact_streak_thresh
                            consecutive frames regardless of speed.
                            Catches grabs, shoves, hand placements.
        """
        _dbg = debug_out if debug_out is not None else {}

        if owner_bbox is None or not pose_landmarks or len(pose_landmarks) < 13:
            for side in ('left', 'right'):
                self.contact_streak[track_id][side] = 0
            _dbg['arm_contact'] = False
            return False

        # Guard: overlapping bounding boxes are normal when two people stand close.
        # When they overlap, arm landmarks naturally fall inside the owner zone —
        # this is NOT contact.  Suppress the sustained streak path; only the
        # speed-gated fast-contact path can fire in that case.
        body_iou = self._box_iou(person_bbox, owner_bbox)
        boxes_overlapping = body_iou > overlap_iou_thresh
        _dbg['body_iou'] = round(body_iou, 3)

        pose_bbox = self.behavior_analyzer.get_pose_bbox(track_id) or person_bbox
        face_zone, torso_zone = self._owner_contact_zones(owner_bbox)

        # ── Facing (read existing history — no new vote) ─────────────────────
        f_hist = self.facing_history[track_id]
        facing = False
        if f_hist:
            tc = sum(1 for v in f_hist if v is True)
            fc_count = sum(1 for v in f_hist if v is False)
            nc = sum(1 for v in f_hist if v is None)
            facing = (nc <= len(f_hist) / 2) and (tc > fc_count)

        # ── Torso velocity (read-only — updated by _detect_arm_entering_owner) ─
        t_hist = self.torso_history[track_id]
        if len(t_hist) >= 2:
            torso_dx = t_hist[-1][0] - t_hist[-2][0]
            torso_dy = t_hist[-1][1] - t_hist[-2][1]
        else:
            torso_dx = torso_dy = 0.0

        triggered  = False
        max_streak = 0

        for side, sh_idx, el_idx, wr_idx in (
            ("left",  11, 13, 15),
            ("right", 12, 14, 16),
        ):
            if sh_idx >= len(pose_landmarks):
                self.contact_streak[track_id][side] = 0
                continue

            sh_vis = getattr(pose_landmarks[sh_idx], 'visibility', 0.0)
            el_vis = (getattr(pose_landmarks[el_idx], 'visibility', 0.0)
                      if el_idx < len(pose_landmarks) else 0.0)
            wr_vis = (getattr(pose_landmarks[wr_idx], 'visibility', 0.0)
                      if wr_idx < len(pose_landmarks) else 0.0)

            if sh_vis < 0.3:
                self.contact_streak[track_id][side] = 0
                continue

            sh_g = self._landmark_to_global(pose_landmarks[sh_idx], pose_bbox)

            # ── Choose best segment ──────────────────────────────────────────
            use_forearm  = el_vis >= 0.35 and wr_vis >= 0.35
            use_upperarm = not use_forearm and el_vis >= 0.35

            if use_forearm:
                el_g = self._landmark_to_global(pose_landmarks[el_idx], pose_bbox)
                wr_g = self._landmark_to_global(pose_landmarks[wr_idx], pose_bbox)
                seg_p1, seg_p2 = el_g, wr_g
                # Wrist speed relative to torso (wrist_history already updated)
                w_hist = self.wrist_history[track_id][side]
                if len(w_hist) >= 2:
                    rel_dx = (w_hist[-1][0] - w_hist[-2][0]) - torso_dx
                    rel_dy = (w_hist[-1][1] - w_hist[-2][1]) - torso_dy
                    rel_speed = math.hypot(rel_dx, rel_dy)
                else:
                    rel_speed = 0.0
                can_fast = True
            elif use_upperarm:
                el_g = self._landmark_to_global(pose_landmarks[el_idx], pose_bbox)
                seg_p1, seg_p2 = sh_g, el_g
                rel_speed    = 0.0   # wrist invisible → skip fast-contact gate
                can_fast     = False
            else:
                self.contact_streak[track_id][side] = 0
                continue

            in_face  = self._segment_intersects_zone(seg_p1, seg_p2, face_zone)
            in_torso = self._segment_intersects_zone(seg_p1, seg_p2, torso_zone)
            in_zone  = in_face or in_torso

            # Only increment the sustained streak when boxes are NOT overlapping.
            # Overlapping boxes mean proximity, not physical contact.
            if in_zone and not boxes_overlapping:
                self.contact_streak[track_id][side] += 1
            else:
                self.contact_streak[track_id][side] = 0

            streak     = self.contact_streak[track_id][side]
            max_streak = max(max_streak, streak)

            # FAST CONTACT
            required_fast_speed = (
                overlap_fast_speed_thresh if boxes_overlapping else fast_rel_speed_thresh
            )
            if in_zone and can_fast and rel_speed >= required_fast_speed and facing:
                triggered = True
                break

            # SUSTAINED CONTACT (grabs / shoves / hand placement)
            # Only fires when boxes are not overlapping (streak is suppressed otherwise)
            if streak >= contact_streak_thresh:
                triggered = True
                break

        _dbg['arm_contact']    = triggered
        _dbg['contact_streak'] = max_streak
        return triggered

    # ------------------------------------------------------------------
    # Main scoring
    # ------------------------------------------------------------------

    def _person_near_owner(self, person_bbox, owner_bbox, threshold_px=200):
        """True if the person's box edge is within threshold_px of the owner's box edge."""
        if owner_bbox is None:
            return False
        return self._box_to_box_distance(person_bbox, owner_bbox) < threshold_px

    def analyze_person(self, frame, person, context, frame_index=None, person_moved=True):
        """
        Threat ruleset.

        OWNER  → always SAFE.

        UNKNOWN (only evaluated when owner is in the frame):
          THREAT     if: punching/slapping detected (pose landmarks)
                     or: holding any confirmed weapon
          SUSPICIOUS if: angry face detected (face_score >= 0.22)
                     [face concealment → SUSPICIOUS, evaluated in the main loop]
          SAFE       otherwise, or when owner is not in the frame.

        Returns: (level, score, explanation, debug)
        """
        try:
            track_id = person["id"]
            identity = person.get("identity", "unknown")
            bbox     = person["bbox"]
            weapons  = person.get("weapons", [])

            # ── Weapon presence ──────────────────────────────────────────────
            has_weapon       = False
            strongest_weapon = None
            for w in weapons:
                cls  = w["class_name"]
                conf = float(w["conf"])
                if cls in self.WEAPON_WEIGHTS and conf >= self.WEAPON_THRESHOLDS.get(cls, 0.50):
                    has_weapon       = True
                    strongest_weapon = cls
                    break

            # ── Aggression from cache (updated separately) ───────────────────
            cached              = self.aggression_cache.get(track_id, {})
            aggression_score    = float(cached.get("aggression_score", 0.0))
            face_score          = float(cached.get("face_score", 0.0))
            aggression_state    = str(cached.get("state", "CALM"))
            hand_movement_score = float(cached.get("debug", {}).get("hand_movement_score", 0.0))
            angry               = face_score >= 0.22

            # ── Pose analysis — keeps landmark cache fresh for face-concealment check ──
            # Punch/slap THREAT is evaluated directly in threat_detection_test.py
            # (see the comment there) so it is visible and easy to verify.
            self.behavior_analyzer.analyze_person(
                frame, bbox, track_id,
                frame_index=frame_index, run_pose=True,
            )
            pose_signals     = self.behavior_analyzer.get_pose_signals(track_id, stabilized=True)
            raw_pose_signals = self.behavior_analyzer.get_pose_signals(track_id, stabilized=False)

            owner_present = context.get("owner_bbox") is not None

            _dbg = {
                "aggression_score":    aggression_score,
                "aggression_state":    aggression_state,
                "face_score":          face_score,
                "hand_movement_score": hand_movement_score,
                "has_weapon":          has_weapon,
                "attack_motion":       pose_signals.get("attack_motion", False),
                "attack_motion_raw":   raw_pose_signals.get("attack_motion", False),
            }

            # ── Owner and enrolled family members are always SAFE ────────────
            if identity in ("owner", "family_member"):
                return "SAFE", 0.0, ["OWNER" if identity == "owner" else "FAMILY"], _dbg

            # ── Unknown without owner in frame → SAFE ─────────────────────────
            if not owner_present:
                return "SAFE", 0.0, ["SAFE_NO_OWNER"], _dbg

            # ── From here: unknown person, owner is in the frame ─────────────

            # THREAT: holding any confirmed weapon
            if has_weapon:
                return "THREAT", 85.0, [f"ARMED:{strongest_weapon}"], _dbg

            # SUSPICIOUS: aggression state above CALM
            if aggression_state in ("WATCH", "AGGRESSIVE"):
                return "SUSPICIOUS", 50.0, [f"AGGRESSION:{aggression_state}"], _dbg

            # SUSPICIOUS: angry face detected
            if angry:
                return "SUSPICIOUS", 45.0, ["ANGRY_FACE"], _dbg

            return "SAFE", 0.0, ["SAFE"], _dbg

        except Exception as e:
            print(f"[EnhancedThreatAnalyzer] analyze_person error: {e}")
            return "SAFE", 0.0, ["ERROR"], {}

    def get_pose_landmarks(self, track_id):
        """Get pose landmarks for a track_id from the behavior analyzer"""
        return self.behavior_analyzer.get_pose_landmarks(track_id)
