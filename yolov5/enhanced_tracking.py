"""
Enhanced Tracking Components — integrated from owner-tracking-robot

This module adds advanced tracking features to the bodyguard robot system:
- Motion prediction for smoother tracking
- Enhanced fingerprinting with multiple visual features
- Obstacle detection for threat assessment
- Configuration management
- Enhanced visualization
"""

import cv2
import numpy as np
import math
import collections
import time
from collections import deque


# ---------------------------
# MOTION PREDICTOR
# ---------------------------
# This class predicts where a person box should be for a few short missing-frame
# gaps after ByteTrack fails to return that person. It uses a lightweight
# constant-velocity Kalman-style model so the fallback is smoother and less
# jittery than raw linear extrapolation, while staying cheap enough for the Pi.
class MotionPredictor:
    """Lightweight constant-velocity Kalman predictor for short box dropouts."""

    def __init__(self, history_size=5):
        self.history = collections.deque(maxlen=history_size)
        self.size_history = collections.deque(maxlen=history_size)
        self._dt = 1.0
        self._F = np.array([
            [1, 0, 0, 0, self._dt, 0, 0, 0],
            [0, 1, 0, 0, 0, self._dt, 0, 0],
            [0, 0, 1, 0, 0, 0, self._dt, 0],
            [0, 0, 0, 1, 0, 0, 0, self._dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        self._H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)
        self._Q = np.diag([1.0, 1.0, 0.8, 0.8, 4.0, 4.0, 2.0, 2.0]).astype(np.float32)
        self._R = np.diag([16.0, 16.0, 25.0, 25.0]).astype(np.float32)
        self._I = np.eye(8, dtype=np.float32)
        self.state = None
        self.cov = None
        self.last_prediction = None

    def _predict_internal(self, steps=1):
        if self.state is None:
            return None, None

        predicted_state = self.state.copy()
        predicted_cov = self.cov.copy()
        for _ in range(max(1, int(steps))):
            predicted_state = self._F @ predicted_state
            predicted_cov = self._F @ predicted_cov @ self._F.T + self._Q
        return predicted_state, predicted_cov

    def update(self, box):
        """Update the filter with a measured bounding box."""
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)

        self.history.append([cx, cy])
        self.size_history.append([w, h])
        measurement = np.array([cx, cy, w, h], dtype=np.float32)

        if self.state is None:
            vx = vy = vw = vh = 0.0
            if len(self.history) >= 2 and len(self.size_history) >= 2:
                prev_center = self.history[-2]
                prev_size = self.size_history[-2]
                vx = cx - prev_center[0]
                vy = cy - prev_center[1]
                vw = w - prev_size[0]
                vh = h - prev_size[1]
            self.state = np.array([cx, cy, w, h, vx, vy, vw, vh], dtype=np.float32)
            self.cov = np.diag([25.0, 25.0, 36.0, 36.0, 100.0, 100.0, 64.0, 64.0]).astype(np.float32)
            self.last_prediction = [int(x1), int(y1), int(x2), int(y2)]
            return

        self.state = self._F @ self.state
        self.cov = self._F @ self.cov @ self._F.T + self._Q

        innovation = measurement - (self._H @ self.state)
        innovation_cov = self._H @ self.cov @ self._H.T + self._R
        kalman_gain = self.cov @ self._H.T @ np.linalg.inv(innovation_cov)

        self.state = self.state + kalman_gain @ innovation
        self.cov = (self._I - kalman_gain @ self._H) @ self.cov

    def predict(self, steps=1):
        """Predict the next box using the current filtered state."""
        predicted_state, predicted_cov = self._predict_internal(steps)
        if predicted_state is None:
            return None

        cx, cy, w, h = predicted_state[:4]
        w = max(1.0, float(w))
        h = max(1.0, float(h))
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        self.state = predicted_state
        self.cov = predicted_cov
        self.last_prediction = [int(x1), int(y1), int(x2), int(y2)]
        return self.last_prediction

    def peek_predict(self, steps=1):
        """Return a predicted box without advancing the filter state."""
        predicted_state, _ = self._predict_internal(steps)
        if predicted_state is None:
            return None

        cx, cy, w, h = predicted_state[:4]
        w = max(1.0, float(w))
        h = max(1.0, float(h))
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return [int(x1), int(y1), int(x2), int(y2)]


# ---------------------------
# ENHANCED FINGERPRINT
# ---------------------------
class AdaptiveFingerprint:
    """
    Lightweight appearance fingerprint used for short-term person re-matching.

    The current project only uses the basic color-histogram path, so this class
    intentionally stays simple and Pi-friendly. The `enhanced` and
    `pose_landmarks` arguments are kept only for API compatibility with the
    surrounding tracking code.
    """

    def __init__(self):
        self.base_color_hist = None
        self.update_counter = 0
        self.stable_frames = 0
        self.snapshot_hists = []

    def _compute_base_hist(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None,
                            [8, 8, 8], [0, 180, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def build(self, frame, box, enhanced=False):
        """Build the initial lightweight fingerprint."""
        try:
            x1, y1, x2, y2 = [int(v) for v in box]
            h, w = frame.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return False

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return False

            self.base_color_hist = self._compute_base_hist(roi)
            self.snapshot_hists = [self.base_color_hist.copy()]
            return True
        except Exception as e:
            print(f"Fingerprint build error: {e}")
            return False

    def update(self, frame, box, score):
        """Gradually refresh the base histogram when the track is stable."""
        try:
            self.update_counter += 1

            if score > 0.7:
                self.stable_frames += 1
            else:
                self.stable_frames = max(0, self.stable_frames - 1)

            if self.base_color_hist is None:
                return

            if self.stable_frames >= 10 and self.update_counter % 10 == 0:
                x1, y1, x2, y2 = [int(v) for v in box]
                h, w = frame.shape[:2]
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)

                if x2 > x1 and y2 > y1:
                    roi = frame[y1:y2, x1:x2]
                    new_hist = self._compute_base_hist(roi)
                    alpha = 0.1
                    self.base_color_hist = (1 - alpha) * self.base_color_hist + alpha * new_hist

                    # Keep a tiny gallery of distinct appearance snapshots so
                    # re-matching survives small viewpoint / lighting changes.
                    is_new_view = True
                    for snap in self.snapshot_hists:
                        if float(cv2.compareHist(snap, new_hist, cv2.HISTCMP_CORREL)) > 0.95:
                            is_new_view = False
                            break
                    if is_new_view:
                        self.snapshot_hists.append(new_hist.copy())
                        if len(self.snapshot_hists) > 4:
                            self.snapshot_hists.pop(0)
        except Exception as e:
            print(f"Fingerprint update error: {e}")

    def compare(self, frame, box, pose_landmarks=None, enhanced=False):
        """Compare a candidate crop against the stored base histogram."""
        try:
            x1, y1, x2, y2 = [int(v) for v in box]
            h, w = frame.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)

            if x2 <= x1 or y2 <= y1 or self.base_color_hist is None:
                return 0.0

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return 0.0

            hist = self._compute_base_hist(roi)
            scores = [float(cv2.compareHist(self.base_color_hist, hist, cv2.HISTCMP_CORREL))]
            for snap in self.snapshot_hists:
                scores.append(float(cv2.compareHist(snap, hist, cv2.HISTCMP_CORREL)))
            return max(scores)
        except Exception as e:
            print(f"Fingerprint compare error: {e}")
            return 0.0


# ---------------------------
# CONFIGURATION SYSTEM
# ---------------------------
class BodyguardConfig:
    """Configuration management for bodyguard system"""

    def __init__(self):
        # Camera settings
        self.camera_width = 640
        self.camera_height = 480
        self.camera_fps = 15

        # YOLO settings
        self.yolo_model_path = "best.pt"
        self.yolo_every_n_frames = 2
        self.conf_threshold = 0.30

        # Tracking settings
        self.iou_threshold = 0.20
        self.distance_threshold = 300
        self.person_missing_frames = 30
        self.object_missing_frames = 15

        # Thresholds
        self.entry_thresholds = {
            "person": 0.50,
            "knife": 0.50,
            "gun": 0.65,
            "baseball_bat": 0.45,
            "hammer": 0.40,
        }

        self.keep_thresholds = {
            "person": 0.20,
            "knife": 0.35,
            "gun": 0.40,
            "baseball_bat": 0.30,
            "hammer": 0.30,
        }

        # Owner recognition
        self.owner_recognition_enabled = True
        self.owner_recognition_every = 2
        self.owner_distance_threshold = 0.45  # live face ~2x enrolled dist; strangers ~0.55+
        self.owner_confirm_frames = 2          # 2 consecutive matches instead of 3 — faster lock-on
        self.owner_lost_frames_threshold = 5
        self.face_expand = 0.15               # larger padding around detected face crop
        self.min_owner_face_size = 40         # accept smaller faces (farther away / partially visible)

        # Motion prediction
        self.motion_prediction_history = 5
        self.prediction_enabled = True

        # Performance settings
        self.rpi_mode = False
        if self.rpi_mode:
            self.camera_width = 320
            self.camera_height = 240
            self.camera_fps = 10
            self.yolo_every_n_frames = 5
            self.owner_recognition_every = 15
            self.min_owner_face_size = 30

    def update_for_rpi(self):
        """Update settings for Raspberry Pi performance"""
        self.rpi_mode = True
        self.camera_width = 640
        self.camera_height = 480
        self.camera_fps = 15
        self.yolo_every_n_frames = 2   # async thread; 640×480 is fast enough at this rate
        self.owner_recognition_every = 2  # run face recognition every 2 frames on Pi
        self.min_owner_face_size = 25


# ---------------------------
# PERSON RE-ID STORE
# ---------------------------
# Without it, the same real person could keep getting a new displayed ID whenever the tracker 
# resets
class PersonReIDStore:
    """
    Face-based person re-identification store.

    Each person gets a stable display ID the first time their face is seen.
    When they reappear with a new ByteTrack ID, their face is compared against
    stored embeddings to restore their previous stable ID.

    No body histograms.  No lost pool.  One job: face → stable_id.
    """

    EXPIRY_SECONDS = 50   # forget a person if not seen for 50 seconds

    def __init__(self):
        self._records     = []   # list of dicts: {face_emb, stable_id, is_owner, last_seen}
        self._next_stable = 1    # monotonically increasing human-readable person number

    def find_match(self, face_emb, cosine_dist_fn, threshold=0.32):
        """Return the closest matching record, or None if no match within threshold."""
        if face_emb is None:
            return None
        best_rec  = None
        best_dist = threshold
        for r in self._records:
            if r["face_emb"] is None:
                continue  # skip persons registered without a face
            d = cosine_dist_fn(face_emb, r["face_emb"])
            if d < best_dist:
                best_dist = d
                best_rec  = r
        return best_rec

    def register(self, face_emb=None, is_owner=False):
        """Register a new person and return their record (with a fresh stable_id)."""
        r = {
            "face_emb":  face_emb,
            "stable_id": self._next_stable,
            "is_owner":  is_owner,
            "last_seen": time.time(),
        }
        self._next_stable += 1
        self._records.append(r)
        return r

    def update_face(self, stable_id, face_emb):
        """Refresh the face embedding for a known person."""
        for r in self._records:
            if r["stable_id"] == stable_id:
                r["face_emb"]  = face_emb
                r["last_seen"] = time.time()
                return

    def touch(self, stable_id):
        """Mark a person as recently seen so they are not expired."""
        for r in self._records:
            if r["stable_id"] == stable_id:
                r["last_seen"] = time.time()
                return

    def mark_owner(self, stable_id):
        """Mark a known person as the owner."""
        for r in self._records:
            if r["stable_id"] == stable_id:
                r["is_owner"]  = True
                r["last_seen"] = time.time()
                return

    def owner_record(self):
        """Return the owner's record, or None if owner not yet registered."""
        for r in self._records:
            if r["is_owner"]:
                return r
        return None

    def tick(self):
        """Drop records not seen for EXPIRY_SECONDS (owner is never removed)."""
        now = time.time()
        self._records = [
            r for r in self._records
            if r["is_owner"] or (now - r["last_seen"]) < self.EXPIRY_SECONDS
        ]


# ---------------------------
# ENHANCED VISUALIZATION
# ---------------------------
# These functions are all about drawing visual information on your video frame so we can see 
# what your system is thinking. They don’t affect tracking or detection logic at all, 
# they just overlay graphics.
def draw_angle_indicator(frame, angle_deg, frame_width, center_y=80, radius=40):
    """Draw angle indicator on frame"""
    center_x = frame_width // 2

    # Draw base circle and line
    cv2.circle(frame, (center_x, center_y), radius, (100, 100, 100), 2)
    cv2.line(frame, (center_x - radius, center_y),
             (center_x + radius, center_y), (100, 100, 100), 1)

    # Draw vertical line
    cv2.line(frame, (center_x, center_y - radius),
             (center_x, center_y + radius), (100, 100, 100), 1)

    # Calculate indicator position
    max_angle = 45  # Max angle for visualization
    limited_angle = max(-max_angle, min(max_angle, angle_deg))
    angle_rad = math.radians(limited_angle)

    indicator_length = radius * 0.8
    indicator_x = center_x + int(indicator_length * math.sin(angle_rad))
    indicator_y = center_y - int(indicator_length * math.cos(angle_rad))

    # Draw indicator line
    cv2.line(frame, (center_x, center_y), (indicator_x, indicator_y), (0, 255, 0), 2)

    # Draw angle text
    cv2.putText(frame, f"{angle_deg:.1f}°", (indicator_x + 5, indicator_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)


def draw_danger_zone(frame, zone_rect, color=(0, 0, 255), alpha=0.3):
    """Draw danger zone overlay"""
    x1, y1, x2, y2 = zone_rect

    # Create overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    # Blend with original frame
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw border
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def draw_center_overlay(frame, text, color=(0, 255, 0), font_scale=0.8, thickness=2):
    """Draw centered text overlay"""
    h, w = frame.shape[:2]
    center_x = w // 2
    center_y = h // 2

    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                                font_scale, thickness)

    # Draw background rectangle
    bg_x1 = center_x - text_w // 2 - 10
    bg_y1 = center_y - text_h // 2 - 10
    bg_x2 = center_x + text_w // 2 + 10
    bg_y2 = center_y + text_h // 2 + 10

    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 2)

    # Draw text
    text_x = center_x - text_w // 2
    text_y = center_y + text_h // 2
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness)


def draw_confidence_bar(frame, confidence, x=10, y=50, width=150, height=10):
    """Draw confidence visualization bar"""
    # Background
    cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)

    # Fill based on confidence
    fill_width = int(width * min(1.0, confidence))
    color = (0, int(255 * confidence), int(55 * (1 - confidence)))
    cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)

    # Border
    cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 200, 200), 1)


# ---------------------------
# ENHANCED TRACKER INTEGRATION
# ---------------------------
# It does 3 main things:
# 1. Remembers how each person moves using MotionPredictor
# 2. Remembers what each person looks like using AdaptiveFingerprint
# 3. Helps recover from short misses by predicting position and comparing appearance
class EnhancedPersonTracker:
    """Enhanced person tracker with motion prediction and advanced features"""

    def __init__(self, config=None):
        self.config = config or BodyguardConfig()
        self.motion_predictors = {}  # track_id -> MotionPredictor
        self.fingerprints = {}       # track_id -> AdaptiveFingerprint
        self.last_boxes = {}
        self.prediction_streak = {}
        self.using_prediction = {}

    def add_person(self, track_id, initial_box, frame):
        """Add new person to tracking"""
        self.motion_predictors[track_id] = MotionPredictor(self.config.motion_prediction_history)
        self.fingerprints[track_id] = AdaptiveFingerprint()
        self.last_boxes[track_id] = initial_box
        self.prediction_streak[track_id] = 0
        self.using_prediction[track_id] = False

        # Build initial fingerprint
        self.fingerprints[track_id].build(frame, initial_box, enhanced=False)

    def update_person(self, track_id, new_box, frame, score):
        """Update existing person track"""
        if track_id not in self.motion_predictors:
            self.add_person(track_id, new_box, frame)
            return

        # Update motion predictor
        self.motion_predictors[track_id].update(new_box)

        # Update fingerprint
        self.fingerprints[track_id].update(frame, new_box, score)

        # Reset prediction state
        self.prediction_streak[track_id] = 0
        self.using_prediction[track_id] = False
        self.last_boxes[track_id] = new_box

    def predict_person_position(self, track_id, frame):
        """Predict person position when detection fails"""
        if track_id not in self.motion_predictors:
            return None

        prediction = self.motion_predictors[track_id].predict()
        if prediction:
            self.prediction_streak[track_id] += 1
            self.using_prediction[track_id] = True
            return prediction
        return None

    def match_with_prediction(self, track_id, detections, frame):
        """Try to match detections with motion prediction"""
        if track_id not in self.motion_predictors:
            return None

        prediction = self.predict_person_position(track_id, frame)
        if not prediction:
            return None

        best_match = None
        best_score = 0

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if int(cls) != 0:  # Not a person
                continue

            det_box = [int(x1), int(y1), int(x2), int(y2)]

            # Check IoU with prediction
            iou_score = self._calculate_iou(prediction, det_box)
            if iou_score > 0.3:  # Good overlap with prediction
                score = iou_score * conf
                if score > best_score:
                    best_score = score
                    best_match = det_box

        return best_match

    def compare_fingerprint(self, track_id, frame, candidate_box, enhanced=False):
        """Compare candidate box with stored appearance plus geometry consistency."""
        if track_id not in self.fingerprints:
            return 0.0

        appearance = self.fingerprints[track_id].compare(frame, candidate_box, enhanced=enhanced)
        geometry = 0.0

        if track_id in self.last_boxes:
            last_box = self.last_boxes[track_id]
            last_iou = self._calculate_iou(last_box, candidate_box)

            last_w = max(1.0, float(last_box[2] - last_box[0]))
            last_h = max(1.0, float(last_box[3] - last_box[1]))
            cand_w = max(1.0, float(candidate_box[2] - candidate_box[0]))
            cand_h = max(1.0, float(candidate_box[3] - candidate_box[1]))

            size_ratio = min(cand_w / last_w, last_w / cand_w) * min(cand_h / last_h, last_h / cand_h)
            geometry += 0.45 * max(0.0, float(last_iou))
            geometry += 0.20 * max(0.0, float(size_ratio))

        predictor = self.motion_predictors.get(track_id)
        if predictor is not None:
            pred_box = predictor.peek_predict()
            if pred_box is not None:
                pred_iou = self._calculate_iou(pred_box, candidate_box)
                geometry += 0.35 * max(0.0, float(pred_iou))

        return float(0.65 * max(0.0, appearance) + 0.35 * min(1.0, geometry))

    def cleanup_missing_tracks(self, active_track_ids):
        """Remove tracking data for disappeared persons"""
        active = set(active_track_ids)

        for track_id in list(self.motion_predictors.keys()):
            if track_id not in active:
                del self.motion_predictors[track_id]
                if track_id in self.fingerprints:
                    del self.fingerprints[track_id]
                if track_id in self.last_boxes:
                    del self.last_boxes[track_id]
                if track_id in self.prediction_streak:
                    del self.prediction_streak[track_id]
                if track_id in self.using_prediction:
                    del self.using_prediction[track_id]

    def _calculate_iou(self, boxA, boxB):
        """Calculate IoU between two boxes"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH

        if interArea == 0:
            return 0.0

        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        denom = areaA + areaB - interArea

        return interArea / denom if denom > 0 else 0.0
