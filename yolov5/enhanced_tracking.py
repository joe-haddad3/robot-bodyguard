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
class MotionPredictor:
    """Simple motion prediction using linear extrapolation"""

    def __init__(self, history_size=5):
        self.history = collections.deque(maxlen=history_size)
        self.size_history = collections.deque(maxlen=history_size)
        self.velocity = [0, 0]
        self.last_prediction = None

    def update(self, box):
        """Update with new measurement"""
        x1, y1, x2, y2 = box
        center = [(x1 + x2) / 2, (y1 + y2) / 2]
        size = [x2 - x1, y2 - y1]

        self.history.append(center)
        self.size_history.append(size)

        # Update velocity if we have enough history
        if len(self.history) >= 2:
            prev = self.history[-2]
            curr = self.history[-1]
            self.velocity = [curr[0] - prev[0], curr[1] - prev[1]]

    def predict(self, steps=1):
        """Predict next position using linear extrapolation"""
        if len(self.history) < 2:
            return None

        last_center = self.history[-1]
        predicted_center = [
            last_center[0] + self.velocity[0] * steps,
            last_center[1] + self.velocity[1] * steps
        ]

        if len(self.size_history) >= 2:
            last_size = self.size_history[-1]
            size_velocity = [
                self.size_history[-1][0] - self.size_history[-2][0],
                self.size_history[-1][1] - self.size_history[-2][1]
            ]
            predicted_size = [
                last_size[0] + size_velocity[0] * steps,
                last_size[1] + size_velocity[1] * steps
            ]
        else:
            predicted_size = self.size_history[-1] if self.size_history else [100, 200]

        # Convert back to bounding box
        w, h = predicted_size
        cx, cy = predicted_center
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        self.last_prediction = [int(x1), int(y1), int(x2), int(y2)]
        return self.last_prediction


# ---------------------------
# ENHANCED FINGERPRINT
# ---------------------------
class AdaptiveFingerprint:
    """Enhanced fingerprint with multiple visual features"""

    def __init__(self):
        self.base_color_hist = None
        self.enhanced_color_hist = None
        self.shape_feats = None
        self.edge_profile = None
        self.hog_features = None
        self.spatial_color = None
        self.skeletal_signatures = None  # Geometric ratios from pose landmarks
        # Per-zone histograms: [head/hair, torso/shirt, lower-body]
        # Zone boundaries (fraction of box height): 0–0.25, 0.25–0.65, 0.65–1.0
        self.zone_hists = None   # list of 3 normalised HSV histograms
        self.needs_enhanced = False

        # Update tracking
        self.update_counter = 0
        self.stable_frames = 0

    def build(self, frame, box, enhanced=False):
        """Build initial fingerprint"""
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

            # Basic color histogram
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            self.base_color_hist = cv2.calcHist([hsv], [0, 1, 2], None,
                                              [8, 8, 8], [0, 180, 0, 256, 0, 256])
            self.base_color_hist = cv2.normalize(self.base_color_hist, self.base_color_hist).flatten()

            if enhanced:
                self.needs_enhanced = True
                # Enhanced features for crowded scenes
                self.enhanced_color_hist = self._compute_enhanced_color_hist(hsv)
                self.shape_feats = self._compute_shape_features(roi)
                self.edge_profile = self._compute_edge_profile(roi)
                self.spatial_color = self._compute_spatial_color(roi)
                # HOG features (simplified)
                self.hog_features = self._compute_simple_hog(roi)
                # Per-zone histograms: head/hair, torso, lower-body
                self.zone_hists = self._compute_zone_hists(roi)

            return True
        except Exception as e:
            print(f"Fingerprint build error: {e}")
            return False

    def update(self, frame, box, score):
        """Gradually update fingerprint with new observations"""
        try:
            self.update_counter += 1

            # Only update if score is good and we've seen stability
            if score > 0.7:
                self.stable_frames += 1
            else:
                self.stable_frames = max(0, self.stable_frames - 1)

            # Update every 10 stable frames
            if self.stable_frames >= 10 and self.update_counter % 10 == 0:
                alpha = 0.1  # Slow update rate
                x1, y1, x2, y2 = [int(v) for v in box]
                h, w = frame.shape[:2]
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)

                if x2 > x1 and y2 > y1:
                    roi = frame[y1:y2, x1:x2]
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    new_hist = cv2.calcHist([hsv], [0, 1, 2], None,
                                           [8, 8, 8], [0, 180, 0, 256, 0, 256])
                    new_hist = cv2.normalize(new_hist, new_hist).flatten()

                    if self.base_color_hist is not None:
                        self.base_color_hist = (1 - alpha) * self.base_color_hist + alpha * new_hist

        except Exception as e:
            print(f"Fingerprint update error: {e}")

    def compare(self, frame, box, pose_landmarks=None, enhanced=False):
        """Compare candidate with fingerprint"""
        try:
            x1, y1, x2, y2 = [int(v) for v in box]
            h, w = frame.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return 0.0

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return 0.0

            # Basic color comparison
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None,
                              [8, 8, 8], [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            if self.base_color_hist is None:
                return 0.0

            base_score = cv2.compareHist(self.base_color_hist, hist, cv2.HISTCMP_CORREL)

            if not enhanced or not self.needs_enhanced:
                return float(base_score)

            # Enhanced comparison
            # Weights (sum = 1.0 when all features present):
            #   zone_hists   0.40  — head/hair + shirt + trousers compared separately
            #   edge_profile 0.20  — body silhouette / shape
            #   enh_color    0.15  — whole-body colour (backup when zones are weak)
            #   spatial_color 0.10 — quadrant colours
            #   hog          0.10  — gradient orientations / body shape
            #   shape_feats  0.03  — Hu moments (low — not very discriminative)
            #   skeletal     0.02  — pose ratios (rarely available on RPi)
            enhanced_score = 0.0

            # Zone histograms: head/hair · torso/shirt · lower-body
            # Per-zone weights: head 0.25, torso 0.45, lower 0.30
            if self.zone_hists is not None:
                new_zones = self._compute_zone_hists(roi)
                if new_zones is not None:
                    zone_weights = [0.25, 0.45, 0.30]
                    zone_score = 0.0
                    for saved_z, new_z, wz in zip(self.zone_hists, new_zones, zone_weights):
                        s = float(cv2.compareHist(saved_z, new_z, cv2.HISTCMP_CORREL))
                        zone_score += wz * max(0.0, s)
                    enhanced_score += 0.40 * zone_score

            # Edge profile — body silhouette / shape
            if self.edge_profile is not None:
                new_edges = self._compute_edge_profile(roi)
                if new_edges is not None:
                    edge_sim = np.corrcoef(self.edge_profile, new_edges)[0, 1]
                    enhanced_score += 0.20 * max(0.0, float(edge_sim))

            # Whole-body enhanced colour histogram (backup)
            if self.enhanced_color_hist is not None:
                enh_hist = cv2.calcHist([hsv], [0, 1, 2], None,
                                        [16, 16, 16], [0, 180, 0, 256, 0, 256])
                enh_hist = cv2.normalize(enh_hist, enh_hist).flatten()
                enhanced_score += 0.15 * max(0.0, float(
                    cv2.compareHist(self.enhanced_color_hist, enh_hist, cv2.HISTCMP_CORREL)))

            # Spatial colour (4 quadrants)
            if self.spatial_color is not None:
                new_spatial = self._compute_spatial_color(roi)
                if new_spatial is not None:
                    spatial_sim = cv2.compareHist(self.spatial_color, new_spatial, cv2.HISTCMP_CORREL)
                    enhanced_score += 0.10 * max(0.0, float(spatial_sim))

            # HOG — gradient-based shape descriptor
            if self.hog_features is not None:
                new_hog = self._compute_simple_hog(roi)
                if new_hog is not None:
                    hog_sim = np.dot(self.hog_features, new_hog) / (
                        np.linalg.norm(self.hog_features) * np.linalg.norm(new_hog))
                    enhanced_score += 0.10 * max(0.0, float(hog_sim))

            # Shape features (Hu moments — low weight, not very discriminative)
            if self.shape_feats is not None:
                new_shape = self._compute_shape_features(roi)
                if new_shape is not None:
                    shape_sim = np.dot(self.shape_feats, new_shape) / (
                        np.linalg.norm(self.shape_feats) * np.linalg.norm(new_shape))
                    enhanced_score += 0.03 * max(0.0, float(shape_sim))

            # Skeletal signatures (pose ratios — rarely available on RPi)
            if pose_landmarks and self.skeletal_signatures is not None:
                candidate_signatures = self._extract_skeletal_signatures(pose_landmarks)
                if candidate_signatures is not None:
                    skeletal_score = self.compare_skeletal(candidate_signatures)
                    enhanced_score += 0.02 * skeletal_score

            return float(0.5 * base_score + 0.5 * enhanced_score)

        except Exception as e:
            print(f"Fingerprint compare error: {e}")
            return 0.0

    def _compute_enhanced_color_hist(self, hsv):
        """Compute enhanced color histogram with spatial weighting"""
        hist = cv2.calcHist([hsv], [0, 1, 2], None,
                          [16, 16, 16], [0, 180, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def _compute_zone_hists(self, roi):
        """Split the person bounding box into 3 vertical zones and compute
        a colour histogram for each:
          Zone 0 (top 25%)  — head / hair region
          Zone 1 (25–65%)   — torso / shirt
          Zone 2 (65–100%)  — lower body / trousers

        Returns a list of 3 normalised HSV histograms, or None on failure.
        Each histogram uses 12 × 12 Hue × Saturation bins (ignores Value to
        reduce lighting sensitivity).
        """
        try:
            h, w = roi.shape[:2]
            if h < 30:   # too small to zone reliably
                return None
            boundaries = [0, int(0.25 * h), int(0.65 * h), h]
            zones = []
            for i in range(3):
                zone = roi[boundaries[i]:boundaries[i + 1], :]
                if zone.size == 0:
                    zones.append(np.zeros((12 * 12,), dtype=np.float32))
                    continue
                hsv_z = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
                # Use only Hue + Saturation — Value changes with lighting
                z_hist = cv2.calcHist([hsv_z], [0, 1], None,
                                      [12, 12], [0, 180, 0, 256])
                z_hist = cv2.normalize(z_hist, z_hist).flatten().astype(np.float32)
                zones.append(z_hist)
            return zones
        except Exception as e:
            print(f"Zone hist error: {e}")
            return None

    def _compute_shape_features(self, roi):
        """Compute shape-based features"""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Simple shape descriptors
            moments = cv2.moments(gray)
            hu_moments = cv2.HuMoments(moments).flatten()
            return hu_moments / np.linalg.norm(hu_moments) if np.linalg.norm(hu_moments) > 0 else None
        except:
            return None

    def _compute_edge_profile(self, roi):
        """Compute edge profile"""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            # Horizontal and vertical projections
            h_proj = np.sum(edges, axis=0) / max(1, np.sum(edges))
            v_proj = np.sum(edges, axis=1) / max(1, np.sum(edges))
            return np.concatenate([h_proj, v_proj])
        except:
            return None

    def _compute_spatial_color(self, roi):
        """Compute spatial color distribution"""
        try:
            # Divide into 4 quadrants and compute color histograms
            h, w = roi.shape[:2]
            h_mid, w_mid = h // 2, w // 2

            quadrants = [
                roi[:h_mid, :w_mid],
                roi[:h_mid, w_mid:],
                roi[h_mid:, :w_mid],
                roi[h_mid:, w_mid:]
            ]

            hist = []
            for quad in quadrants:
                if quad.size > 0:
                    hsv = cv2.cvtColor(quad, cv2.COLOR_BGR2HSV)
                    q_hist = cv2.calcHist([hsv], [0, 1, 2], None,
                                        [4, 4, 4], [0, 180, 0, 256, 0, 256])
                    hist.extend(q_hist.flatten())
                else:
                    hist.extend([0] * 64)

            hist = np.array(hist)
            return cv2.normalize(hist, hist).flatten()
        except:
            return None

    def _compute_simple_hog(self, roi):
        """Compute simplified HOG features"""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 128))

            # Simple gradient computation
            gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=1)
            gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=1)
            mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

            # Simple binning (9 bins for 0-180 degrees)
            bins = np.zeros(9)
            for i in range(9):
                mask = (ang >= i*20) & (ang < (i+1)*20)
                bins[i] = np.sum(mag[mask])

            return bins / np.linalg.norm(bins) if np.linalg.norm(bins) > 0 else None
        except:
            return None


# ---------------------------
# SKELETAL SIGNATURE METHODS
# ---------------------------
    def build_with_pose(self, frame, box, pose_landmarks, enhanced=False):
        """Build fingerprint with pose landmarks for skeletal signatures"""
        # First build regular fingerprint
        success = self.build(frame, box, enhanced)

        if success and pose_landmarks:
            self.skeletal_signatures = self._extract_skeletal_signatures(pose_landmarks)

        return success

    def _extract_skeletal_signatures(self, landmarks):
        """Extract view-angle invariant geometric ratios from pose landmarks"""
        try:
            # Define key landmark indices (MediaPipe pose landmarks)
            NOSE = 0
            LEFT_SHOULDER = 11
            RIGHT_SHOULDER = 12
            LEFT_HIP = 23
            RIGHT_HIP = 24
            LEFT_KNEE = 25
            RIGHT_KNEE = 26
            LEFT_ANKLE = 27
            RIGHT_ANKLE = 28

            # Extract 3D coordinates (x, y, z) - using only x,y for now since z might be less reliable
            def get_point(idx):
                return np.array([landmarks[idx].x, landmarks[idx].y])

            # Key body measurements (distances)
            shoulder_width = np.linalg.norm(get_point(LEFT_SHOULDER) - get_point(RIGHT_SHOULDER))
            hip_width = np.linalg.norm(get_point(LEFT_HIP) - get_point(RIGHT_HIP))
            left_leg_length = np.linalg.norm(get_point(LEFT_HIP) - get_point(LEFT_ANKLE))
            right_leg_length = np.linalg.norm(get_point(RIGHT_HIP) - get_point(RIGHT_ANKLE))
            torso_height = np.linalg.norm(get_point(NOSE) - (get_point(LEFT_HIP) + get_point(RIGHT_HIP)) / 2)

            # Ratios that are view-angle invariant
            shoulder_to_hip_ratio = shoulder_width / hip_width if hip_width > 0 else 0
            leg_to_torso_ratio = (left_leg_length + right_leg_length) / 2 / torso_height if torso_height > 0 else 0
            left_right_leg_ratio = left_leg_length / right_leg_length if right_leg_length > 0 else 1

            # Additional geometric features
            knee_height_ratio = (np.linalg.norm(get_point(LEFT_KNEE) - get_point(LEFT_HIP)) +
                               np.linalg.norm(get_point(RIGHT_KNEE) - get_point(RIGHT_HIP))) / 2 / torso_height if torso_height > 0 else 0

            # Store as normalized feature vector
            signatures = np.array([
                shoulder_to_hip_ratio,
                leg_to_torso_ratio,
                left_right_leg_ratio,
                knee_height_ratio,
                shoulder_width,  # absolute measurements for additional context
                torso_height
            ])

            # Normalize to make comparison scale-invariant
            signatures = signatures / np.linalg.norm(signatures) if np.linalg.norm(signatures) > 0 else signatures

            return signatures

        except Exception as e:
            print(f"Skeletal signature extraction error: {e}")
            return None

    def compare_skeletal(self, other_signatures):
        """Compare skeletal signatures using cosine similarity"""
        if self.skeletal_signatures is None or other_signatures is None:
            return 0.0

        try:
            # Cosine similarity for normalized vectors
            similarity = np.dot(self.skeletal_signatures, other_signatures) / (
                np.linalg.norm(self.skeletal_signatures) * np.linalg.norm(other_signatures)
            )
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
        except:
            return 0.0


# ---------------------------
# OBSTACLE DETECTOR
# ---------------------------
class ObstacleDetector:
    """Detect obstacles (non-person objects) in danger zone"""

    def __init__(self, min_obstacle_area_ratio=0.01):
        self.min_obstacle_area_ratio = min_obstacle_area_ratio
        self.last_obstacles = []

    def get_nearest_obstacle(self, frame, yolo_results):
        """Find nearest obstacle in danger zone"""
        if yolo_results is None:
            return None, 0

        obstacles = []
        frame_area = frame.shape[0] * frame.shape[1]

        # Extract non-person detections as potential obstacles
        for det in yolo_results:
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls)

            # Skip persons (class 0)
            if cls == 0:
                continue

            area = (x2 - x1) * (y2 - y1)
            area_ratio = area / frame_area

            if area_ratio >= self.min_obstacle_area_ratio:
                obstacles.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'conf': float(conf),
                    'class': cls,
                    'area_ratio': area_ratio
                })

        if not obstacles:
            self.last_obstacles = []
            return None, 0

        # Find obstacle with highest confidence
        best_obstacle = max(obstacles, key=lambda x: x['conf'])
        self.last_obstacles = obstacles

        return best_obstacle['bbox'], best_obstacle['area_ratio']


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
        self.owner_distance_threshold = 0.35  # slightly looser so angled/partial faces still match
        self.owner_confirm_frames = 2          # 2 consecutive matches instead of 3 — faster lock-on
        self.owner_lost_frames_threshold = 5
        self.face_expand = 0.15               # larger padding around detected face crop
        self.min_owner_face_size = 40         # accept smaller faces (farther away / partially visible)

        # Motion prediction
        self.motion_prediction_history = 5
        self.prediction_enabled = True

        # Performance settings
        self.rpi_mode = True
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
        self.camera_width = 320
        self.camera_height = 240
        self.camera_fps = 10
        self.yolo_every_n_frames = 5
        self.owner_recognition_every = 15
        self.min_owner_face_size = 25


# ---------------------------
# PERSON RE-ID STORE
# ---------------------------

class PersonReIDStore:
    """
    Face-based person re-identification store.

    Each person gets a stable display ID the first time their face is seen.
    When they reappear with a new ByteTrack ID, their face is compared against
    stored embeddings to restore their previous stable ID.

    No body histograms.  No lost pool.  One job: face → stable_id.
    """

    EXPIRY_SECONDS = 120   # forget a person if not seen for 2 minutes

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
        """Compare candidate box with stored fingerprint"""
        if track_id not in self.fingerprints:
            return 0.0

        return self.fingerprints[track_id].compare(frame, candidate_box, enhanced)

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