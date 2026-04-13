import cv2
import torch
import math
import time
import os
import threading
import numpy as np

from types import SimpleNamespace
from collections import deque, defaultdict

from enhanced_threat_analyzer import EnhancedThreatAnalyzer
from camera_utils import LatestFrameCamera, make_camera
from owner_recognizer_facenet import FaceNetOwnerRecognizer, cosine_distance
from clip_owner_recognizer import CLIPOwnerRecognizer, cosine_similarity as clip_cosine_similarity
from enhanced_tracking import (
    MotionPredictor, AdaptiveFingerprint, ObstacleDetector,
    BodyguardConfig, EnhancedPersonTracker, PersonReIDStore,
    draw_angle_indicator, draw_confidence_bar
)
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.engine.results import Boxes

# Helper to distinguish which recognizer is active without circular imports
# “Try to get the attribute recognizer_type from the object”
# If it exists → use its value
# If it doesn’t exist → use default 'facenet'
def _is_clip(recognizer):
    return getattr(recognizer, 'recognizer_type', 'facenet') == 'clip'

# -----------------------------
# CONFIGURATION
# -----------------------------
config = BodyguardConfig()
if config.rpi_mode:
    config.update_for_rpi()

# Override with local settings
CAM_W = config.camera_width
CAM_H = config.camera_height
CAM_FPS = config.camera_fps
YOLO_EVERY = config.yolo_every_n_frames if config.rpi_mode else 1

# Camera settings
CAMERA_INDEX = 0  # Default camera index

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")

cam = make_camera(
    src=CAMERA_INDEX,
    width=CAM_W,
    height=CAM_H,
    fps=CAM_FPS
).start()

model = torch.hub.load(
    os.path.dirname(os.path.abspath(__file__)),
    "custom",
    path=MODEL_PATH,
    source="local",
    force_reload=False
)

# YOLO confidence threshold is set low so ByteTrack can use lower-confidence
# boxes during second-stage association, which is the core of why ByteTrack
# survives blur, partial turns, and brief occlusion better than a naive tracker.
model.conf = 0.10
                    


# -----------------------------
# ASYNC YOLO RUNNER
# Runs YOLO inference in a background thread (YOLO inference is slow) so the main loop
# never blocks waiting for detection. The main loop submits a
# frame every YOLO_EVERY frames and immediately reads the latest
# completed result — display stays fluid even when YOLO is slow.
# -----------------------------
class AsyncYOLO:
    def __init__(self, yolo_model):
        self._model       = yolo_model
        self._det_lock    = threading.Lock()
        self._frame_lock  = threading.Lock()
        self._detections  = np.empty((0, 6), dtype=np.float32)
        self._result_seq  = 0
        self._new_frame   = None
        self._frame_ready = threading.Event()
        self._thread      = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, frame):
        """Queue a frame for inference. Non-blocking — drops any unprocessed queued frame."""
        with self._frame_lock:
            self._new_frame = frame.copy()
        self._frame_ready.set()

    @property
    def detections(self):
        """Latest completed detection result — numpy array (N, 6) [x1,y1,x2,y2,conf,cls]."""
        with self._det_lock:
            return self._detections

    def latest_result(self):
        """Return the latest detections plus a monotonically increasing result id."""
        with self._det_lock:
            return self._detections.copy(), self._result_seq

    def _worker(self):
        while True:
            self._frame_ready.wait()
            self._frame_ready.clear()
            with self._frame_lock:
                frame = self._new_frame
            if frame is None:
                continue
            results = self._model(frame)
            dets = results.xyxy[0].cpu().numpy()
            with self._det_lock:
                self._detections = dets
                self._result_seq += 1


yolo_runner = AsyncYOLO(model)


# -----------------------------
# CLASS SETUP
# -----------------------------
CLASS_NAMES = {
    0: "person",
    1: "knife",
    2: "gun",
    3: "baseball_bat",
    4: "hammer",
}

WEAPON_CLASSES = {"knife", "gun", "baseball_bat", "hammer"}


# -----------------------------
# THRESHOLDS (from config)
# -----------------------------
ENTRY_THRESHOLDS = config.entry_thresholds
KEEP_THRESHOLDS = config.keep_thresholds

PERSON_MISSING_FRAMES = config.person_missing_frames
OBJECT_MISSING_FRAMES = config.object_missing_frames

CLASS_CONFIRM_FRAMES = {
    "person": 1,
    "knife": 2,    # must appear in 2 frames before tracking — reduces false knife alerts
    "gun": 2,
    "baseball_bat": 2,
    "hammer": 2,
}

CLASS_SWITCH_CONFIRM_FRAMES = 2

CLASS_PRIORITY = {
    "gun": 4,
    "knife": 3,
    "hammer": 2,
    "baseball_bat": 1,
}

FACE_MIN_SIZE = (60, 60)

# How often (frames) to try extracting face embeddings for persons who don't have one yet
FACE_UPDATE_EVERY = 15

# CLIP is expensive on RPi CPU — only run recognition every N frames.
# FaceNet is fast enough to run every OWNER_RECOGNITION_EVERY frames as before.
CLIP_RECOGNITION_EVERY = 45 if config.rpi_mode else 20  # ~3s RPi / ~1s PC at 15fps

# Minimum pixel change ratio WITHIN a person's bounding box crop
# required to re-run MediaPipe models on that person.
# 3% change in their crop means they visibly moved — otherwise use cached result.
PERSON_MOVEMENT_THRESHOLD = 0.03

# MSE motion gate: skip submitting to YOLO when the scene is completely static
# and no persons are currently tracked. Threshold tuned for 640×480; scale up
# proportionally for higher resolutions.
_MOTION_MSE_THRESHOLD = 150.0
_prev_gray_for_motion = None

def _has_motion(frame_bgr):
    """Return True if the frame differs enough from the previous one (MSE gate)."""
    global _prev_gray_for_motion
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if _prev_gray_for_motion is None or _prev_gray_for_motion.shape != gray.shape:
        _prev_gray_for_motion = gray
        return True
    mse = float(np.mean(
        (gray.astype(np.float32) - _prev_gray_for_motion.astype(np.float32)) ** 2
    ))
    _prev_gray_for_motion = gray
    return mse > _MOTION_MSE_THRESHOLD

# -----------------------------
# OWNER RECOGNITION (from config)
# -----------------------------
OWNER_RECOGNITION_ENABLED = config.owner_recognition_enabled
OWNER_RECOGNITION_EVERY = config.owner_recognition_every
OWNER_DISTANCE_THRESHOLD = config.owner_distance_threshold   # enrolled samples score 0.02-0.15 vs mean; live ~2x; strangers ~0.50+
OWNER_CONFIRM_FRAMES = config.owner_confirm_frames          # must match 3 consecutive frames to avoid false positives
OWNER_LOST_FRAMES_THRESHOLD = config.owner_lost_frames_threshold
FACE_EXPAND = config.face_expand
MIN_OWNER_FACE_SIZE = config.min_owner_face_size


# -----------------------------
# TRACKING STATE
# -----------------------------
next_person_id = 0
next_object_id = 0

active_persons = {}
# Tracks that are being held in a grace period while we collect features
# and attempt re-identification before committing to a new person ID.
# Structure: track_id → {bbox, conf, first_seen, face_emb, hists: []}
pending_tracks = {}
PENDING_GRACE_SECONDS = 0.5   # max time before forcing resolution

active_objects = {}
object_class_history = {}
switch_candidate_history = defaultdict(lambda: deque(maxlen=5))

threat_analyzer = EnhancedThreatAnalyzer(config=config)
recognized_owner_id = None
owner_lock_loss_count = 0
owner_profile = None
owner_prediction_streak = 0  # Consecutive frames maintained by prediction

# Recovery confirmation — require the same candidate to match N consecutive
# frames before promoting them to recognized_owner_id.  Prevents a single
# histogram or CLIP hit on a stranger's similar clothing from triggering a
# false owner lock.
RECOVERY_CONFIRM_FRAMES = 2
_recovery_candidate_pid    = None
_recovery_candidate_streak = 0

# Dual recognizer strategy:
#   owner_recognizer = FaceNet (primary, fast, face-based — always loaded)
#   clip_recognizer  = CLIP    (fallback, slow, body-based — used only when
#                               FaceNet finds no face in a person crop, e.g. turned away)
#
# FaceNet handles frontal / angled faces well.
# CLIP handles back-to-camera, covered face, or heavy occlusion.
# Both can run together — whichever finds a confident match updates the streak.
owner_recognizer = FaceNetOwnerRecognizer(threshold=OWNER_DISTANCE_THRESHOLD)

clip_recognizer = None   # populated below if CLIP is installed
try:
    _cr = CLIPOwnerRecognizer(threshold=0.82)
    if _cr.enabled:
        clip_recognizer = _cr  # always assign — gallery populated live from owner tracking
        if _cr.get_gallery_size() > 0:
            print(f"[INIT] CLIP loaded ({clip_recognizer.get_gallery_size()} gallery items from previous session)")
        else:
            print("[INIT] CLIP loaded (empty gallery — will auto-populate once owner is confirmed)")
    else:
        print("[INIT] CLIP not available (install: pip install git+https://github.com/openai/CLIP.git)")
except Exception as e:
    print(f"[INIT] CLIP load failed: {e} — face-only recognition active")

# Enhanced tracking components
enhanced_tracker = EnhancedPersonTracker(config)

# Ultralytics ByteTrack defaults, mirroring the official bytetrack.yaml config.
BYTE_TRACKER_ARGS = SimpleNamespace(
    tracker_type="bytetrack",
    track_high_thresh=0.20,
    track_low_thresh=0.10,
    new_track_thresh=0.15,
    track_buffer=30 if not config.rpi_mode else 60,
    match_thresh=0.80,
    fuse_score=True,
)
byte_tracker = BYTETracker(BYTE_TRACKER_ARGS, frame_rate=CAM_FPS)

# Face-based re-identification store.
# One job: face embedding → stable display ID.
reid_store = PersonReIDStore()

# Fallback display-ID counter for persons whose face was never seen
obstacle_detector = ObstacleDetector()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

frame_count = 0
raw_detections = []
last_yolo_result_seq = -1

# Per-person grayscale crop cache — used to detect whether a person
# moved since the last frame before running expensive MediaPipe models.
person_prev_crops = {}   # track_id -> grayscale crop (numpy array)


# -----------------------------
# HELPERS
# -----------------------------
def bbox_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def bbox_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_width(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1)


def bbox_height(box):
    x1, y1, x2, y2 = box
    return max(0, y2 - y1)


def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def box_intersection_area(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    return interW * interH


def box_visibility_ratio(box, frame_shape):
    fh, fw = frame_shape[:2]
    frame_box = [0, 0, fw, fh]
    inter = box_intersection_area(box, frame_box)
    area = bbox_area(box)
    return inter / area if area > 0 else 0.0


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    areaA = bbox_area(boxA)
    areaB = bbox_area(boxB)
    denom = areaA + areaB - interArea

    if denom <= 0:
        return 0.0

    return interArea / denom


def point_in_box(pt, box):
    x, y = pt
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def class_id_to_name(cls_id):
    return CLASS_NAMES.get(cls_id, None)


def confirm_needed(class_name):
    return CLASS_CONFIRM_FRAMES.get(class_name, 1)


def draw_text_block(frame, lines, x, y, color):
    line_h = 18
    for i, line in enumerate(lines):
        yy = y + i * line_h
        if 0 < yy < frame.shape[0] - 5:
            cv2.putText(
                frame,
                line,
                (x, yy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1
            )


def match_detection_to_tracks(det_box, tracks, iou_threshold=0.20, dist_threshold=250, skip_ids=None, frame=None):
    if skip_ids is None:
        skip_ids = set()

    best_id = None
    best_score = -1

    det_center = bbox_center(det_box)

    # First pass: IoU matching (most reliable)
    for tid, t in tracks.items():
        if tid in skip_ids:
            continue
        track_box = t["bbox"]
        overlap = iou(det_box, track_box)
        if overlap >= iou_threshold and overlap > best_score:
            best_score = overlap
            best_id = tid

    if best_id is not None:
        return best_id

    # Second pass: Distance + fingerprint matching
    best_score = -1e9
    best_fp_score = 0.0

    for tid, t in tracks.items():
        if tid in skip_ids:
            continue
        track_center = bbox_center(t["bbox"])
        d = distance(det_center, track_center)

        # Relaxed distance check (increased from 250 to 350 for multi-person tolerance)
        if d < dist_threshold:
            # Base score inversely proportional to distance
            score = 2.0 / (1.0 + d * 0.01)

            # Add fingerprint comparison if available (enhanced with skeletal features)
            fp_score = 0.0
            if frame is not None and tid in enhanced_tracker.fingerprints:
                fp_score = enhanced_tracker.compare_fingerprint(tid, frame, det_box, enhanced=True)
                best_fp_score = max(best_fp_score, fp_score)
                
                # Boost score significantly if fingerprint matches well
                if fp_score > 0.80:
                    score += 3.0  # Strong match
                elif fp_score > 0.65:
                    score += 1.5  # Good match
                elif fp_score > 0.50:
                    score += 0.8  # Moderate match
                elif fp_score > 0.35:
                    score += 0.3  # Weak match

            if score > best_score:
                best_score = score
                best_id = tid

    # If top fingerprint score is very high, prefer that match over distance alone
    if best_id is not None and best_fp_score > 0.75:
        return best_id

    # Third pass: Motion prediction matching (for lost tracks)
    if best_id is None and frame is not None:
        for tid in tracks.keys():
            if tid in skip_ids:
                continue

            predicted_box = enhanced_tracker.predict_person_position(tid, frame)
            if predicted_box:
                pred_iou = iou(det_box, predicted_box)
                if pred_iou > 0.15:
                    best_id = tid
                    break

    return best_id


def should_accept_new(class_name, conf):
    return conf >= ENTRY_THRESHOLDS.get(class_name, 0.50)


def should_keep_existing(class_name, conf):
    return conf >= KEEP_THRESHOLDS.get(class_name, 0.35)


def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=FACE_MIN_SIZE
    )
    return faces


def face_to_person_box(face, frame_shape):
    x, y, w, h = face
    H, W = frame_shape[:2]

    x1 = max(0, int(x - 0.8 * w))
    y1 = max(0, int(y - 0.6 * h))
    x2 = min(W - 1, int(x + w + 0.8 * w))
    y2 = min(H - 1, int(y + h + 2.8 * h))

    return [x1, y1, x2, y2]


def maybe_switch_object_class(object_id, candidate_class, candidate_conf):
    obj = active_objects[object_id]
    current_class = obj["class_name"]

    if current_class == candidate_class:
        switch_candidate_history[object_id].clear()
        return

    current_priority = CLASS_PRIORITY.get(current_class, 0)
    candidate_priority = CLASS_PRIORITY.get(candidate_class, 0)

    if candidate_priority <= current_priority:
        switch_candidate_history[object_id].clear()
        return

    if candidate_conf < ENTRY_THRESHOLDS.get(candidate_class, 0.50):
        switch_candidate_history[object_id].clear()
        return

    switch_candidate_history[object_id].append(candidate_class)

    recent = list(switch_candidate_history[object_id])[-CLASS_SWITCH_CONFIRM_FRAMES:]
    if len(recent) == CLASS_SWITCH_CONFIRM_FRAMES and all(c == candidate_class for c in recent):
        obj["class_name"] = candidate_class
        switch_candidate_history[object_id].clear()


def expand_face_box(face_box, frame_shape, expand_ratio=0.10):
    x, y, w, h = face_box
    H, W = frame_shape[:2]

    pad_x = int(w * expand_ratio)
    pad_y = int(h * expand_ratio)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W, x + w + pad_x)
    y2 = min(H, y + h + pad_y)

    return (x1, y1, x2, y2)


def detect_faces_full_frame(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=FACE_MIN_SIZE
    )

    results = []
    for (x, y, w, h) in faces:
        if w < MIN_OWNER_FACE_SIZE or h < MIN_OWNER_FACE_SIZE:
            continue

        x1, y1, x2, y2 = expand_face_box((x, y, w, h), frame_bgr.shape, FACE_EXPAND)

        if x2 <= x1 or y2 <= y1:
            continue

        face_crop_bgr = frame_bgr[y1:y2, x1:x2]
        if face_crop_bgr.size == 0:
            continue

        results.append({
            "face_box": (x1, y1, x2, y2),
            "face_crop_bgr": face_crop_bgr,
            "center": ((x1 + x2) / 2, (y1 + y2) / 2),
            "raw_face": (x, y, w, h),
        })

    return results


def match_faces_to_persons(detected_faces, persons):
    assignments = {}
    used_pids = set()

    for face in detected_faces:
        fc = face["center"]

        best_pid = None
        best_score = -1e9

        for pid, pdata in persons.items():
            if not pdata.get("confirmed", False):
                continue
            if pid in used_pids:
                continue

            pbox = pdata["bbox"]
            px1, py1, px2, py2 = pbox
            pc = bbox_center(pbox)
            person_h = max(1.0, bbox_height(pbox))

            d = distance(fc, pc)
            norm_d = d / person_h

            score = 0.0

            if point_in_box(fc, pbox):
                score += 3.0

            score += max(0.0, 2.0 - norm_d)

            fx1, fy1, fx2, fy2 = face["face_box"]
            face_box = [fx1, fy1, fx2, fy2]
            score += 2.0 * iou(face_box, pbox)

            upper_half_box = [px1, py1, px2, py1 + int(0.55 * (py2 - py1))]
            if point_in_box(fc, upper_half_box):
                score += 1.0

            if score > best_score:
                best_score = score
                best_pid = pid

        if best_pid is not None and best_score > 1.0:
            assignments[best_pid] = face
            used_pids.add(best_pid)

    return assignments


def _try_extract_face_emb(frame, box):
    """
    Try to find a face in the upper 55 % of a person box and return its FaceNet embedding.
    Returns a (512,) float32 numpy array, or None if no face was found.

    """
    if not owner_recognizer.enabled:
        return None
    x1, y1, x2, y2 = (int(v) for v in box)
    H, W = frame.shape[:2]
    face_y2 = y1 + int((y2 - y1) * 0.55)
    crop = frame[max(0, y1):min(H, face_y2), max(0, x1):min(W, x2)]
    if crop.size == 0:
        return None
    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(20, 20))
    if len(faces) == 0:
        return None
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    if fw < MIN_OWNER_FACE_SIZE or fh < MIN_OWNER_FACE_SIZE:
        return None
    pad_x = int(fw * FACE_EXPAND)
    pad_y = int(fh * FACE_EXPAND)
    fc = crop[max(0, fy - pad_y):min(crop.shape[0], fy + fh + pad_y),
               max(0, fx - pad_x):min(crop.shape[1], fx + fw + pad_x)]
    if fc.size == 0:
        return None
    return owner_recognizer.embed_face_rgb(cv2.cvtColor(fc, cv2.COLOR_BGR2RGB))


def run_owner_recognition_in_yolo_boxes(frame):
    """Search for the owner inside each tracked person bounding box.

    FaceNet (primary) — runs every call for every person:
      1. Haar cascade inside upper 55 % → precise face crop
      2. Full upper crop to FaceNet if Haar misses → angled / partial faces

    CLIP (fallback per person) — runs only when FaceNet found no usable face
      AND clip_recognizer is loaded AND this is a CLIP-scheduled frame:
      - Full body crop — works with back-to-camera, covered face, side views
      - Rate-limited by CLIP_RECOGNITION_EVERY to protect RPi CPU
    """
    global recognized_owner_id, owner_profile, owner_lock_loss_count

    if not OWNER_RECOGNITION_ENABLED:
        recognized_owner_id = None
        return

    if recognized_owner_id is not None and recognized_owner_id in active_persons:
        return

    for pdata in active_persons.values():
        pdata["face_box_global"] = None
        pdata["owner_distance"] = 999.0
        pdata["owner_label"] = "UNKNOWN"

    H, W = frame.shape[:2]
    best_pid = None
    best_dist = 999.0
    best_face_crop = None
    _clip_frame_ok = (clip_recognizer is not None) and (frame_count % CLIP_RECOGNITION_EVERY == 0)

    for pid, pdata in active_persons.items():
        if not pdata.get("confirmed", False):
            continue

        x1, y1, x2, y2 = int(pdata["bbox"][0]), int(pdata["bbox"][1]), int(pdata["bbox"][2]), int(pdata["bbox"][3])

        dist     = 999.0
        is_owner = False
        haar_face_found = False  # True only when Haar cascade detected a real face

        # ── FaceNet (primary) — works whenever the face is at least partially visible ──
        face_y2 = y1 + int((y2 - y1) * 0.55)
        cx1 = max(0, x1); cy1 = max(0, y1)
        cx2 = min(W, x2); cy2 = min(H, face_y2)

        if cx2 > cx1 and cy2 > cy1:
            person_crop = frame[cy1:cy2, cx1:cx2]
            if person_crop.size > 0:
                face_crop_rgb  = None
                face_box_global = None

                # Attempt 1: Haar cascade
                gray_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray_crop, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
                )
                if len(faces) > 0:
                    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                    if fw >= MIN_OWNER_FACE_SIZE and fh >= MIN_OWNER_FACE_SIZE:
                        pad_x = int(fw * FACE_EXPAND); pad_y = int(fh * FACE_EXPAND)
                        fx1 = max(0, fx - pad_x); fy1 = max(0, fy - pad_y)
                        fx2 = min(person_crop.shape[1], fx + fw + pad_x)
                        fy2 = min(person_crop.shape[0], fy + fh + pad_y)
                        face_crop_bgr = person_crop[fy1:fy2, fx1:fx2]
                        if face_crop_bgr.size > 0:
                            face_crop_rgb   = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                            face_box_global = (cx1 + fx1, cy1 + fy1, cx1 + fx2, cy1 + fy2)
                            haar_face_found = True  # Haar found a real face

                # Attempt 2: full upper crop as face input (angled / partial face)
                if face_crop_rgb is None and min(person_crop.shape[:2]) >= MIN_OWNER_FACE_SIZE:
                    face_crop_rgb  = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                    face_box_global = (cx1, cy1, cx2, cy2)
                    # haar_face_found stays False — this is a fallback, not a real face detection

                if face_crop_rgb is not None:
                    pdata["face_box_global"] = face_box_global

                    face_emb = owner_recognizer.embed_face_rgb(face_crop_rgb)
                    stable_id = pdata.get('stable_id')
                    if stable_id:
                        reid_store.update_face(stable_id, face_emb)

                    mean_dist = (
                        1.0 - float(np.dot(face_emb, owner_recognizer.mean_embedding))
                        if owner_recognizer.mean_embedding is not None else 999.0
                    )
                    dist     = float(mean_dist)
                    is_owner = dist < OWNER_DISTANCE_THRESHOLD
                    if is_owner:
                        best_face_crop = face_crop_rgb

        # ── CLIP (fallback) — kicks in when Haar found no real face ──────────
        # Gated on haar_face_found (not face_found) so CLIP actually runs when
        # the person is facing away, covered, or at an angle where Haar misses.
        # The old gate (not face_found) never fired because the upper-body fallback
        # always set face_found=True, silently disabling CLIP.
        if not haar_face_found and _clip_frame_ok:
            bx1 = max(0, x1); by1 = max(0, y1)
            bx2 = min(W, x2); by2 = min(H, y2)
            if bx2 > bx1 and by2 > by1:
                body_crop = frame[by1:by2, bx1:bx2]
                if body_crop.size > 0:
                    clip_result = clip_recognizer.recognize(body_crop)
                    # Use CLIP result only if it's more confident than FaceNet gave us
                    if clip_result["distance"] < dist:
                        dist     = clip_result["distance"]
                        is_owner = clip_result["is_owner"]
                    pdata["face_box_global"] = None  # no face box for CLIP path

        # ── Update streak and pick the best candidate this frame ──────────────
        pdata["owner_distance"] = dist
        pdata["owner_label"]    = "OWNER" if is_owner else "UNKNOWN"

        if is_owner:
            pdata["owner_match_streak"] = pdata.get("owner_match_streak", 0) + 1
        else:
            pdata["owner_match_streak"] = 0

        if pdata["owner_match_streak"] >= OWNER_CONFIRM_FRAMES and dist < best_dist:
            best_dist = dist
            best_pid  = pid

    if best_pid is not None:
        recognized_owner_id = best_pid
        owner_lock_loss_count = 0
        owner_profile = build_owner_profile(frame, active_persons[best_pid]["bbox"], best_face_crop)
        owner_stable = active_persons[best_pid].get('stable_id')
        if owner_stable is None and owner_profile.get("face_embedding") is not None:
            owner_rec = reid_store.find_match(owner_profile["face_embedding"], cosine_distance)
            if owner_rec is None:
                owner_rec = reid_store.register(owner_profile["face_embedding"], is_owner=True)
            owner_stable = owner_rec["stable_id"]
            active_persons[best_pid]["stable_id"] = owner_stable
        if owner_stable:
            reid_store.mark_owner(owner_stable)
        via = "FaceNet+CLIP" if (best_face_crop is None and clip_recognizer) else "FaceNet"
        print(f"[DEBUG] Built owner profile for {best_pid} via {via}")
        for pid in active_persons:
            active_persons[pid]["owner_confirmed"] = (pid == recognized_owner_id)
            active_persons[pid]["identity"] = "owner" if pid == recognized_owner_id else "unknown"


def build_owner_profile(frame, box, face_rgb=None, pose_landmarks=None):
    profile = {
        "face_embedding": None,
        "fingerprint": AdaptiveFingerprint(),
        "last_seen_box": box,
        "last_seen_frame": frame_count,
        "lost_frames": 0,
    }

    # Build fingerprint with pose landmarks if available
    if pose_landmarks:
        profile["fingerprint"].build_with_pose(frame, box, pose_landmarks, enhanced=True)
    else:
        profile["fingerprint"].build(frame, box, enhanced=True)

    if face_rgb is not None and owner_recognizer.enabled:
        profile["face_embedding"] = owner_recognizer.embed_face_rgb(face_rgb)
    return profile


def update_owner_profile(frame, box, face_rgb=None, score=1.0, pose_landmarks=None):
    global owner_profile
    if owner_profile is None:
        owner_profile = build_owner_profile(frame, box, face_rgb, pose_landmarks)
        return

    owner_profile["last_seen_box"] = box
    owner_profile["last_seen_frame"] = frame_count
    owner_profile["lost_frames"] = 0
    owner_profile["fingerprint"].update(frame, box, score)

    # Update skeletal signatures if pose landmarks are available and we don't have them yet
    if pose_landmarks and owner_profile["fingerprint"].skeletal_signatures is None:
        owner_profile["fingerprint"].skeletal_signatures = owner_profile["fingerprint"]._extract_skeletal_signatures(pose_landmarks)

    if face_rgb is not None and owner_recognizer.enabled:
        owner_profile["face_embedding"] = owner_recognizer.embed_face_rgb(face_rgb)


def recover_owner_by_profile(frame):
    """Scan active persons for the owner when recognized_owner_id is None.

    Three paths, tried in order (cheapest/fastest first):

    FaceNet path — compares face embeddings against the saved face_embedding
                   stored in owner_profile. Fast, but requires a visible face.

    CLIP path  — compares each person's full body crop against the gallery.
                 Called only every CLIP_RECOGNITION_EVERY frames (expensive).
                 Works with back-to-camera, covered face, side views.

    Fingerprint path — compares each person's clothing/appearance histogram
                       against the enhanced fingerprint built when the owner was
                       first recognised. Cheap (no model inference), runs every
                       frame. Works even when the face is fully hidden and CLIP
                       is not installed.  Threshold 0.72 — clothing similarity
                       must be high enough to avoid false positives between
                       people with similar outfits.
    """
    H, W = frame.shape[:2]

    # ── FaceNet: compare face against saved profile embedding ─────────────────
    best_pid   = None
    best_sim   = 0.0

    if owner_profile is not None and owner_profile.get("face_embedding") is not None:
        for pid, pdata in active_persons.items():
            if pid == recognized_owner_id:
                continue
            face_emb = _try_extract_face_emb(frame, pdata["bbox"])
            if face_emb is None:
                continue
            sim = max(0.0, 1.0 - cosine_distance(face_emb, owner_profile["face_embedding"]))
            if sim >= 0.68 and sim > best_sim:
                best_sim = sim
                best_pid = pid

    if best_pid is not None:
        return best_pid  # FaceNet found them — no need for CLIP or fingerprint

    # ── CLIP fallback: gallery match on body crops when face not visible ───────
    # Rate-limited: CLIP is expensive on RPi CPU
    # Run CLIP 4× more aggressively during recovery (owner is lost — urgency > CPU cost)
    _recover_clip_every = max(3, CLIP_RECOGNITION_EVERY // 4)
    if clip_recognizer is not None and clip_recognizer.owner_gallery and frame_count % _recover_clip_every == 0:
        for pid, pdata in active_persons.items():
            if pid == recognized_owner_id:
                continue
            x1, y1, x2, y2 = [int(v) for v in pdata["bbox"]]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(W, x2); y2 = min(H, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            result = clip_recognizer.recognize(crop)
            if result["is_owner"] and result["similarity"] > best_sim:
                best_sim = result["similarity"]
                best_pid = pid

    if best_pid is not None:
        return best_pid  # CLIP found them — no need for fingerprint

    # ── Fingerprint fallback: clothing/appearance histogram match ─────────────
    # Runs every frame (cheap), works even when face is completely hidden and
    # CLIP is not installed. Uses the enhanced fingerprint (zone hists, edges,
    # HOG, spatial colour) built when the owner was first recognised.
    if owner_profile is not None:
        fp = owner_profile.get("fingerprint")
        if fp is not None and fp.base_color_hist is not None:
            for pid, pdata in active_persons.items():
                if pid == recognized_owner_id:
                    continue
                x1, y1, x2, y2 = [int(v) for v in pdata["bbox"]]
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(W, x2); y2 = min(H, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                box_candidate = [x1, y1, x2, y2]
                # Primary: blended-histogram comparison (single view, fast)
                score = fp.compare(frame, box_candidate, enhanced=fp.needs_enhanced)
                # Multi-view: best match across all stored angle snapshots
                snap_score = fp.compare_best_snapshot(frame, box_candidate)
                score = max(score, snap_score)
                if score > 0.72 and score > best_sim:
                    best_sim = score
                    best_pid = pid

    return best_pid


def make_hand_zone(person_box):
    x1, y1, x2, y2 = person_box
    w = bbox_width(person_box)
    h = bbox_height(person_box)

    hx1 = int(x1 - 0.25 * w)
    hy1 = int(y1 + 0.18 * h)
    hx2 = int(x2 + 0.25 * w)
    hy2 = int(y1 + 0.78 * h)

    return [hx1, hy1, hx2, hy2]


def weapon_person_match_score(weapon_box, person_box):
    wc = bbox_center(weapon_box)
    pc = bbox_center(person_box)

    dist = distance(wc, pc)
    person_h = max(1, bbox_height(person_box))
    norm_dist = dist / person_h

    overlap = iou(weapon_box, person_box)
    hand_zone = make_hand_zone(person_box)

    in_hand_zone = point_in_box(wc, hand_zone)
    inside_person = point_in_box(wc, person_box)

    score = 0.0
    score += max(0.0, 2.0 - norm_dist)
    score += 3.0 * overlap

    if inside_person:
        score += 1.5
    if in_hand_zone:
        score += 2.0

    return score


def assign_weapons_to_people(active_persons, active_objects):
    assignments = {pid: [] for pid in active_persons.keys()}

    confirmed_people = {
        pid: pdata for pid, pdata in active_persons.items()
        if pdata.get("confirmed", False)
    }

    confirmed_weapons = {
        oid: obj for oid, obj in active_objects.items()
        if obj.get("confirmed", False) and obj.get("class_name") in WEAPON_CLASSES
    }

    for oid, obj in confirmed_weapons.items():
        weapon_box = obj["bbox"]

        best_pid = None
        best_score = -1e9
        second_best_score = -1e9

        for pid, pdata in confirmed_people.items():
            person_box = pdata["bbox"]
            score = weapon_person_match_score(weapon_box, person_box)

            if score > best_score:
                second_best_score = best_score
                best_score = score
                best_pid = pid
            elif score > second_best_score:
                second_best_score = score

        if best_pid is None:
            continue

        margin = best_score - second_best_score

        if best_score < 1.2 or margin < 0.35:
            continue

        if confirmed_people[best_pid].get("identity", "unknown") == "owner" and margin < 0.60:
            continue

        assignments[best_pid].append({
            "class_name": obj["class_name"],
            "conf": obj["conf"],
            "bbox": obj["bbox"],
            "object_id": oid,
            "match_score": best_score,
            "match_margin": margin,
        })

    return assignments


def update_person_track(person_id, box, conf, source, frame=None):
    if person_id not in active_persons:
        return

    old_conf = active_persons[person_id]["conf"]
    smoothed_conf = 0.8 * old_conf + 0.2 * conf

    old_box = active_persons[person_id]["bbox"]
    smoothed_box = [
        int(0.8 * old_box[0] + 0.2 * box[0]),
        int(0.8 * old_box[1] + 0.2 * box[1]),
        int(0.8 * old_box[2] + 0.2 * box[2]),
        int(0.8 * old_box[3] + 0.2 * box[3]),
    ]

    active_persons[person_id]["bbox"] = smoothed_box
    active_persons[person_id]["conf"] = smoothed_conf
    active_persons[person_id]["missing"] = 0
    active_persons[person_id]["seen_count"] += 1
    active_persons[person_id]["confirmed"] = True
    active_persons[person_id]["source"] = source

    if frame is not None:
        enhanced_tracker.update_person(person_id, smoothed_box, frame, smoothed_conf)


def ensure_active_person_track(track_id, box, conf, source="bytetrack", frame=None):
    """Keep a visible person box alive immediately, even while identity is still resolving."""
    if track_id in active_persons:
        update_person_track(track_id, box, conf, source, frame)
        return active_persons[track_id]

    active_persons[track_id] = {
        "bbox": box,
        "conf": conf,
        "missing": 0,
        "identity": "unknown",
        "seen_count": 1,
        "confirmed": True,
        "source": source,
        "owner_confirmed": False,
        "owner_distance": 999.0,
        "owner_label": "UNKNOWN",
        "face_box_global": None,
        "owner_match_streak": 0,
        "stable_id": None,
        "identifying": True,
    }

    if frame is not None and track_id not in enhanced_tracker.fingerprints:
        enhanced_tracker.add_person(track_id, box, frame)
    return active_persons[track_id]


# -----------------------------
# MAIN LOOP
# -----------------------------
WINDOW_NAME = "Enhanced Threat Detection"
DISPLAY_W, DISPLAY_H = 640, 480  # fixed display size regardless of capture resolution

while True:
    frame = cam.read()
    if frame is None:
        time.sleep(0.01)
        continue

    frame_count += 1

    # =========================================================
    # YOLO DETECTION
    # On laptop we run a fresh detection every frame so tracking sees the
    # current person position instead of stale boxes. The async path stays as
    # an RPi fallback where throughput matters more than absolute freshness.
    # =========================================================
    if config.rpi_mode:
        _motion = _has_motion(frame)
        should_submit_yolo = bool(active_persons)
        if not should_submit_yolo:
            should_submit_yolo = (frame_count % YOLO_EVERY == 0 or yolo_runner.detections.shape[0] == 0)
        if should_submit_yolo and (_motion or active_persons):
            yolo_runner.submit(frame)
        raw_detections, yolo_result_seq = yolo_runner.latest_result()
        if yolo_result_seq != last_yolo_result_seq:
            last_yolo_result_seq = yolo_result_seq
    else:
        results = model(frame)
        raw_detections = results.xyxy[0].cpu().numpy()

    # =========================================================
    # PHASE 1: PERSON DETECTION + TRACKING
    # =========================================================
    person_rows = []
    for det in raw_detections:
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        conf = float(conf)
        class_name = class_id_to_name(cls)
        if class_name != "person":
            continue
        if conf < BYTE_TRACKER_ARGS.track_low_thresh:
            continue
        person_rows.append([float(x1), float(y1), float(x2), float(y2), conf, 0.0])

    if person_rows:
        person_boxes = Boxes(np.array(person_rows, dtype=np.float32), frame.shape[:2])
    else:
        person_boxes = Boxes(np.empty((0, 6), dtype=np.float32), frame.shape[:2])

    tracked_rows = byte_tracker.update(person_boxes, img=frame)

    previous_active_persons = active_persons
    active_persons = {}
    pending_tracks.clear()

    for tracked_row in tracked_rows:
        if len(tracked_row) < 7:
            continue

        x1, y1, x2, y2, track_id, conf, _cls = tracked_row[:7]
        track_id = int(track_id)
        bbox_list = [float(x1), float(y1), float(x2), float(y2)]
        conf = float(conf)

        prev = previous_active_persons.get(track_id, {})
        active_persons[track_id] = {
            "bbox": bbox_list,
            "conf": conf,
            "missing": 0,
            "identity": prev.get("identity", "unknown"),
            "seen_count": prev.get("seen_count", 0) + 1,
            "confirmed": True,
            "source": "ultralytics_bytetrack",
            "owner_confirmed": prev.get("owner_confirmed", False),
            "owner_distance": prev.get("owner_distance", 999.0),
            "owner_match_streak": prev.get("owner_match_streak", 0),
            "owner_label": prev.get("owner_label", "UNKNOWN"),
            "face_box_global": None,
            "stable_id": prev.get("stable_id"),
            "identifying": False,
        }

        if track_id in enhanced_tracker.fingerprints:
            enhanced_tracker.update_person(track_id, bbox_list, frame, conf)
        else:
            enhanced_tracker.add_person(track_id, bbox_list, frame)

    # ── Preserve tracks dropped temporarily by ByteTracker ──────────────────
    # ByteTracker only outputs tracks it can actively match each frame.
    # When a person moves fast or is briefly occluded, their track enters
    # ByteTracker's internal "lost" pool (kept alive for track_buffer frames)
    # but is NOT included in tracked_rows.  Without this block even a single
    # missed frame wipes the entry from active_persons, clears
    # recognized_owner_id, and forces a full face-recognition restart —
    # which is why the owner box disappears the moment you move.
    # Fix: carry every previous entry forward at the motion-predicted position
    # for up to PERSON_MISSING_FRAMES frames.  Once ByteTracker re-associates
    # the detection, the real tracked_rows entry takes over normally.
    for old_pid, old_data in previous_active_persons.items():
        if old_pid in active_persons:
            continue  # ByteTracker is still actively tracking — no action needed
        missing_count = old_data.get("missing", 0) + 1
        if missing_count <= PERSON_MISSING_FRAMES:
            predicted_box = enhanced_tracker.predict_person_position(old_pid, frame)
            kept_box = predicted_box if predicted_box is not None else old_data["bbox"]
            active_persons[old_pid] = {
                **old_data,
                "bbox": kept_box,
                "missing": missing_count,
                "source": "predicted",
            }
        else:
            # Truly gone after grace period — clean up crop cache
            person_prev_crops.pop(old_pid, None)

    enhanced_tracker.cleanup_missing_tracks(active_persons.keys())

    if recognized_owner_id is not None and recognized_owner_id not in active_persons:
        recognized_owner_id = None
        owner_lock_loss_count = 0

    # Update owner profile while the owner track is live
    if recognized_owner_id is not None and recognized_owner_id in active_persons:
        owner_box = active_persons[recognized_owner_id]["bbox"]
        update_owner_profile(frame, owner_box, pose_landmarks=threat_analyzer.get_pose_landmarks(recognized_owner_id))

        # ── Fingerprint multi-view snapshot ───────────────────────────────────
        # Every 2 frames while the owner is tracked, capture their full-body
        # appearance into the snapshot gallery.  compare_best_snapshot() then
        # matches against every stored view, so even a back-view or side-view
        # re-entry can be identified without the face being visible.
        if owner_profile is not None and frame_count % 2 == 0:
            owner_profile["fingerprint"].add_snapshot(frame, owner_box)

        # ── Live CLIP gallery snapshot ─────────────────────────────────────────
        # While the owner is tracked, periodically add their full-body crop to
        # the CLIP gallery so that CLIP can re-identify them on re-entry even
        # when their face is not visible (back-to-camera, side view, covered).
        # Without this the gallery stays empty and the CLIP recovery path never
        # fires.  Cap the gallery size so it doesn't grow without bound.
        _CLIP_GALLERY_UPDATE_EVERY = 10   # ~every 0.7 s at 15 fps
        _CLIP_GALLERY_MAX_SIZE     = 24   # at most 24 snapshots in memory/on disk
        if (clip_recognizer is not None
                and frame_count % _CLIP_GALLERY_UPDATE_EVERY == 0
                and clip_recognizer.get_gallery_size() < _CLIP_GALLERY_MAX_SIZE):
            H_f, W_f = frame.shape[:2]
            bx1 = max(0, int(owner_box[0])); by1 = max(0, int(owner_box[1]))
            bx2 = min(W_f, int(owner_box[2])); by2 = min(H_f, int(owner_box[3]))
            if bx2 > bx1 and by2 > by1:
                _body_crop = frame[by1:by2, bx1:bx2]
                if _body_crop.size > 0:
                    _clip_emb = clip_recognizer.embed_image(_body_crop)
                    if _clip_emb is not None:
                        # Only add if sufficiently different from existing entries
                        # (avoids storing near-duplicate frames)
                        _is_novel = all(
                            clip_cosine_similarity(_clip_emb, ref) < 0.97
                            for ref in clip_recognizer.owner_gallery
                        )
                        if _is_novel or clip_recognizer.get_gallery_size() == 0:
                            clip_recognizer.add_to_gallery(_clip_emb, label="live")

    # Every FACE_UPDATE_EVERY frames: try to extract / refresh face embeddings
    # for all active persons so the reid_store stays up to date.
    # For the owner specifically, also keep owner_profile["face_embedding"] fresh
    # so that recover_owner_by_profile can re-identify them by face if they
    # disappear and reappear without showing their face again.
    if frame_count % FACE_UPDATE_EVERY == 0 or (recognized_owner_id is not None and frame_count % 15 == 0):
        for pid, pdata in list(active_persons.items()):
            emb = _try_extract_face_emb(frame, pdata["bbox"])
            if emb is not None:
                stable_id = pdata.get('stable_id')
                if stable_id:
                    reid_store.update_face(stable_id, emb)
                else:
                    rec = reid_store.find_match(emb, cosine_distance)
                    if rec is None:
                        rec = reid_store.register(emb, is_owner=(pid == recognized_owner_id))
                    elif pid == recognized_owner_id:
                        reid_store.mark_owner(rec["stable_id"])
                    else:
                        reid_store.touch(rec["stable_id"])
                    pdata["stable_id"] = rec["stable_id"]
                if pid == recognized_owner_id and owner_profile is not None:
                    owner_profile["face_embedding"] = emb

    # Tick ReID store to age out stale lost records
    reid_store.tick()

    # Recover owner when currently unrecognised.
    # FaceNet path runs freely (gated by owner_profile having a face_embedding).
    # CLIP fallback inside recover_owner_by_profile() is rate-limited separately.
    # A confirmation streak of RECOVERY_CONFIRM_FRAMES is required before the
    # candidate is promoted — prevents a single histogram/CLIP hit on a stranger
    # with similar clothing from triggering a false owner lock.
    _should_recover = recognized_owner_id is None and active_persons
    if _should_recover:
        recovered_pid = recover_owner_by_profile(frame)
        if recovered_pid is not None:
            if recovered_pid == _recovery_candidate_pid:
                _recovery_candidate_streak += 1
            else:
                _recovery_candidate_pid    = recovered_pid
                _recovery_candidate_streak = 1

            if _recovery_candidate_streak >= RECOVERY_CONFIRM_FRAMES:
                recognized_owner_id        = _recovery_candidate_pid
                _recovery_candidate_pid    = None
                _recovery_candidate_streak = 0
                owner_lock_loss_count      = 0
                owner_prediction_streak    = 0
                owner_box = active_persons[recognized_owner_id]["bbox"]
                update_owner_profile(frame, owner_box, pose_landmarks=threat_analyzer.get_pose_landmarks(recognized_owner_id))
        else:
            # No match this frame — decay streak so a single missed frame doesn't
            # reset a near-confirmed candidate, but a sustained absence does.
            _recovery_candidate_streak = max(0, _recovery_candidate_streak - 1)
            if _recovery_candidate_streak == 0:
                _recovery_candidate_pid = None
    else:
        _recovery_candidate_pid    = None
        _recovery_candidate_streak = 0

    # Owner recognition: search for owner's face inside each YOLO person box
    # IMPROVED: Run more frequently (every 10 frames instead of 30) and also when owner confidence is low
    owner_needs_check = (
        recognized_owner_id is None or 
        (recognized_owner_id is not None and active_persons.get(recognized_owner_id, {}).get("owner_distance", 999) > 0.5)
    )
    if (
        OWNER_RECOGNITION_ENABLED
        and owner_needs_check
        and frame_count % 10 == 0  # Run every 10 frames instead of 30
    ):
        run_owner_recognition_in_yolo_boxes(frame)

    # Sync identity labels
    # IMPROVED: Add temporal consistency to prevent rapid owner switching
    effective_owner_id = recognized_owner_id if recognized_owner_id in active_persons else None
    
    # Track owner stability - don't switch owner too frequently
    if effective_owner_id != recognized_owner_id:
        if recognized_owner_id is not None:
            # Owner was lost, require stronger evidence to reassign
            owner_lock_loss_count += 1
        else:
            # No current owner, can assign more easily
            pass
    
    for pid in active_persons:
        if pid == recognized_owner_id:
            active_persons[pid]["identity"] = "owner"
            active_persons[pid]["owner_confirmed"] = True
        else:
            active_persons[pid]["identity"] = "unknown"
            active_persons[pid]["owner_confirmed"] = False

    # If no persons are being tracked, show a minimal overlay and skip Phase 2
    if not active_persons:
        cv2.putText(frame, "No persons detected", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.imshow(WINDOW_NAME, cv2.resize(frame, (DISPLAY_W, DISPLAY_H)) if frame.shape[1] != DISPLAY_W or frame.shape[0] != DISPLAY_H else frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    # =========================================================
    # PHASE 2: FULL ANALYSIS (objects / weapons, threat scoring,
    # behavior, drawing) — only reached when persons are present
    # =========================================================

    # --- Object / weapon detection from the same YOLO result ---
    current_object_detections = []

    for det in raw_detections:
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        conf = float(conf)

        class_name = class_id_to_name(cls)
        if class_name is None or class_name == "person":
            continue

        box = [int(x1), int(y1), int(x2), int(y2)]

        matched_id = match_detection_to_tracks(
            box, active_objects, iou_threshold=0.20, dist_threshold=180
        )

        if matched_id is None:
            if not should_accept_new(class_name, conf):
                continue
        else:
            if not should_keep_existing(class_name, conf):
                continue

        current_object_detections.append({
            "bbox": box,
            "conf": conf,
            "class_name": class_name,
        })

    # --- Update object tracks ---
    matched_object_ids = set()

    for det in current_object_detections:
        box = det["bbox"]
        conf = det["conf"]
        candidate_class = det["class_name"]

        object_id = match_detection_to_tracks(
            box, active_objects, iou_threshold=0.20, dist_threshold=180
        )

        if object_id is None:
            object_id = next_object_id
            next_object_id += 1

            active_objects[object_id] = {
                "bbox": box,
                "conf": conf,
                "class_name": candidate_class,
                "missing": 0,
                "seen_count": 1,
                "confirmed": False,
            }

            object_class_history[object_id] = deque(maxlen=5)
            object_class_history[object_id].append(candidate_class)

        else:
            old_conf = active_objects[object_id]["conf"]
            smoothed_conf = 0.7 * old_conf + 0.3 * conf

            old_box = active_objects[object_id]["bbox"]
            smoothed_box = [
                int(0.7 * old_box[0] + 0.3 * box[0]),
                int(0.7 * old_box[1] + 0.3 * box[1]),
                int(0.7 * old_box[2] + 0.3 * box[2]),
                int(0.7 * old_box[3] + 0.3 * box[3]),
            ]

            active_objects[object_id]["bbox"] = smoothed_box
            active_objects[object_id]["conf"] = smoothed_conf
            active_objects[object_id]["missing"] = 0
            active_objects[object_id]["seen_count"] += 1

            maybe_switch_object_class(object_id, candidate_class, conf)

        needed = confirm_needed(active_objects[object_id]["class_name"])
        if active_objects[object_id]["seen_count"] >= needed:
            active_objects[object_id]["confirmed"] = True

        matched_object_ids.add(object_id)

    for oid in list(active_objects.keys()):
        if oid not in matched_object_ids:
            active_objects[oid]["missing"] += 1

            if oid in switch_candidate_history:
                switch_candidate_history[oid].clear()

            if active_objects[oid]["missing"] > OBJECT_MISSING_FRAMES:
                del active_objects[oid]
                if oid in object_class_history:
                    del object_class_history[oid]
                if oid in switch_candidate_history:
                    del switch_candidate_history[oid]

    threat_analyzer.cleanup_missing_tracks(active_persons.keys())

    # --- Obstacle Detection ---
    obstacle, obstacle_area = obstacle_detector.get_nearest_obstacle(frame, raw_detections)

    # --- Global weapon ownership ---
    weapon_assignments = assign_weapons_to_people(active_persons, active_objects)

    # --- Draw pending (identifying) persons ---
    # These are real detections that are still in the grace period; show a
    # subtle gray box so the person is visible but not yet labelled.
    _fH_draw, _fW_draw = frame.shape[:2]
    for _pt_id, _pt in pending_tracks.items():
        if _pt_id in active_persons:
            continue
        _px1, _py1, _px2, _py2 = [int(v) for v in _pt['bbox']]
        _px1 = max(0, min(_px1, _fW_draw - 1))
        _py1 = max(0, min(_py1, _fH_draw - 1))
        _px2 = max(0, min(_px2, _fW_draw - 1))
        _py2 = max(0, min(_py2, _fH_draw - 1))
        if _px2 > _px1 and _py2 > _py1:
            cv2.rectangle(frame, (_px1, _py1), (_px2, _py2), (160, 160, 160), 1)
            elapsed = time.time() - _pt['first_seen']
            dots = "." * (int(elapsed * 3) % 4 + 1)   # animated "..."
            cv2.putText(frame, f"ID?{dots}", (_px1, max(14, _py1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    # --- Analyze and draw all confirmed persons ---
    owner_bbox = active_persons[effective_owner_id]["bbox"] if effective_owner_id in active_persons else None

    for pid, person_data in active_persons.items():
        if not person_data["confirmed"]:
            continue

        pbox = person_data["bbox"]
        assigned_weapons = weapon_assignments.get(pid, [])

        is_real_owner = (pid == recognized_owner_id)
        identity = "owner" if is_real_owner else "unknown"

        # Per-person pixel change: check if this person visibly moved since
        # the last frame. If not, MediaPipe models will use cached results
        # (no re-inference needed — saves significant CPU on RPi).
        if len(pbox) != 4:
            print(f"[ERROR] Invalid bbox for person {pid}: {pbox}")
            continue
        x1c, y1c, x2c, y2c = [int(v) for v in pbox]
        fH, fW = frame.shape[:2]
        x1c = max(0, min(x1c, fW - 1))
        x2c = max(0, min(x2c, fW - 1))
        y1c = max(0, min(y1c, fH - 1))
        y2c = max(0, min(y2c, fH - 1))
        person_moved = True
        if x2c > x1c and y2c > y1c:
            crop_gray = cv2.cvtColor(frame[y1c:y2c, x1c:x2c], cv2.COLOR_BGR2GRAY)
            prev_crop  = person_prev_crops.get(pid)
            if prev_crop is not None and prev_crop.shape == crop_gray.shape:
                diff     = cv2.absdiff(crop_gray, prev_crop)
                ch_ratio = np.count_nonzero(diff > 20) / max(1, diff.size)
                person_moved = ch_ratio > PERSON_MOVEMENT_THRESHOLD
            person_prev_crops[pid] = crop_gray
        person_data["person_moved"] = person_moved

        person = {
            "id": pid,
            "bbox": pbox,
            "identity": identity,
            "weapons": assigned_weapons,
        }

        context = {
            "mode": "indoor",
            "owner_present": effective_owner_id in active_persons,
            "time_of_day": "day",
            "owner_bbox": owner_bbox,
            "owner_id": effective_owner_id,
        }

        threat_level, threat_score, explanation, debug = threat_analyzer.analyze_person(
            frame,
            person,
            context,
            frame_index=frame_count,
            person_moved=person_moved,
        )

        if len(pbox) != 4:
            print(f"[ERROR] Invalid bbox for drawing person {pid}: {pbox}")
            continue
        x1, y1, x2, y2 = [int(v) for v in pbox]
        print(f"[DEBUG] Drawing person {pid} with bbox: {pbox} → int: ({x1}, {y1}, {x2}, {y2})")

        if is_real_owner:
            color = (255, 255, 0)
        else:
            color = (0, 255, 0)
            if threat_level == "SUSPICIOUS":
                color = (0, 255, 255)
            elif threat_level == "THREAT":
                color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_real_owner else 2)

        display_id = person_data.get('stable_id') or pid
        main_label = "OWNER" if is_real_owner else f"ID {display_id} | {threat_level} {threat_score:.1f}"
        cv2.putText(frame, main_label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        info_lines = []

        if not is_real_owner:
            info_lines.append(f"threat_score: {threat_score:.1f}")
            info_lines.append(f"ID_score: {debug.get('identity_score', 0.0):.1f}")
            info_lines.append(f"weapon_score: {debug.get('weapon_score', 0.0):.1f}")
            info_lines.append(f"aggr_score: {debug.get('aggression_score', 0.0):.2f}")
            info_lines.append(f"aggr_pts: {debug.get('aggression_points', 0.0):.1f}")
            info_lines.append(f"behavior: {debug.get('behavior_score', 0.0):.1f}")
            info_lines.append(f"owner_prox: {debug.get('owner_proximity_score', 0.0):.1f}")
            info_lines.append(f"robot_dist: {debug.get('robot_distance_score', 0.0):.1f}")

        info_lines.append(f"person_conf: {person_data['conf']:.2f}")
        info_lines.append(f"source: {person_data.get('source', 'yolo')}")
        if person_data.get("identifying", False):
            info_lines.append("tracking: identifying")

        if person_data.get("owner_distance", 999.0) < 999.0:
            info_lines.append(
                f"owner_dist: {person_data['owner_distance']:.3f} | streak: {person_data.get('owner_match_streak', 0)}"
            )

        if assigned_weapons:
            weapon_text = ", ".join(
                [f"{w['class_name']}({w['conf']:.2f})" for w in assigned_weapons[:3]]
            )
            info_lines.append(f"weapons: {weapon_text}")
        else:
            info_lines.append("weapons: none")

        draw_text_block(frame, info_lines, x1, min(frame.shape[0] - 110, y2 + 20), color)

        face_box = person_data.get("face_box_global", None)
        if face_box is not None:
            fx1, fy1, fx2, fy2 = face_box
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (200, 200, 0), 1)

    # --- Draw confirmed objects ---
    for oid, obj in active_objects.items():
        if not obj["confirmed"]:
            continue

        x1, y1, x2, y2 = obj["bbox"]
        label = f"{obj['class_name']} {obj['conf']:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # --- Enhanced visualization ---
    dz_w = int(CAM_W * 0.5)
    dz_h = int(CAM_H * 0.33)
    dz_x1 = CAM_W // 2 - dz_w // 2
    dz_y1 = CAM_H - dz_h - 10
    danger_zone = (dz_x1, dz_y1, dz_x1 + dz_w, dz_y1 + dz_h)

    # Draw confidence bar for owner if available
    if recognized_owner_id is not None and recognized_owner_id in active_persons:
        owner_conf = active_persons[recognized_owner_id]["conf"]
        draw_confidence_bar(frame, owner_conf, x=10, y=60)

    summary_owner = f"OWNER ID {recognized_owner_id}" if recognized_owner_id in active_persons else "NONE"
    cv2.putText(
        frame,
        f"Owner: {summary_owner}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )

    cv2.imshow(WINDOW_NAME, cv2.resize(frame, (DISPLAY_W, DISPLAY_H)) if frame.shape[1] != DISPLAY_W or frame.shape[0] != DISPLAY_H else frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cam.stop()
cv2.destroyAllWindows()
