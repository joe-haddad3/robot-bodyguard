# The orchestrator: It opens the camera, runs YOLO, feeds person detections into ByteTrack, 
# runs owner recognition, assigns weapons to people, calls threat analysis, and draws 
# everything on screen.

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
from multi_person_recognizer import MultiPersonRecognizer, cosine_distance
from enhanced_tracking import (
    MotionPredictor,
    BodyguardConfig, EnhancedPersonTracker, PersonReIDStore,
    draw_angle_indicator, draw_confidence_bar
)
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.engine.results import Boxes
import shared_state

# Helper to distinguish which recognizer is active without circular imports
# “Try to get the attribute recognizer_type from the object”
# If it exists → use its value
# If it doesn’t exist → use default 'facenet'
def _is_clip(recognizer):
    return getattr(recognizer, 'recognizer_type', 'facenet') == 'clip'

# -----------------------------
# CONFIGURATION
# -----------------------------
import platform as _platform

# Auto-detect Raspberry Pi (ARM CPU or device-tree model file present)
_IS_RPI = (
    os.path.exists("/proc/device-tree/model") or
    _platform.machine() in ("aarch64", "armv7l", "armv6l")
)

config = BodyguardConfig()
if _IS_RPI:
    config.update_for_rpi()

# C270 webcam supports 640×480 @ 30 fps — use full 30 fps regardless of mode
CAM_W   = config.camera_width
CAM_H   = config.camera_height
CAM_FPS = 30
YOLO_EVERY = config.yolo_every_n_frames if config.rpi_mode else 1

# Camera settings
CAMERA_INDEX = 0 # Default camera index

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

# Demo tuning: bias toward smoother person continuity in sparse scenes.
# Keep wrong boxes from lingering on screen, but allow a slightly longer grace
# so blur / brief occlusion does not instantly drop the box.
OBJECT_MISSING_FRAMES = config.object_missing_frames
NON_OWNER_PREDICTION_FRAMES = 24  # lock: keep non-owner box for ~0.8 s without detection
OWNER_PREDICTION_FRAMES = 60      # lock: keep owner box for ~2.0 s without detection
MIN_SEEN_FRAMES_FOR_PREDICTION = 2
OBJECT_MISSING_FRAMES = min(OBJECT_MISSING_FRAMES, 2)

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
FACE_UPDATE_EVERY = 30

# Face detection (Haar/MTCNN box finding) runs every frame.
# FaceNet embedding + identification only runs every N frames — expensive on Pi.
FACE_RECOGNITION_EVERY = 5 if config.rpi_mode else 1
_face_recog_counter    = 0

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
OWNER_CONFIRM_FRAMES = 2                                     # 2 consecutive matches — runs every frame so 2 is already strict
OWNER_LOST_FRAMES_THRESHOLD = config.owner_lost_frames_threshold
FACE_EXPAND = config.face_expand
MIN_OWNER_FACE_SIZE = config.min_owner_face_size

# When a real face is detected by Haar, FaceNet is gated on this strict threshold.
# Both mean-embedding distance AND best-sample distance must clear this threshold.
# 0.30 = FaceNetOwnerRecognizer's own default (enrolled samples score 0.02–0.15,
# live match ~0.20–0.28, strangers typically 0.45+).
FACENET_STRICT_THRESHOLD = 0.35


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

# track_id → identify result dict for enrolled non-owner persons (family members)
recognized_trusted_ids: dict = {}

# track_id → last alert time (unix seconds) — 30-second cooldown per person
alert_cooldowns: dict = {}

# Recovery confirmation — require the same candidate to match N consecutive
# frames before promoting them to recognized_owner_id.  Prevents a single
multi_recognizer = MultiPersonRecognizer(threshold=OWNER_DISTANCE_THRESHOLD)

# Enhanced tracking components
enhanced_tracker = EnhancedPersonTracker(config)

# Demo-tuned ByteTrack settings: prefer continuity over strictness in
# low-crowd scenes so people survive blur, weak lighting, and short occlusion.
BYTE_TRACKER_ARGS = SimpleNamespace(
    tracker_type="bytetrack",
    track_high_thresh=0.15,
    track_low_thresh=0.07,
    new_track_thresh=0.10,
    track_buffer=45 if not config.rpi_mode else 75,
    match_thresh=0.90,        # More permissive matching for demo continuity
    fuse_score=True,
)
byte_tracker = BYTETracker(BYTE_TRACKER_ARGS, frame_rate=CAM_FPS)

# Face-based re-identification store.
# One job: face embedding → stable display ID.
reid_store = PersonReIDStore()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# MTCNN — used only in the owner recognition path so embeddings match enrollment.
if config.rpi_mode:
    _mtcnn_recognizer = None
    print("[INIT] RPi mode: using Haar fallback for face crops to reduce latency")
else:
    try:
        from facenet_pytorch import MTCNN as _MTCNN
        _mtcnn_recognizer = _MTCNN(
            image_size=160,
            margin=20,
            keep_all=False,
            min_face_size=MIN_OWNER_FACE_SIZE,
            thresholds=[0.6, 0.7, 0.7],
            post_process=False,
            device="cpu",
        )
        print("[INIT] MTCNN loaded — owner recognition will use aligned face crops (matches enrollment)")
    except Exception as _e:
        _mtcnn_recognizer = None
        print(f"[INIT] MTCNN unavailable ({_e}) — falling back to Haar for face crops")

# Mask / face-concealment detector
from mask_detector import MaskDetector
mask_detector = MaskDetector()

frame_count = 0
raw_detections = []
last_yolo_result_seq = -1

# Per-person grayscale crop cache — used to detect whether a person
# moved since the last frame before running expensive MediaPipe models.
person_prev_crops = {}   # track_id -> grayscale crop (numpy array)

# Face-concealment streak: how many consecutive frames MTCNN failed to find a
# face in the upper region of this person's box while they were facing the camera.
# THREAT is only raised after FACE_CONCEALED_MIN_FRAMES consecutive frames so
# that one missed detection (motion blur, glance away) does not trigger an alert.
face_concealed_streak = {}   # track_id -> int
FACE_CONCEALED_MIN_FRAMES = 4


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


def owner_track_changed_enough(prev_box, curr_box):
    """Return True when the owner's box moved enough to justify faster face checks."""
    prev_center = bbox_center(prev_box)
    curr_center = bbox_center(curr_box)
    center_shift = distance(prev_center, curr_center)
    owner_scale = max(40.0, 0.5 * (bbox_height(curr_box) + bbox_width(curr_box)))
    size_ratio = abs(bbox_height(curr_box) - bbox_height(prev_box)) / max(1.0, bbox_height(prev_box))
    return center_shift > owner_scale * 0.12 or size_ratio > 0.10 or iou(prev_box, curr_box) < 0.65


def point_in_box(pt, box):
    x, y = pt
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def get_confirmed_weapon_names(weapons):
    """Return confirmed weapon classes that pass the threat thresholds."""
    confirmed = []
    for weapon in weapons:
        cls = weapon.get("class_name")
        conf = float(weapon.get("conf", 0.0))
        if cls in threat_analyzer.WEAPON_WEIGHTS and conf >= threat_analyzer.WEAPON_THRESHOLDS.get(cls, 0.50):
            confirmed.append(cls)
    return confirmed


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


def draw_boxed_label(
    frame,
    text,
    x,
    y,
    bg_color,
    text_color=(255, 255, 255),
    font_scale=0.50,
    thickness=1,
    padding=4,
):
    """Draw a filled label box with text anchored near the top-left of a bbox."""
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    text_y = max(text_h + padding + baseline, y)
    top_left = (max(0, x - padding), max(0, text_y - text_h - padding))
    bottom_right = (
        min(frame.shape[1] - 1, x + text_w + padding),
        min(frame.shape[0] - 1, text_y + baseline + padding),
    )
    cv2.rectangle(frame, top_left, bottom_right, bg_color, -1)
    cv2.putText(
        frame,
        text,
        (x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
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


def _face_tensor_to_embedding(face_tensor):
    """Convert an MTCNN-aligned face tensor into the same embedding format used at enrollment."""
    import torch as _torch
    from multi_person_recognizer import l2_normalize as _l2

    face_np = face_tensor.numpy().transpose(1, 2, 0)  # (160,160,3) in [0,255]
    face_np = (face_np / 255.0 - 0.5) / 0.5
    face_np = face_np.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    if multi_recognizer._session is not None:
        emb = multi_recognizer._session.run(None, {multi_recognizer._input_name: face_np})[0][0]
    elif multi_recognizer._pt_model is not None:
        with _torch.no_grad():
            emb = multi_recognizer._pt_model(_torch.from_numpy(face_np)).cpu().numpy()[0]
    else:
        return None
    return _l2(emb)


def _extract_aligned_face_embedding(face_bgr):
    """
    Build a runtime embedding from a face crop using the same alignment path as enrollment.
    Returns (embedding, aligned_face_rgb) or (None, None) when no usable face is found.
    """
    if not multi_recognizer._registry or face_bgr is None or face_bgr.size == 0:
        return None, None

    if _mtcnn_recognizer is not None:
        from PIL import Image as _PIL_Image
        try:
            pil_crop = _PIL_Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
            face_tensor = _mtcnn_recognizer(pil_crop)
            if face_tensor is not None:
                emb = _face_tensor_to_embedding(face_tensor)
                aligned_rgb = np.clip(
                    face_tensor.numpy().transpose(1, 2, 0), 0, 255
                ).astype(np.uint8)
                return emb, aligned_rgb
        except RuntimeError as e:
            print(f"[DEBUG] MTCNN aligned face extraction skipped: {e}")
        except Exception as e:
            print(f"[DEBUG] Aligned face extraction error skipped: {e}")

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(20, 20))
    if len(faces) == 0:
        return None, None

    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    if fw < MIN_OWNER_FACE_SIZE or fh < MIN_OWNER_FACE_SIZE:
        return None, None

    pad_x = int(fw * FACE_EXPAND)
    pad_y = int(fh * FACE_EXPAND)
    fc = face_bgr[
        max(0, fy - pad_y):min(face_bgr.shape[0], fy + fh + pad_y),
        max(0, fx - pad_x):min(face_bgr.shape[1], fx + fw + pad_x),
    ]
    if fc.size == 0:
        return None, None

    face_rgb = cv2.cvtColor(fc, cv2.COLOR_BGR2RGB)
    return multi_recognizer.embed_face_rgb(face_rgb), face_rgb


def _try_extract_face_emb(frame, box):
    """
    Try to find a face in the upper 55 % of a person box and return its FaceNet embedding.
    Uses MTCNN when available (matches enrollment alignment); falls back to Haar.
    Returns a (512,) float32 numpy array, or None if no face was found.
    """
    if not multi_recognizer._registry:
        return None
    x1, y1, x2, y2 = (int(v) for v in box)
    H, W = frame.shape[:2]
    face_y2 = y1 + int((y2 - y1) * 0.55)
    cx1 = max(0, x1)
    cy1 = max(0, y1)
    cx2 = min(W, x2)
    cy2 = min(H, face_y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    if crop.shape[0] < max(20, MIN_OWNER_FACE_SIZE // 2) or crop.shape[1] < max(20, MIN_OWNER_FACE_SIZE // 2):
        return None
    emb, _ = _extract_aligned_face_embedding(crop)
    return emb


def run_owner_recognition_in_yolo_boxes(frame):
    """
    Multi-person face recognition — two-phase design for Pi performance:

    Phase 1 (every frame): Haar/MTCNN face detection → updates face_box_global
                           so the yellow face rectangle is always fresh on screen.
    Phase 2 (every FACE_RECOGNITION_EVERY frames): FaceNet embed + identify →
                           updates owner lock, family-member trust, stranger flag.
    """
    global recognized_owner_id, owner_profile, owner_lock_loss_count, _face_recog_counter

    if not OWNER_RECOGNITION_ENABLED:
        return

    _face_recog_counter += 1
    do_recognition = bool(multi_recognizer._registry) and (_face_recog_counter % FACE_RECOGNITION_EVERY == 0)

    H, W = frame.shape[:2]

    # ── Step 1: MTCNN on (optionally downscaled) frame ───────────────────────
    _MTCNN_MAX_W = 640
    if W > _MTCNN_MAX_W:
        _scale = _MTCNN_MAX_W / W
        detect_frame = cv2.resize(frame, (_MTCNN_MAX_W, int(H * _scale)))
    else:
        _scale = 1.0
        detect_frame = frame

    from PIL import Image as _PIL
    pil_frame = _PIL.fromarray(cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB))

    if _mtcnn_recognizer is not None:
        boxes_det, probs_det = _mtcnn_recognizer.detect(pil_frame)
        if boxes_det is not None and _scale != 1.0:
            boxes_det = [[v / _scale for v in b] for b in boxes_det]
    else:
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        haar_faces = face_cascade.detectMultiScale(gray_full, 1.1, 4, minSize=(24, 24))
        if len(haar_faces) > 0:
            boxes_det = [[x, y, x+w, y+h] for (x, y, w, h) in haar_faces]
            probs_det = [1.0] * len(haar_faces)
        else:
            boxes_det, probs_det = None, None

    # Reset face-box display state
    for pdata in active_persons.values():
        pdata["face_box_global"] = None

    if boxes_det is None or len(boxes_det) == 0:
        for pdata in active_persons.values():
            pdata["owner_match_streak"] = max(0, pdata.get("owner_match_streak", 0) - 1)
        return

    best_pid       = None
    best_dist      = 999.0
    best_face_crop = None

    detected_faces = []
    for box, prob in zip(boxes_det, probs_det if probs_det is not None else []):
        if prob is None or prob < 0.70:
            continue

        bx1, by1, bx2, by2 = [int(v) for v in box]
        detected_faces.append({
            "face_box": (bx1, by1, bx2, by2),
            "center": ((bx1 + bx2) / 2, (by1 + by2) / 2),
        })

    face_assignments = match_faces_to_persons(detected_faces, active_persons)
    if not face_assignments:
        for pdata in active_persons.values():
            pdata["owner_match_streak"] = max(0, pdata.get("owner_match_streak", 0) - 1)
        return
    assigned_pids = set(face_assignments.keys())
    for pid, pdata in active_persons.items():
        if pid not in assigned_pids:
            pdata["owner_match_streak"] = max(0, pdata.get("owner_match_streak", 0) - 1)

    # ── Step 2 & 3: embed each face, identify, update per-person state ───────
    for matched_pid, face in face_assignments.items():
        matched_pdata = active_persons.get(matched_pid)
        if matched_pdata is None:
            continue
        bx1, by1, bx2, by2 = [int(v) for v in face["face_box"]]

        # Phase 1 complete: face box is now visible on screen every frame
        matched_pdata["face_box_global"] = (bx1, by1, bx2, by2)

        if not do_recognition:
            continue   # skip FaceNet this frame — detection-only pass

        pad  = 15
        cfx1 = max(0, bx1 - pad); cfy1 = max(0, by1 - pad)
        cfx2 = min(W, bx2 + pad); cfy2 = min(H, by2 + pad)
        face_bgr = frame[cfy1:cfy2, cfx1:cfx2]
        if face_bgr.size == 0:
            continue

        emb, aligned_face_rgb = _extract_aligned_face_embedding(face_bgr)
        if emb is None:
            emb = _try_extract_face_emb(frame, matched_pdata["bbox"])
            if emb is not None:
                aligned_face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        if emb is None:
            matched_pdata["owner_distance"] = 999.0
            continue

        result   = multi_recognizer.identify_from_embedding(emb)

        role     = result.get("role", "unknown")
        matched  = result.get("matched", False)
        dist     = result.get("distance", 999.0)
        is_owner = matched and role == "owner"
        is_family = matched and role == "family_member"

        matched_pdata["owner_distance"]  = float(dist)

        if is_owner:
            matched_pdata["owner_label"]           = "OWNER"
            matched_pdata["owner_match_streak"]    = matched_pdata.get("owner_match_streak", 0) + 1
            matched_pdata["face_confirmed_stranger"] = False
        elif is_family:
            matched_pdata["owner_label"]           = result.get("name", "FAMILY")
            matched_pdata["owner_match_streak"]    = 0
            matched_pdata["face_confirmed_stranger"] = False
            recognized_trusted_ids[matched_pid]    = result
        else:
            matched_pdata["owner_label"]           = "UNKNOWN"
            matched_pdata["owner_match_streak"]    = 0
            matched_pdata["face_confirmed_stranger"] = True
            # Revoke owner lock if a non-owner face is inside the locked owner box
            if matched_pid == recognized_owner_id:
                recognized_owner_id  = None
                owner_lock_loss_count = 0
            # Clear trusted status if face now identified as stranger
            recognized_trusted_ids.pop(matched_pid, None)

        # ── Step 4: lock owner after streak ──────────────────────────────────
        if (is_owner
                and matched_pid != recognized_owner_id
                and matched_pdata["owner_match_streak"] >= OWNER_CONFIRM_FRAMES
                and dist < best_dist):
            best_dist      = dist
            best_pid       = matched_pid
            best_face_crop = aligned_face_rgb

        stable_id = matched_pdata.get("stable_id")
        if stable_id:
            reid_store.update_face(stable_id, emb)

    if best_pid is not None:
        recognized_owner_id   = best_pid
        owner_lock_loss_count = 0
        owner_profile = build_owner_profile(
            active_persons[best_pid]["bbox"], best_face_crop
        )
        owner_stable = active_persons[best_pid].get("stable_id")
        if owner_stable is None and owner_profile.get("face_embedding") is not None:
            owner_rec = reid_store.find_match(owner_profile["face_embedding"], cosine_distance)
            if owner_rec is None:
                owner_rec = reid_store.register(owner_profile["face_embedding"], is_owner=True)
            owner_stable = owner_rec["stable_id"]
            active_persons[best_pid]["stable_id"] = owner_stable
        if owner_stable:
            reid_store.mark_owner(owner_stable)
        print(f"[OWNER] Locked pid={best_pid}  dist={best_dist:.3f}")
        for pid in active_persons:
            active_persons[pid]["owner_confirmed"] = (pid == recognized_owner_id)
            active_persons[pid]["identity"] = "owner" if pid == recognized_owner_id else "unknown"


def build_owner_profile(box, face_rgb=None):
    profile = {
        "face_embedding": None,
        "last_seen_box": box,
        "last_seen_frame": frame_count,
        "lost_frames": 0,
    }
    if face_rgb is not None and multi_recognizer._registry:
        profile["face_embedding"] = multi_recognizer.embed_face_rgb(face_rgb)
    return profile


def update_owner_profile(box, face_rgb=None):
    global owner_profile
    if owner_profile is None:
        owner_profile = build_owner_profile(box, face_rgb)
        return
    owner_profile["last_seen_box"] = box
    owner_profile["last_seen_frame"] = frame_count
    owner_profile["lost_frames"] = 0
    if face_rgb is not None and multi_recognizer._registry:
        owner_profile["face_embedding"] = multi_recognizer.embed_face_rgb(face_rgb)


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
        if pdata.get("confirmed", False) and pdata.get("source") != "predicted"
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


def weapon_has_plausible_holder(weapon_box, persons, min_score=1.2):
    """Reject floating weapon detections that are not spatially plausible for any person."""
    confirmed_people = [
        pdata for pdata in persons.values()
        if pdata.get("confirmed", False)
    ]
    if not confirmed_people:
        return False

    best_score = max(
        weapon_person_match_score(weapon_box, pdata["bbox"])
        for pdata in confirmed_people
    )
    return best_score >= min_score


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
        "face_confirmed_stranger": False,  # True = face seen + FaceNet said UNKNOWN → block body-based promotion
    }

    if frame is not None and track_id not in enhanced_tracker.fingerprints:
        enhanced_tracker.add_person(track_id, box, frame)
    return active_persons[track_id]


# -----------------------------
# MAIN LOOP
# -----------------------------
WINDOW_NAME = "Enhanced Threat Detection"
DISPLAY_W, DISPLAY_H = 640, 480  # fixed display size regardless of capture resolution
HEADLESS = not bool(os.environ.get("DISPLAY", ""))

# -----------------------------
# FLASK API SERVER (daemon thread)
# Starts the enrollment + detection API on port 5000 so the mobile app
# can connect while the detection loop runs in the main thread.
# -----------------------------
import enrollment_server as _enrollment_server
_flask_thread = threading.Thread(
    target=lambda: _enrollment_server.app.run(
        host="0.0.0.0", port=5000, use_reloader=False, threaded=True
    ),
    daemon=True,
)
_flask_thread.start()
print("[INIT] Flask API server started on port 5000")

ALERT_COOLDOWN_SECONDS = 30

print("[INIT] Waiting for app to send POST /api/v1/start ...")

while True:
    if not shared_state.is_running():
        time.sleep(0.1)
        continue

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

    current_person_boxes = [row[:4] for row in person_rows]

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
            "face_confirmed_stranger": prev.get("face_confirmed_stranger", False),
            "last_threat_level": prev.get("last_threat_level", "SAFE"),
            "last_threat_score": prev.get("last_threat_score", 0.0),
        }

        if track_id in enhanced_tracker.fingerprints:
            enhanced_tracker.update_person(track_id, bbox_list, frame, conf)
        else:
            enhanced_tracker.add_person(track_id, bbox_list, frame)

    # ── Re-merge: recover identity when ByteTrack assigns a new ID ──────────
    # When a person is briefly lost, their old track_id enters the grace period
    # (predicted boxes below).  When ByteTrack re-detects them it issues a FRESH
    # track_id — that new entry has no history and starts as "unknown".
    # This pass checks every brand-new track_id against predicted boxes from the
    # previous frame.  If the overlap is good enough (IoU ≥ 0.30) we inherit the
    # old entry's identity, stable_id, and owner state so the person doesn't
    # reset every time they are briefly occluded.
    # Also updates recognized_owner_id when the owner returns under a new ID.
    predicted_prev_pids = {
        pid for pid, pdata in previous_active_persons.items()
        if pid not in active_persons and pdata.get("missing", 0) > 0
    }
    for new_pid, new_data in list(active_persons.items()):
        if new_pid in previous_active_persons:
            continue   # continuous track — no merge needed
        new_box  = new_data["bbox"]
        best_iou = 0.0
        best_old = None
        for old_pid in predicted_prev_pids:
            old_box  = previous_active_persons[old_pid]["bbox"]
            score    = iou(new_box, old_box)
            if score > best_iou:
                best_iou = score
                best_old = old_pid
        if best_old is not None and best_iou >= 0.30:
            old_data = previous_active_persons[best_old]
            # Inherit appearance/history so face recognition can quickly re-confirm.
            # NEVER inherit the owner lock itself — a stranger walking through the
            # owner's predicted box position must not steal recognized_owner_id.
            # Face recognition runs every frame and re-locks within 1-2 frames if
            # the face genuinely matches.
            for field in ("stable_id", "owner_distance", "owner_match_streak",
                          "owner_label", "face_confirmed_stranger", "seen_count",
                          "last_threat_level", "last_threat_score"):
                if field in old_data:
                    active_persons[new_pid][field] = old_data[field]
            trusted_info = recognized_trusted_ids.pop(best_old, None)
            if trusted_info is not None:
                recognized_trusted_ids[new_pid] = trusted_info
            # identity and owner_confirmed stay at defaults ("unknown", False).
            # recognized_owner_id is NOT re-pointed here; the grace-period predicted
            # box continues holding the owner label while face recognition verifies.
            predicted_prev_pids.discard(best_old)

    # ── Preserve tracks dropped temporarily by ByteTracker ──────────────────
    # ByteTracker only outputs tracks it can actively match each frame.
    # When a person moves fast or is briefly occluded, their track enters
    # ByteTracker's internal "lost" pool (kept alive for track_buffer frames)
    # but is NOT included in tracked_rows.  Without this block even a single
    # missed frame wipes the entry from active_persons, clears
    # recognized_owner_id, and forces a full face-recognition restart —
    # which is why the owner box disappears the moment you move.
    # Fix: carry only mature tracks forward briefly at the motion-predicted
    # position. Normal tracks get a very short grace, while the locked owner
    # gets a slightly longer one so their box survives brief blur/occlusion
    # without letting false boxes linger on screen.
    for old_pid, old_data in previous_active_persons.items():
        if old_pid in active_persons:
            continue  # ByteTracker is still actively tracking — no action needed
        missing_count = old_data.get("missing", 0) + 1
        was_owner = old_pid == recognized_owner_id or old_data.get("owner_confirmed", False)
        seen_count = old_data.get("seen_count", 0)
        grace_frames = OWNER_PREDICTION_FRAMES if was_owner else NON_OWNER_PREDICTION_FRAMES
        if not was_owner and seen_count < MIN_SEEN_FRAMES_FOR_PREDICTION:
            grace_frames = 0

        if missing_count <= grace_frames:
            predicted_box = enhanced_tracker.predict_person_position(old_pid, frame)
            kept_box = predicted_box if predicted_box is not None else old_data["bbox"]

            # Lock invalidation: drop the box if it has drifted entirely off-screen.
            # This prevents a predicted box from chasing someone who left the frame.
            fH_lock, fW_lock = frame.shape[:2]
            lx1, ly1, lx2, ly2 = [int(v) for v in kept_box]
            if lx2 <= 0 or lx1 >= fW_lock or ly2 <= 0 or ly1 >= fH_lock:
                person_prev_crops.pop(old_pid, None)
                face_concealed_streak.pop(old_pid, None)
                continue  # box is off-screen — release the lock

            # Identity-break: if face recognition recently flagged this locked
            # track as a confirmed stranger while it held the owner label, the
            # owner reference was already cleared by run_owner_recognition_in_yolo_boxes.
            # Mirror that here so the locked box doesn't keep "owner" styling.
            if old_pid == recognized_owner_id and old_data.get("face_confirmed_stranger", False):
                recognized_owner_id = None
                owner_lock_loss_count = 0

            # Drop stale locks once the current YOLO pass no longer sees a person
            # near the predicted box. Keep the first missing frame to absorb a
            # single detector hiccup; after that, no support means no person there.
            has_person_support = any(
                iou(kept_box, det_box) >= 0.10
                for det_box in current_person_boxes
            )
            if missing_count > 1 and not has_person_support:
                person_prev_crops.pop(old_pid, None)
                face_concealed_streak.pop(old_pid, None)
                continue

            active_persons[old_pid] = {
                **old_data,
                "bbox": kept_box,
                "missing": missing_count,
                "source": "predicted",
            }
        else:
            # Truly gone after grace period — clean up per-person caches
            person_prev_crops.pop(old_pid, None)
            face_concealed_streak.pop(old_pid, None)
            recognized_trusted_ids.pop(old_pid, None)
            alert_cooldowns.pop(old_pid, None)

    enhanced_tracker.cleanup_missing_tracks(active_persons.keys())

    if recognized_owner_id is not None and recognized_owner_id not in active_persons:
        recognized_owner_id = None
        owner_lock_loss_count = 0

    # Update owner profile while the owner track is live
    if recognized_owner_id is not None and recognized_owner_id in active_persons:
        owner_box = active_persons[recognized_owner_id]["bbox"]
        update_owner_profile(owner_box)


    # Every FACE_UPDATE_EVERY frames: refresh face embeddings for all tracked persons.
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

    # Security mode split:
    #   HOME -> keep the "full" owner-recognition path active all the time so
    #           owner/family labels stay current and the owner lock can break
    #           immediately if the visible face changes.
    #   AWAY -> still run face recognition whenever persons are present, but the
    #           downstream policy is stricter: any visible unknown person becomes
    #           an immediate THREAT.
    current_security_mode = shared_state.get_mode()
    if OWNER_RECOGNITION_ENABLED and active_persons:
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
        elif pid in recognized_trusted_ids:
            active_persons[pid]["identity"] = "family_member"
            active_persons[pid]["owner_confirmed"] = False
        else:
            active_persons[pid]["identity"] = "unknown"
            active_persons[pid]["owner_confirmed"] = False

    # Push owner presence to shared_state so Flask API stays up to date
    owner_in_frame = (effective_owner_id is not None)
    shared_state.set_owner_present(owner_in_frame)

    # If no persons are being tracked, show a minimal overlay and skip Phase 2
    if not active_persons:
        cv2.putText(frame, "No persons detected", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        _empty = cv2.resize(frame, (DISPLAY_W, DISPLAY_H)) if frame.shape[1] != DISPLAY_W or frame.shape[0] != DISPLAY_H else frame
        _ok, _jpg = cv2.imencode(".jpg", _empty, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if _ok:
            shared_state.update_frame(_jpg.tobytes())
        if not HEADLESS:
            cv2.imshow(WINDOW_NAME, _empty)
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

        # Weapon detections must be spatially plausible for at least one tracked
        # person. This removes floating false positives like ceiling lights being
        # labeled as knives.
        if class_name in WEAPON_CLASSES and not weapon_has_plausible_holder(box, active_persons):
            continue

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

            # If the weapon's last known holder is still being tracked, double
            # the expiry window — the weapon is likely just momentarily occluded
            # by the person's body or a pose change.
            holder_pid   = active_objects[oid].get("holder_pid")
            holder_alive = holder_pid is not None and holder_pid in active_persons
            ttl = OBJECT_MISSING_FRAMES * 2 if holder_alive else OBJECT_MISSING_FRAMES

            if active_objects[oid]["missing"] > ttl:
                del active_objects[oid]
                if oid in object_class_history:
                    del object_class_history[oid]
                if oid in switch_candidate_history:
                    del switch_candidate_history[oid]

    # --- Global weapon ownership ---
    weapon_assignments = assign_weapons_to_people(active_persons, active_objects)

    # Store holder_pid back into active_objects so the next frame's expiry
    # check can extend grace when the holder person is still being tracked.
    for pid, weapons in weapon_assignments.items():
        for w in weapons:
            oid = w.get("object_id")
            if oid is not None and oid in active_objects:
                active_objects[oid]["holder_pid"] = pid

    threat_analyzer.cleanup_missing_tracks(active_persons.keys())

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
        is_trusted = (pid in recognized_trusted_ids)
        identity = (
            "owner" if is_real_owner
            else "family_member" if is_trusted
            else "unknown"
        )
        is_predicted = person_data.get("source") == "predicted"

        if len(pbox) != 4:
            print(f"[ERROR] Invalid bbox for person {pid}: {pbox}")
            continue
        x1c, y1c, x2c, y2c = [int(v) for v in pbox]
        fH, fW = frame.shape[:2]
        x1c = max(0, min(x1c, fW - 1))
        x2c = max(0, min(x2c, fW - 1))
        y1c = max(0, min(y1c, fH - 1))
        y2c = max(0, min(y2c, fH - 1))

        if is_predicted:
            # ── Predicted (locked) box ─────────────────────────────────────────
            # The box position is motion-extrapolated and may be stale.
            # Skip all expensive analysis (aggression, pose, mask, threat detectors)
            # to avoid false positives from a box that doesn't reflect reality.
            # Use the threat state cached from the last real detection frame.
            threat_level = person_data.get("last_threat_level", "SAFE")
            threat_score = person_data.get("last_threat_score", 0.0)
            explanation  = [f"LOCKED({person_data.get('missing', 0)})"]
            debug        = {}
            mask_result  = {"mask": False, "label": "UNKNOWN"}
            person_moved = False
        else:
            # ── Per-person pixel change (movement gate) ────────────────────────
            # Minimum crop change needed before re-running MediaPipe models.
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

            # ── Aggression ────────────────────────────────────────────────────
            if identity not in ("owner", "family_member"):
                threat_analyzer.update_aggression(
                    frame, pid, pbox, frame_count, person_moved=person_moved
                )

            # ── Mask / face-concealment detection ─────────────────────────────
            mask_result = {"mask": False, "label": "UNKNOWN"}
            if identity not in ("owner", "family_member") and mask_detector.enabled:
                face_y2 = y1c + int((y2c - y1c) * 0.45)
                face_crop = frame[y1c:face_y2, x1c:x2c] if face_y2 > y1c else None
                if face_crop is not None and face_crop.size > 0:
                    mask_result = mask_detector.detect(
                        face_crop, person_box_height=(y2c - y1c)
                    )

            person = {
                "id": pid,
                "bbox": pbox,
                "identity": identity,
                "weapons": assigned_weapons,
            }

            context = {
                "mode": current_security_mode.lower(),
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

            # ── Punch / slap → THREAT ─────────────────────────────────────────
            # Same check as test_behavior_analyzer_live.py:
            #   behavior_analyzer returns "Punching/slapping" in behaviors
            #   ↔  stabilized attack_motion = True
            # Only fires for unknown persons when the owner is in the frame.
            if identity not in ("owner", "family_member") and owner_bbox is not None:
                ba_signals = threat_analyzer.behavior_analyzer.get_pose_signals(
                    pid, stabilized=True
                )
                if ba_signals.get("attack_motion", False):
                    threat_level = "THREAT"
                    threat_score = 92.0
                    explanation  = ["ATTACK_MOTION"]

            # ── Face-concealment threat gate ───────────────────────────────────
            if identity not in ("owner", "family_member") and owner_bbox is not None and mask_result.get("mask"):
                _pose_lm      = threat_analyzer.get_pose_landmarks(pid)
                _nose_vis     = 0.0
                if _pose_lm is not None and len(_pose_lm) > 0:
                    _nose_vis = getattr(_pose_lm[0], 'visibility', 0.0)
                _facing_camera = _nose_vis >= 0.5
                if _facing_camera:
                    face_concealed_streak[pid] = face_concealed_streak.get(pid, 0) + 1
                else:
                    face_concealed_streak[pid] = 0
            else:
                face_concealed_streak[pid] = 0

            if face_concealed_streak.get(pid, 0) >= FACE_CONCEALED_MIN_FRAMES:
                threat_level = "SUSPICIOUS"
                threat_score = 60.0
                explanation  = ["FACE_CONCEALED"]

            # ── AWAY mode: any unknown person is an automatic THREAT ──────────
            # In HOME mode we keep the full owner-recognition / behavior-driven
            # pipeline. In AWAY mode we switch to stricter visitor screening:
            # if a visible person is not recognized as the owner or a trusted
            # family member, they are immediately escalated.
            confirmed_weapon_names = get_confirmed_weapon_names(assigned_weapons)
            if identity not in ("owner", "family_member") and owner_bbox is not None and confirmed_weapon_names:
                threat_level = "THREAT"
                threat_score = 95.0
                explanation  = [f"ARMED_UNKNOWN_WITH_OWNER:{'/'.join(confirmed_weapon_names)}"]

            current_mode = shared_state.get_mode()
            if identity not in ("owner", "family_member"):
                if current_mode == "AWAY":
                    threat_level = "THREAT"
                    threat_score = 95.0
                    explanation  = ["UNKNOWN_PERSON_AWAY_MODE"]

        # Cache so predicted-box frames can reuse the last real result
        if is_trusted:
            trusted_name = recognized_trusted_ids.get(pid, {}).get("name", "Family Member")
            threat_level = "SAFE"
            threat_score = 0.0
            explanation = [f"FAMILY_MEMBER:{trusted_name}"]

        person_data["last_threat_level"] = threat_level
        person_data["last_threat_score"] = threat_score

        # ── Alert generation (30-second cooldown per track) ───────────────────
        if identity not in ("owner", "family_member") and threat_level in ("THREAT", "SUSPICIOUS"):
            _now = time.time()
            _last = alert_cooldowns.get(pid, 0.0)
            if _now - _last >= ALERT_COOLDOWN_SECONDS:
                alert_cooldowns[pid] = _now
                _trusted_info = recognized_trusted_ids.get(pid, {})
                shared_state.add_alert(
                    level=threat_level,
                    person_id=_trusted_info.get("person_id", f"track_{pid}"),
                    name=_trusted_info.get("name", "Unknown"),
                    explanation=explanation,
                    track_id=pid,
                )

        x1, y1, x2, y2 = [int(v) for v in pbox]

        if is_real_owner:
            color = (255, 255, 0)
        elif is_trusted:
            color = (0, 0, 0)   # black box for family member
        else:
            color = (0, 255, 0)
            if threat_level == "SUSPICIOUS":
                color = (0, 255, 255)
            elif threat_level == "THREAT":
                color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_real_owner else 2)

        display_id = person_data.get('stable_id') or pid
        if is_real_owner:
            main_label = "OWNER"
        elif is_trusted:
            _tname = recognized_trusted_ids[pid].get("name", "FAMILY")
            main_label = f"FAMILY MEMBER: {_tname}"
        else:
            main_label = f"ID {display_id} | {threat_level} {threat_score:.1f}"
        if is_trusted:
            draw_boxed_label(
                frame,
                main_label,
                x1,
                max(20, y1 - 10),
                bg_color=(0, 0, 0),
                text_color=(255, 255, 255),
                font_scale=0.52,
                thickness=1,
            )
        else:
            cv2.putText(frame, main_label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        info_lines = []

        if not is_real_owner and not is_trusted:
            info_lines.append(f"threat: {threat_score:.1f} | {', '.join(explanation)}")
            info_lines.append(
                f"aggr: {debug.get('aggression_score', 0.0):.2f} | "
                f"hand: {debug.get('hand_movement_score', 0.0):.2f} | "
                f"weapon: {'yes' if debug.get('has_weapon') else 'no'}"
            )
            info_lines.append(f"face: {mask_result.get('label', 'UNKNOWN')}")
            info_lines.append(
                f"attack_motion: {debug.get('attack_motion', False)} | "
                f"raw: {debug.get('attack_motion_raw', False)}"
            )

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

    display_frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H)) if frame.shape[1] != DISPLAY_W or frame.shape[0] != DISPLAY_H else frame

    # Stream annotated frame to shared_state so the Flask /api/v1/stream endpoint
    # can serve it as MJPEG to the mobile app.
    _ok, _jpg = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if _ok:
        shared_state.update_frame(_jpg.tobytes())

    if not HEADLESS:
        cv2.imshow(WINDOW_NAME, display_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cam.stop()
if not HEADLESS:
    cv2.destroyAllWindows()
