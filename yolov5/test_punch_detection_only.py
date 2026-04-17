import cv2
import math
import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import torch

from camera_utils import make_camera
from enhanced_threat_analyzer import EnhancedThreatAnalyzer
from enhanced_tracking import (
    BodyguardConfig,
    EnhancedPersonTracker,
    PersonReIDStore,
    draw_confidence_bar,
)
from owner_recognizer_facenet import FaceNetOwnerRecognizer, cosine_distance
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker


config = BodyguardConfig()
if config.rpi_mode:
    config.update_for_rpi()


CAM_W = config.camera_width
CAM_H = config.camera_height
CAM_FPS = config.camera_fps
YOLO_EVERY = config.yolo_every_n_frames if config.rpi_mode else 1
CAMERA_INDEX = 0
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")


class AsyncYOLO:
    def __init__(self, yolo_model):
        self._model = yolo_model
        self._det_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._detections = np.empty((0, 6), dtype=np.float32)
        self._result_seq = 0
        self._new_frame = None
        self._frame_ready = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, frame):
        with self._frame_lock:
            self._new_frame = frame.copy()
        self._frame_ready.set()

    def latest_result(self):
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


FACE_MIN_SIZE = (60, 60)
FACE_UPDATE_EVERY = 15
PERSON_MOVEMENT_THRESHOLD = 0.03
NON_OWNER_PREDICTION_FRAMES = 2
OWNER_PREDICTION_FRAMES = 6
MIN_SEEN_FRAMES_FOR_PREDICTION = 3
FACENET_STRICT_THRESHOLD = 0.35

OWNER_RECOGNITION_ENABLED = config.owner_recognition_enabled
OWNER_RECOGNITION_EVERY = config.owner_recognition_every
OWNER_DISTANCE_THRESHOLD = config.owner_distance_threshold
OWNER_CONFIRM_FRAMES = 2
FACE_EXPAND = config.face_expand
MIN_OWNER_FACE_SIZE = config.min_owner_face_size

_MOTION_MSE_THRESHOLD = 150.0
_prev_gray_for_motion = None

active_persons = {}
recognized_owner_id = None
owner_lock_loss_count = 0
owner_profile = None
frame_count = 0
last_yolo_result_seq = -1
person_prev_crops = {}


def _has_motion(frame_bgr):
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


def iou(box_a, box_b):
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_w = max(0, x_b - x_a)
    inter_h = max(0, y_b - y_a)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


def draw_text_block(frame, lines, x, y, color):
    font_scale = 0.52
    thickness = 2
    line_h = 24
    for i, line in enumerate(lines):
        yy = y + i * line_h
        if 0 < yy < frame.shape[0] - 5:
            cv2.putText(
                frame,
                line,
                (x, yy),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
            )


def expand_face_box(face_box, frame_shape, expand_ratio=0.10):
    x, y, w, h = face_box
    h_frame, w_frame = frame_shape[:2]
    pad_x = int(w * expand_ratio)
    pad_y = int(h * expand_ratio)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_frame, x + w + pad_x)
    y2 = min(h_frame, y + h + pad_y)
    return (x1, y1, x2, y2)


def _try_extract_face_emb(frame, box):
    if not owner_recognizer.enabled:
        return None
    x1, y1, x2, y2 = (int(v) for v in box)
    h_frame, w_frame = frame.shape[:2]
    face_y2 = y1 + int((y2 - y1) * 0.55)
    cx1 = max(0, x1)
    cy1 = max(0, y1)
    cx2 = min(w_frame, x2)
    cy2 = min(h_frame, face_y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    if crop.shape[0] < max(20, MIN_OWNER_FACE_SIZE // 2) or crop.shape[1] < max(20, MIN_OWNER_FACE_SIZE // 2):
        return None

    if _mtcnn_recognizer is not None:
        from PIL import Image as _PIL_Image
        try:
            pil_crop = _PIL_Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            face_tensor = _mtcnn_recognizer(pil_crop)
            if face_tensor is None:
                return None
            import torch as _torch
            from owner_recognizer_facenet import l2_normalize as _l2

            face_np = face_tensor.numpy().transpose(1, 2, 0)
            face_np = (face_np / 255.0 - 0.5) / 0.5
            face_np = face_np.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
            if owner_recognizer._session is not None:
                emb = owner_recognizer._session.run(None, {owner_recognizer._input_name: face_np})[0][0]
            elif owner_recognizer._pt_model is not None:
                with _torch.no_grad():
                    emb = owner_recognizer._pt_model(_torch.from_numpy(face_np)).cpu().numpy()[0]
            else:
                return None
            return _l2(emb)
        except Exception:
            return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(20, 20))
    if len(faces) == 0:
        return None
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    if fw < MIN_OWNER_FACE_SIZE or fh < MIN_OWNER_FACE_SIZE:
        return None
    pad_x = int(fw * FACE_EXPAND)
    pad_y = int(fh * FACE_EXPAND)
    fc = crop[
        max(0, fy - pad_y):min(crop.shape[0], fy + fh + pad_y),
        max(0, fx - pad_x):min(crop.shape[1], fx + fw + pad_x),
    ]
    if fc.size == 0:
        return None
    return owner_recognizer.embed_face_rgb(cv2.cvtColor(fc, cv2.COLOR_BGR2RGB))


def build_owner_profile(box, face_rgb=None):
    profile = {
        "face_embedding": None,
        "last_seen_box": box,
        "last_seen_frame": frame_count,
        "lost_frames": 0,
    }
    if face_rgb is not None and owner_recognizer.enabled:
        profile["face_embedding"] = owner_recognizer.embed_face_rgb(face_rgb)
    return profile


def update_owner_profile(box, face_rgb=None):
    global owner_profile
    if owner_profile is None:
        owner_profile = build_owner_profile(box, face_rgb)
        return
    owner_profile["last_seen_box"] = box
    owner_profile["last_seen_frame"] = frame_count
    owner_profile["lost_frames"] = 0
    if face_rgb is not None and owner_recognizer.enabled:
        owner_profile["face_embedding"] = owner_recognizer.embed_face_rgb(face_rgb)


def run_owner_recognition_in_yolo_boxes(frame):
    global recognized_owner_id, owner_profile, owner_lock_loss_count

    if not OWNER_RECOGNITION_ENABLED or not owner_recognizer.enabled:
        return

    h_frame, w_frame = frame.shape[:2]
    max_w = 640
    if w_frame > max_w:
        scale = max_w / w_frame
        detect_frame = cv2.resize(frame, (max_w, int(h_frame * scale)))
    else:
        scale = 1.0
        detect_frame = frame

    from PIL import Image as _PIL

    pil_frame = _PIL.fromarray(cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB))
    if _mtcnn_recognizer is not None:
        boxes_det, probs_det = _mtcnn_recognizer.detect(pil_frame)
        if boxes_det is not None and scale != 1.0:
            boxes_det = [[v / scale for v in b] for b in boxes_det]
    else:
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        haar_faces = face_cascade.detectMultiScale(gray_full, 1.1, 4, minSize=(40, 40))
        if len(haar_faces) > 0:
            boxes_det = [[x, y, x + w, y + h] for (x, y, w, h) in haar_faces]
            probs_det = [1.0] * len(haar_faces)
        else:
            boxes_det, probs_det = None, None

    for pdata in active_persons.values():
        pdata["face_box_global"] = None

    if boxes_det is None or len(boxes_det) == 0:
        for pdata in active_persons.values():
            pdata["owner_match_streak"] = max(0, pdata.get("owner_match_streak", 0) - 1)
        return

    best_pid = None
    best_dist = 999.0
    best_face_crop = None

    for box, prob in zip(boxes_det, probs_det if probs_det is not None else []):
        if prob is None or prob < 0.70:
            continue

        bx1, by1, bx2, by2 = [int(v) for v in box]
        face_cx = (bx1 + bx2) // 2
        face_cy = (by1 + by2) // 2

        matched_pid = None
        matched_pdata = None
        for pid, pdata in active_persons.items():
            if not pdata.get("confirmed", False):
                continue
            px1, py1, px2, py2 = [int(v) for v in pdata["bbox"]]
            upper_y2 = py1 + int((py2 - py1) * 0.70)
            if px1 <= face_cx <= px2 and py1 <= face_cy <= upper_y2:
                matched_pid = pid
                matched_pdata = pdata
                break

        if matched_pid is None:
            continue

        pad = 15
        cfx1 = max(0, bx1 - pad)
        cfy1 = max(0, by1 - pad)
        cfx2 = min(w_frame, bx2 + pad)
        cfy2 = min(h_frame, by2 + pad)
        face_bgr = frame[cfy1:cfy2, cfx1:cfx2]
        if face_bgr.size == 0:
            continue

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        emb = owner_recognizer.embed_face_rgb(face_rgb)

        mean_dist = (
            cosine_distance(emb, owner_recognizer.mean_embedding)
            if owner_recognizer.mean_embedding is not None else 999.0
        )
        sample_dists = [cosine_distance(emb, ref) for ref in owner_recognizer.owner_embeddings]
        best_sample_dist = min(sample_dists) if sample_dists else 999.0
        is_owner = (
            mean_dist < FACENET_STRICT_THRESHOLD and
            best_sample_dist < FACENET_STRICT_THRESHOLD
        )

        matched_pdata["face_box_global"] = (bx1, by1, bx2, by2)
        matched_pdata["owner_distance"] = float(mean_dist)
        matched_pdata["owner_label"] = "OWNER" if is_owner else "UNKNOWN"

        if is_owner:
            matched_pdata["owner_match_streak"] = matched_pdata.get("owner_match_streak", 0) + 1
            matched_pdata["face_confirmed_stranger"] = False
        else:
            matched_pdata["owner_match_streak"] = 0
            matched_pdata["face_confirmed_stranger"] = True
            if matched_pid == recognized_owner_id:
                recognized_owner_id = None
                owner_lock_loss_count = 0

        if (
            is_owner and
            matched_pid != recognized_owner_id and
            matched_pdata["owner_match_streak"] >= OWNER_CONFIRM_FRAMES and
            mean_dist < best_dist
        ):
            best_dist = mean_dist
            best_pid = matched_pid
            best_face_crop = face_rgb

        stable_id = matched_pdata.get("stable_id")
        if stable_id:
            reid_store.update_face(stable_id, emb)

    if best_pid is not None:
        recognized_owner_id = best_pid
        owner_lock_loss_count = 0
        owner_profile = build_owner_profile(active_persons[best_pid]["bbox"], best_face_crop)
        owner_stable = active_persons[best_pid].get("stable_id")
        if owner_stable is None and owner_profile.get("face_embedding") is not None:
            owner_rec = reid_store.find_match(owner_profile["face_embedding"], cosine_distance)
            if owner_rec is None:
                owner_rec = reid_store.register(owner_profile["face_embedding"], is_owner=True)
            owner_stable = owner_rec["stable_id"]
            active_persons[best_pid]["stable_id"] = owner_stable
        if owner_stable:
            reid_store.mark_owner(owner_stable)
        print(f"[OWNER] Locked pid={best_pid} mean={best_dist:.3f}")


cam = make_camera(src=CAMERA_INDEX, width=CAM_W, height=CAM_H, fps=CAM_FPS).start()
model = torch.hub.load(
    os.path.dirname(os.path.abspath(__file__)),
    "custom",
    path=MODEL_PATH,
    source="local",
    force_reload=False,
)
model.conf = 0.10
yolo_runner = AsyncYOLO(model)

threat_analyzer = EnhancedThreatAnalyzer(config=config)
owner_recognizer = FaceNetOwnerRecognizer(threshold=OWNER_DISTANCE_THRESHOLD)
enhanced_tracker = EnhancedPersonTracker(config)
reid_store = PersonReIDStore()

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

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

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
    print("[INIT] MTCNN loaded for punch-only test")
except Exception as exc:
    _mtcnn_recognizer = None
    print(f"[INIT] MTCNN unavailable ({exc}) — falling back to Haar")


WINDOW_NAME = "Punch Detection Only"
DISPLAY_W, DISPLAY_H = 640, 480


while True:
    frame = cam.read()
    if frame is None:
        time.sleep(0.01)
        continue

    frame_count += 1

    if config.rpi_mode:
        motion = _has_motion(frame)
        should_submit_yolo = bool(active_persons)
        if not should_submit_yolo:
            should_submit_yolo = (frame_count % YOLO_EVERY == 0)
        if should_submit_yolo and (motion or active_persons):
            yolo_runner.submit(frame)
        raw_detections, yolo_result_seq = yolo_runner.latest_result()
        if yolo_result_seq != last_yolo_result_seq:
            last_yolo_result_seq = yolo_result_seq
    else:
        results = model(frame)
        raw_detections = results.xyxy[0].cpu().numpy()

    person_rows = []
    for det in raw_detections:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) != 0:
            continue
        if float(conf) < BYTE_TRACKER_ARGS.track_low_thresh:
            continue
        person_rows.append([float(x1), float(y1), float(x2), float(y2), float(conf), 0.0])

    if person_rows:
        person_boxes = Boxes(np.array(person_rows, dtype=np.float32), frame.shape[:2])
    else:
        person_boxes = Boxes(np.empty((0, 6), dtype=np.float32), frame.shape[:2])

    tracked_rows = byte_tracker.update(person_boxes, img=frame)

    previous_active_persons = active_persons
    active_persons = {}

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
            "face_confirmed_stranger": prev.get("face_confirmed_stranger", False),
        }

        if track_id in enhanced_tracker.fingerprints:
            enhanced_tracker.update_person(track_id, bbox_list, frame, conf)
        else:
            enhanced_tracker.add_person(track_id, bbox_list, frame)

    predicted_prev_pids = {
        pid for pid, pdata in previous_active_persons.items()
        if pid not in active_persons and pdata.get("missing", 0) > 0
    }
    for new_pid, new_data in list(active_persons.items()):
        if new_pid in previous_active_persons:
            continue
        new_box = new_data["bbox"]
        best_iou = 0.0
        best_old = None
        for old_pid in predicted_prev_pids:
            score = iou(new_box, previous_active_persons[old_pid]["bbox"])
            if score > best_iou:
                best_iou = score
                best_old = old_pid
        if best_old is not None and best_iou >= 0.30:
            old_data = previous_active_persons[best_old]
            for field in (
                "identity",
                "stable_id",
                "owner_confirmed",
                "owner_distance",
                "owner_match_streak",
                "owner_label",
                "face_confirmed_stranger",
                "seen_count",
            ):
                if field in old_data:
                    active_persons[new_pid][field] = old_data[field]
            if best_old == recognized_owner_id:
                recognized_owner_id = new_pid
            predicted_prev_pids.discard(best_old)

    for old_pid, old_data in previous_active_persons.items():
        if old_pid in active_persons:
            continue
        missing_count = old_data.get("missing", 0) + 1
        was_owner = old_pid == recognized_owner_id or old_data.get("owner_confirmed", False)
        seen_count = old_data.get("seen_count", 0)
        grace_frames = OWNER_PREDICTION_FRAMES if was_owner else NON_OWNER_PREDICTION_FRAMES
        if not was_owner and seen_count < MIN_SEEN_FRAMES_FOR_PREDICTION:
            grace_frames = 0

        if missing_count <= grace_frames:
            predicted_box = enhanced_tracker.predict_person_position(old_pid, frame)
            kept_box = predicted_box if predicted_box is not None else old_data["bbox"]
            active_persons[old_pid] = {
                **old_data,
                "bbox": kept_box,
                "missing": missing_count,
                "source": "predicted",
            }
        else:
            person_prev_crops.pop(old_pid, None)

    enhanced_tracker.cleanup_missing_tracks(active_persons.keys())

    if recognized_owner_id is not None and recognized_owner_id not in active_persons:
        recognized_owner_id = None
        owner_lock_loss_count = 0

    if recognized_owner_id is not None and recognized_owner_id in active_persons:
        update_owner_profile(active_persons[recognized_owner_id]["bbox"])

    if frame_count % FACE_UPDATE_EVERY == 0 or (recognized_owner_id is not None and frame_count % 15 == 0):
        for pid, pdata in list(active_persons.items()):
            emb = _try_extract_face_emb(frame, pdata["bbox"])
            if emb is not None:
                stable_id = pdata.get("stable_id")
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

    reid_store.tick()

    run_recognition = (
        OWNER_RECOGNITION_ENABLED and
        (not config.rpi_mode or frame_count % OWNER_RECOGNITION_EVERY == 0)
    )
    if run_recognition:
        run_owner_recognition_in_yolo_boxes(frame)

    effective_owner_id = recognized_owner_id if recognized_owner_id in active_persons else None
    for pid in active_persons:
        if pid == recognized_owner_id:
            active_persons[pid]["identity"] = "owner"
            active_persons[pid]["owner_confirmed"] = True
        else:
            active_persons[pid]["identity"] = "unknown"
            active_persons[pid]["owner_confirmed"] = False

    if not active_persons:
        cv2.putText(frame, "No persons detected", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.imshow(WINDOW_NAME, cv2.resize(frame, (DISPLAY_W, DISPLAY_H)))
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    owner_bbox = active_persons[effective_owner_id]["bbox"] if effective_owner_id in active_persons else None

    for pid, person_data in active_persons.items():
        if not person_data["confirmed"]:
            continue

        pbox = person_data["bbox"]
        is_real_owner = (pid == recognized_owner_id)
        identity = "owner" if is_real_owner else "unknown"

        if len(pbox) != 4:
            continue
        x1c, y1c, x2c, y2c = [int(v) for v in pbox]
        fh, fw = frame.shape[:2]
        x1c = max(0, min(x1c, fw - 1))
        x2c = max(0, min(x2c, fw - 1))
        y1c = max(0, min(y1c, fh - 1))
        y2c = max(0, min(y2c, fh - 1))

        person_moved = True
        if x2c > x1c and y2c > y1c:
            crop_gray = cv2.cvtColor(frame[y1c:y2c, x1c:x2c], cv2.COLOR_BGR2GRAY)
            prev_crop = person_prev_crops.get(pid)
            if prev_crop is not None and prev_crop.shape == crop_gray.shape:
                diff = cv2.absdiff(crop_gray, prev_crop)
                ch_ratio = np.count_nonzero(diff > 20) / max(1, diff.size)
                person_moved = ch_ratio > PERSON_MOVEMENT_THRESHOLD
            person_prev_crops[pid] = crop_gray

        if identity != "owner":
            threat_analyzer.update_aggression(
                frame,
                pid,
                pbox,
                frame_count,
                person_moved=person_moved,
            )

        person = {
            "id": pid,
            "bbox": pbox,
            "identity": identity,
            "weapons": [],
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

        explanation_set = set(explanation)
        punch_motion_detected = (
            bool(debug.get("ua_thrust", False)) or
            ("UPPER_ARM_THRUST_AT_OWNER" in explanation_set) or
            ("ARM_ENTERING_OWNER" in explanation_set)
        )
        owner_contact_detected = (
            bool(debug.get("arm_contact", False)) or
            ("ARM_CONTACT_WITH_OWNER" in explanation_set)
        )
        strike_rule_detected = "ARM_ENTERING_OWNER" in explanation_set

        x1, y1, x2, y2 = [int(v) for v in pbox]
        if is_real_owner:
            color = (255, 255, 0)
        else:
            color = (0, 255, 0)
            if threat_level == "SUSPICIOUS":
                color = (0, 255, 255)
            elif threat_level == "THREAT":
                color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_real_owner else 2)

        display_id = person_data.get("stable_id") or pid
        if is_real_owner:
            main_label = "OWNER"
        else:
            main_label = f"ID {display_id} | {threat_level} {threat_score:.1f}"
        cv2.putText(frame, main_label, (x1, max(24, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2)

        info_lines = []
        if not is_real_owner:
            info_lines.append(f"why: {', '.join(explanation)}")
            info_lines.append(
                f"PUNCH_MOTION: {'YES' if punch_motion_detected else 'NO'} | "
                f"OWNER_CONTACT: {'YES' if owner_contact_detected else 'NO'}"
            )
            info_lines.append(
                f"STRIKE_RULE: {'YES' if strike_rule_detected else 'NO'} | "
                f"FINAL_THREAT: {'YES' if threat_level == 'THREAT' else 'NO'}"
            )
            info_lines.append(
                f"aggr:{debug.get('aggression_score', 0.0):.2f} "
                f"hand:{debug.get('hand_movement_score', 0.0):.2f}"
            )
            info_lines.append(
                f"facing:{debug.get('facing')} torso:{debug.get('torso_speed', 0.0)} "
                f"wrist:{debug.get('wrist_speed', 0.0)}"
            )
            info_lines.append(
                f"ua:{debug.get('ua_thrust', False)} ua_spd:{debug.get('ua_rel_speed', 0.0)} "
                f"contact:{debug.get('arm_contact', False)} streak:{debug.get('contact_streak', 0)}"
            )
        info_lines.append(f"person_conf: {person_data['conf']:.2f}")
        info_lines.append(f"source: {person_data.get('source', 'yolo')}")
        if person_data.get("owner_distance", 999.0) < 999.0:
            info_lines.append(
                f"owner_dist:{person_data['owner_distance']:.3f} "
                f"streak:{person_data.get('owner_match_streak', 0)}"
            )

        draw_text_block(frame, info_lines, x1, min(frame.shape[0] - 110, y2 + 20), color)

        face_box = person_data.get("face_box_global")
        if face_box is not None:
            fx1, fy1, fx2, fy2 = face_box
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (200, 200, 0), 1)

    if recognized_owner_id is not None and recognized_owner_id in active_persons:
        owner_conf = active_persons[recognized_owner_id]["conf"]
        draw_confidence_bar(frame, owner_conf, x=10, y=60)

    summary_owner = f"OWNER ID {recognized_owner_id}" if recognized_owner_id in active_persons else "NONE"
    cv2.putText(frame, f"Owner: {summary_owner}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

    display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H)) if frame.shape[:2] != (DISPLAY_H, DISPLAY_W) else frame
    cv2.imshow(WINDOW_NAME, display)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.stop()
cv2.destroyAllWindows()
