import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import torch
import math
import time
import os
from collections import deque, defaultdict

from enhanced_threat_analyzer import EnhancedThreatAnalyzer
from camera_utils import LatestFrameCamera
from owner_recognizer_facenet import FaceNetOwnerRecognizer

# -----------------------------
# PLATFORM MODE
# Set RPI_MODE = True when running on Raspberry Pi 5
# -----------------------------
RPI_MODE = False
# -----------------------------
# CAMERA / MODEL SETTINGS
# -----------------------------
CAMERA_INDEX = 0  # change if needed (RPi camera module: usually 0)
CAM_W = 640
CAM_H = 480
CAM_FPS = 15
YOLO_EVERY = 2

if RPI_MODE:
    CAM_W = 320
    CAM_H = 240
    CAM_FPS = 10
    YOLO_EVERY = 5

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")

cam = LatestFrameCamera(
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

model.conf = 0.15


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
# THRESHOLDS
# -----------------------------
ENTRY_THRESHOLDS = {
    "person": 0.50,
    "knife": 0.7,
    "gun": 0.65,
    "baseball_bat": 0.9,
    "hammer": 0.40,
}

KEEP_THRESHOLDS = {
    "person": 0.20,
    "knife": 0.35,    # once detected at 0.50, keep tracking at lower conf (no flickering)
    "gun": 0.7,      # same logic for gun
    "baseball_bat": 0.9,
    "hammer": 0.30,
}

PERSON_MISSING_FRAMES = 40
OBJECT_MISSING_FRAMES = 15

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

FACE_FALLBACK_ENABLED = True

# -----------------------------
# OWNER RECOGNITION
# -----------------------------
OWNER_RECOGNITION_ENABLED = True
OWNER_RECOGNITION_EVERY = 2
OWNER_DISTANCE_THRESHOLD = 0.30   # enrolled samples score 0.02-0.15 vs mean; live ~2x; strangers ~0.50+
OWNER_CONFIRM_FRAMES = 3          # must match 3 consecutive frames to avoid false positives
FACE_EXPAND = 0.10
MIN_OWNER_FACE_SIZE = 60

if RPI_MODE:
    # Face recognition is slow on CPU — run less often
    OWNER_RECOGNITION_EVERY = 15
    MIN_OWNER_FACE_SIZE = 30


# -----------------------------
# TRACKING STATE
# -----------------------------
next_person_id = 0
next_object_id = 0

active_persons = {}
active_objects = {}
object_class_history = {}
switch_candidate_history = defaultdict(lambda: deque(maxlen=5))

threat_analyzer = EnhancedThreatAnalyzer(rpi_mode=RPI_MODE)
recognized_owner_id = None

owner_recognizer = FaceNetOwnerRecognizer(threshold=OWNER_DISTANCE_THRESHOLD)

frame_count = 0
detections = []


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
                0.5,
                color,
                1
            )


def match_detection_to_tracks(det_box, tracks, iou_threshold=0.25, dist_threshold=250, skip_ids=None):
    if skip_ids is None:
        skip_ids = set()

    best_id = None
    best_score = -1

    det_center = bbox_center(det_box)

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

    best_dist = float("inf")
    for tid, t in tracks.items():
        if tid in skip_ids:
            continue
        track_center = bbox_center(t["bbox"])
        d = distance(det_center, track_center)
        if d < dist_threshold and d < best_dist:
            best_dist = d
            best_id = tid

    return best_id


def should_accept_new(class_name, conf):
    return conf >= ENTRY_THRESHOLDS.get(class_name, 0.50)


def should_keep_existing(class_name, conf):
    return conf >= KEEP_THRESHOLDS.get(class_name, 0.35)


def face_box_to_person_box(face_box, frame_shape):
    """Expand an MTCNN face box (x1,y1,x2,y2) to an estimated full-body box."""
    fx1, fy1, fx2, fy2 = face_box
    H, W = frame_shape[:2]
    fw = fx2 - fx1
    fh = fy2 - fy1

    x1 = max(0, int(fx1 - 0.6 * fw))
    y1 = max(0, int(fy1 - 0.4 * fh))
    x2 = min(W - 1, int(fx2 + 0.6 * fw))
    y2 = min(H - 1, int(fy2 + 3.5 * fh))

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



def detect_faces_full_frame(frame_bgr):
    # Use MTCNN (via owner_recognizer) for accurate detection + alignment.
    # Returns face_tensor (pre-aligned, ready for FaceNet) instead of raw crop.
    return owner_recognizer.detect_faces(frame_bgr)


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


def run_owner_recognition(frame):
    global recognized_owner_id

    if not OWNER_RECOGNITION_ENABLED:
        recognized_owner_id = None
        return

    if recognized_owner_id is not None and recognized_owner_id in active_persons:
        return

    for pid, pdata in active_persons.items():
        pdata["face_box_global"] = None
        pdata["owner_distance"] = 999.0
        pdata["owner_label"] = "UNKNOWN"

    detected_faces = detect_faces_full_frame(frame)
    face_assignments = match_faces_to_persons(detected_faces, active_persons)

    best_pid = None
    best_dist = 999.0

    for pid, pdata in active_persons.items():
        if not pdata["confirmed"]:
            continue

        face_info = face_assignments.get(pid, None)

        if face_info is None:
            pdata["owner_match_streak"] = 0
            continue

        pdata["face_box_global"] = face_info["face_box"]

        # recognize_tensor uses the pre-aligned MTCNN tensor — much more accurate
        result = owner_recognizer.recognize_tensor(face_info["face_tensor"])

        dist = float(result.get("distance", 999.0))
        is_owner = dist < OWNER_DISTANCE_THRESHOLD

        pdata["owner_distance"] = dist
        pdata["owner_label"] = "OWNER" if is_owner else "UNKNOWN"

        if is_owner:
            pdata["owner_match_streak"] = pdata.get("owner_match_streak", 0) + 1
        else:
            pdata["owner_match_streak"] = 0

        if pdata["owner_match_streak"] >= OWNER_CONFIRM_FRAMES and dist < best_dist:
            best_dist = dist
            best_pid = pid

    if best_pid is not None:
        recognized_owner_id = best_pid
        for pid in active_persons:
            active_persons[pid]["owner_confirmed"] = (pid == recognized_owner_id)
            active_persons[pid]["identity"] = "owner" if pid == recognized_owner_id else "unknown"


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


def update_person_track(person_id, box, conf, source):
    if person_id not in active_persons:
        return

    old_conf = active_persons[person_id]["conf"]
    smoothed_conf = 0.8 * old_conf + 0.2 * conf

    old_box = active_persons[person_id]["bbox"]
    smoothed_box = [
        int(0.7 * old_box[0] + 0.3 * box[0]),
        int(0.7 * old_box[1] + 0.3 * box[1]),
        int(0.7 * old_box[2] + 0.3 * box[2]),
        int(0.7 * old_box[3] + 0.3 * box[3]),
    ]

    active_persons[person_id]["bbox"] = smoothed_box
    active_persons[person_id]["conf"] = smoothed_conf
    active_persons[person_id]["missing"] = 0
    active_persons[person_id]["seen_count"] += 1
    active_persons[person_id]["confirmed"] = True
    active_persons[person_id]["source"] = source


# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    frame = cam.read()
    if frame is None:
        time.sleep(0.01)
        continue

    frame_count += 1

    if frame_count % YOLO_EVERY == 0:
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()

    current_person_detections = []
    current_object_detections = []

    # STEP 1: YOLO FILTERING
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        conf = float(conf)

        class_name = class_id_to_name(cls)
        if class_name is None:
            continue

        box = [int(x1), int(y1), int(x2), int(y2)]

        if class_name == "person":
            matched_id = match_detection_to_tracks(
                box,
                active_persons,
                iou_threshold=0.20,
                dist_threshold=300
            )

            if matched_id is None:
                if not should_accept_new(class_name, conf):
                    continue
            else:
                if not should_keep_existing(class_name, conf):
                    continue

            current_person_detections.append({
                "bbox": box,
                "conf": conf,
                "class_name": class_name,
                "source": "yolo"
            })

        else:
            matched_id = match_detection_to_tracks(
                box,
                active_objects,
                iou_threshold=0.20,
                dist_threshold=180
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
                "class_name": class_name
            })

    # STEP 2: FACE FALLBACK (MTCNN — same detector used for owner recognition)
    if FACE_FALLBACK_ENABLED:
        mtcnn_faces = owner_recognizer.detect_faces(frame)
        for face in mtcnn_faces:
            pseudo_person_box = face_box_to_person_box(face["face_box"], frame.shape)

            already_covered = False
            for det in current_person_detections:
                if iou(pseudo_person_box, det["bbox"]) > 0.25:
                    already_covered = True
                    break

            if already_covered:
                continue

            current_person_detections.append({
                "bbox": pseudo_person_box,
                "conf": 0.99,
                "class_name": "person",
                "source": "face_fallback"
            })

    # STEP 3: UPDATE PERSONS
    matched_person_ids = set()
    used_detection_idxs = set()

    # 3A: protect locked owner track first
    if recognized_owner_id is not None and recognized_owner_id in active_persons:
        owner_box = active_persons[recognized_owner_id]["bbox"]

        best_idx = None
        best_iou = -1.0
        best_dist = float("inf")
        owner_match_ok = False

        for i, det in enumerate(current_person_detections):
            box = det["bbox"]
            ov = iou(box, owner_box)
            d = distance(bbox_center(box), bbox_center(owner_box))

            if ov > best_iou:
                best_iou = ov
                best_dist = d
                best_idx = i
            elif abs(ov - best_iou) < 1e-6 and d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx is not None:
            det = current_person_detections[best_idx]
            box = det["bbox"]
            conf = det["conf"]
            source = det.get("source", "yolo")

            owner_match_ok = (best_iou >= 0.10) or (best_dist < 120)

            if owner_match_ok:
                update_person_track(recognized_owner_id, box, conf, source)
                matched_person_ids.add(recognized_owner_id)
                used_detection_idxs.add(best_idx)

        if not owner_match_ok:
            active_persons[recognized_owner_id]["owner_confirmed"] = False
            active_persons[recognized_owner_id]["identity"] = "unknown"
            active_persons[recognized_owner_id]["owner_match_streak"] = 0
            active_persons[recognized_owner_id]["owner_distance"] = 999.0
            active_persons[recognized_owner_id]["owner_label"] = "UNKNOWN"
            active_persons[recognized_owner_id]["face_box_global"] = None
            recognized_owner_id = None
            # track stays alive — normal PERSON_MISSING_FRAMES grace applies

    # 3B: match remaining detections to remaining tracks
    for i, det in enumerate(current_person_detections):
        if i in used_detection_idxs:
            continue

        box = det["bbox"]
        conf = det["conf"]
        source = det.get("source", "yolo")

        skip_ids = {recognized_owner_id} if recognized_owner_id is not None else set()

        person_id = match_detection_to_tracks(
            box,
            active_persons,
            iou_threshold=0.20,
            dist_threshold=300,
            skip_ids=skip_ids
        )

        if person_id is None:
            person_id = next_person_id
            next_person_id += 1

            active_persons[person_id] = {
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
            }
        else:
            update_person_track(person_id, box, conf, source)

        matched_person_ids.add(person_id)

    for pid in list(active_persons.keys()):
        if pid not in matched_person_ids:
            active_persons[pid]["missing"] += 1

            if active_persons[pid]["missing"] > PERSON_MISSING_FRAMES:
                del active_persons[pid]
                if recognized_owner_id == pid:
                    recognized_owner_id = None

    if recognized_owner_id is not None and recognized_owner_id not in active_persons:
        recognized_owner_id = None

    # STEP 4: OWNER RECOGNITION
    if (
        OWNER_RECOGNITION_ENABLED
        and recognized_owner_id is None
        and frame_count % OWNER_RECOGNITION_EVERY == 0
    ):
        run_owner_recognition(frame)

    for pid in active_persons:
        if pid == recognized_owner_id:
            active_persons[pid]["identity"] = "owner"
            active_persons[pid]["owner_confirmed"] = True
        else:
            active_persons[pid]["identity"] = "unknown"
            active_persons[pid]["owner_confirmed"] = False

    effective_owner_id = recognized_owner_id if recognized_owner_id in active_persons else None

    # STEP 5: UPDATE OBJECTS
    matched_object_ids = set()

    for det in current_object_detections:
        box = det["bbox"]
        conf = det["conf"]
        candidate_class = det["class_name"]

        object_id = match_detection_to_tracks(
            box,
            active_objects,
            iou_threshold=0.20,
            dist_threshold=180
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
                "confirmed": False
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

    # STEP 6: GLOBAL WEAPON OWNERSHIP
    weapon_assignments = assign_weapons_to_people(active_persons, active_objects)

    # STEP 7: ANALYZE ALL PERSONS
    owner_bbox = active_persons[effective_owner_id]["bbox"] if effective_owner_id in active_persons else None

    for pid, person_data in active_persons.items():
        if not person_data["confirmed"]:
            continue

        pbox = person_data["bbox"]
        assigned_weapons = weapon_assignments.get(pid, [])

        is_real_owner = (pid == recognized_owner_id)
        identity = "owner" if is_real_owner else "unknown"

        person = {
            "id": pid,
            "bbox": pbox,
            "identity": identity,
            "weapons": assigned_weapons
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
        )

        x1, y1, x2, y2 = pbox

        if is_real_owner:
            color = (255, 255, 0)
        else:
            color = (0, 255, 0)
            if threat_level == "SUSPICIOUS":
                color = (0, 255, 255)
            elif threat_level == "THREAT":
                color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_real_owner else 2)

        if is_real_owner:
            main_label = "OWNER"
        else:
            main_label = f"ID {pid} | {threat_level} {threat_score:.1f}"

        cv2.putText(
            frame,
            main_label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

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

    # STEP 8: DRAW OBJECTS
    for oid, obj in active_objects.items():
        if not obj["confirmed"]:
            continue

        x1, y1, x2, y2 = obj["bbox"]
        label = f"{obj['class_name']} {obj['conf']:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

    summary_owner = "NONE"
    if recognized_owner_id in active_persons:
        summary_owner = f"OWNER ID {recognized_owner_id}"

    cv2.putText(
        frame,
        f"Owner: {summary_owner}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.imshow("Threat Detection Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cam.stop()
cv2.destroyAllWindows()
