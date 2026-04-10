import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import torch
import math
import time
import os
import numpy as np
from collections import deque, defaultdict

from enhanced_threat_analyzer import EnhancedThreatAnalyzer
from camera_utils import LatestFrameCamera
from owner_recognizer_facenet import FaceNetOwnerRecognizer

# ─────────────────────────────────────────────────────────────────────────────
# PLATFORM MODE
# ─────────────────────────────────────────────────────────────────────────────
RPI_MODE = False

# ─────────────────────────────────────────────────────────────────────────────
# CAMERA / MODEL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
CAM_W        = 640
CAM_H        = 480
CAM_FPS      = 15
YOLO_EVERY   = 2        # run YOLO every N frames

if RPI_MODE:
    CAM_W      = 320
    CAM_H      = 240
    CAM_FPS    = 10
    YOLO_EVERY = 5

# ─────────────────────────────────────────────────────────────────────────────
# MOTION GATE
# Skip the whole detection pipeline when fewer than this fraction of pixels
# changed since the last processed frame.  Set to 0.0 to disable.
# ─────────────────────────────────────────────────────────────────────────────
MOTION_THRESHOLD = 0.10     # 10 % of pixels must change to trigger detection
MOTION_PIXEL_DIFF = 15      # gray-level difference considered "changed"

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

# ─────────────────────────────────────────────────────────────────────────────
# CLASS SETUP
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "person",
    1: "knife",
    2: "gun",
    3: "baseball_bat",
    4: "hammer",
}
WEAPON_CLASSES = {"knife", "gun", "baseball_bat", "hammer"}

# ─────────────────────────────────────────────────────────────────────────────
# DETECTION THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
ENTRY_THRESHOLDS = {
    "person":       0.50,
    "knife":        0.70,
    "gun":          0.65,
    "baseball_bat": 0.90,
    "hammer":       0.40,
}
KEEP_THRESHOLDS = {
    "person":       0.20,
    "knife":        0.35,
    "gun":          0.70,
    "baseball_bat": 0.90,
    "hammer":       0.30,
}
PERSON_MISSING_FRAMES = 40
OBJECT_MISSING_FRAMES = 15

CLASS_CONFIRM_FRAMES = {
    "person":       1,
    "knife":        2,
    "gun":          2,
    "baseball_bat": 2,
    "hammer":       2,
}
CLASS_SWITCH_CONFIRM_FRAMES = 2
CLASS_PRIORITY = {
    "gun": 4, "knife": 3, "hammer": 2, "baseball_bat": 1,
}

# ─────────────────────────────────────────────────────────────────────────────
# OWNER RECOGNITION
# Relies ONLY on the YOLO person box — no full-frame face search.
# MTCNN is run on a crop of the upper 45 % of each YOLO box, so each
# face is guaranteed to belong to that specific person.
# ─────────────────────────────────────────────────────────────────────────────
OWNER_RECOGNITION_ENABLED  = True
OWNER_RECOGNITION_EVERY    = 2      # run recognition every N frames (when no owner locked)
OWNER_DISTANCE_THRESHOLD   = 0.45   # cosine distance; strangers typically > 0.55
OWNER_CONFIRM_FRAMES       = 2      # consecutive matches needed to lock the owner
OWNER_FACE_REGION_FRAC     = 0.45   # top fraction of YOLO box expected to contain the face
MIN_OWNER_FACE_SIZE        = 60     # minimum pixel width for a valid face crop

if RPI_MODE:
    OWNER_RECOGNITION_EVERY = 15
    MIN_OWNER_FACE_SIZE     = 30

# ─────────────────────────────────────────────────────────────────────────────
# TRACKING STATE
# ─────────────────────────────────────────────────────────────────────────────
next_person_id  = 0
next_object_id  = 0
active_persons  = {}
active_objects  = {}
object_class_history     = {}
switch_candidate_history = defaultdict(lambda: deque(maxlen=5))

threat_analyzer   = EnhancedThreatAnalyzer(rpi_mode=RPI_MODE)
recognized_owner_id = None
owner_recognizer  = FaceNetOwnerRecognizer(threshold=OWNER_DISTANCE_THRESHOLD)

frame_count      = 0
detections       = []
prev_gray_small  = None     # for motion detection


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA);   interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    denom = bbox_area(boxA) + bbox_area(boxB) - interArea
    return interArea / denom if denom > 0 else 0.0

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
            cv2.putText(frame, line, (x, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def match_detection_to_tracks(det_box, tracks,
                               iou_threshold=0.25, dist_threshold=250,
                               skip_ids=None):
    if skip_ids is None:
        skip_ids = set()
    best_id    = None
    best_score = -1
    det_center = bbox_center(det_box)

    for tid, t in tracks.items():
        if tid in skip_ids:
            continue
        overlap = iou(det_box, t["bbox"])
        if overlap >= iou_threshold and overlap > best_score:
            best_score = overlap
            best_id    = tid

    if best_id is not None:
        return best_id

    best_dist = float("inf")
    for tid, t in tracks.items():
        if tid in skip_ids:
            continue
        d = distance(det_center, bbox_center(t["bbox"]))
        if d < dist_threshold and d < best_dist:
            best_dist = d
            best_id   = tid

    return best_id

def should_accept_new(class_name, conf):
    return conf >= ENTRY_THRESHOLDS.get(class_name, 0.50)

def should_keep_existing(class_name, conf):
    return conf >= KEEP_THRESHOLDS.get(class_name, 0.35)

def maybe_switch_object_class(object_id, candidate_class, candidate_conf):
    obj            = active_objects[object_id]
    current_class  = obj["class_name"]
    if current_class == candidate_class:
        switch_candidate_history[object_id].clear()
        return
    if CLASS_PRIORITY.get(candidate_class, 0) <= CLASS_PRIORITY.get(current_class, 0):
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

def make_hand_zone(person_box):
    x1, y1, x2, y2 = person_box
    w = bbox_width(person_box); h = bbox_height(person_box)
    return [int(x1 - 0.25*w), int(y1 + 0.18*h),
            int(x2 + 0.25*w), int(y1 + 0.78*h)]

def weapon_person_match_score(weapon_box, person_box):
    wc   = bbox_center(weapon_box)
    dist = distance(wc, bbox_center(person_box))
    norm = dist / max(1, bbox_height(person_box))
    hand_zone = make_hand_zone(person_box)
    score  = max(0.0, 2.0 - norm)
    score += 3.0 * iou(weapon_box, person_box)
    if point_in_box(wc, person_box):  score += 1.5
    if point_in_box(wc, hand_zone):   score += 2.0
    return score

def assign_weapons_to_people(active_persons, active_objects):
    assignments      = {pid: [] for pid in active_persons}
    confirmed_people = {pid: p for pid, p in active_persons.items() if p.get("confirmed")}
    confirmed_weapons= {oid: o for oid, o in active_objects.items()
                        if o.get("confirmed") and o.get("class_name") in WEAPON_CLASSES}

    for oid, obj in confirmed_weapons.items():
        best_pid = None; best_score = -1e9; second_best = -1e9
        for pid, pdata in confirmed_people.items():
            s = weapon_person_match_score(obj["bbox"], pdata["bbox"])
            if s > best_score:
                second_best = best_score; best_score = s; best_pid = pid
            elif s > second_best:
                second_best = s

        if best_pid is None:
            continue
        margin = best_score - second_best
        if best_score < 1.2 or margin < 0.35:
            continue
        if confirmed_people[best_pid].get("identity") == "owner" and margin < 0.60:
            continue
        assignments[best_pid].append({
            "class_name": obj["class_name"], "conf": obj["conf"],
            "bbox": obj["bbox"], "object_id": oid,
            "match_score": best_score, "match_margin": margin,
        })
    return assignments

def update_person_track(person_id, box, conf, source):
    if person_id not in active_persons:
        return
    p  = active_persons[person_id]
    ob = p["bbox"]
    p["bbox"]       = [int(0.7*ob[i] + 0.3*box[i]) for i in range(4)]
    p["conf"]       = 0.8 * p["conf"] + 0.2 * conf
    p["missing"]    = 0
    p["seen_count"] += 1
    p["confirmed"]  = True
    p["source"]     = source


# =============================================================================
# OWNER RECOGNITION — TIED TO YOLO BOX
# =============================================================================

def recognize_owner_in_yolo_box(frame, person_box):
    """
    Search for the owner's face INSIDE the YOLO person bounding box.

    Only the upper OWNER_FACE_REGION_FRAC of the box is searched —
    that is where the head/face always is.  MTCNN detects and aligns
    the face within that crop, then FaceNet embeds it.

    There is zero ambiguity about which person the face belongs to
    because we search inside their specific YOLO box.

    Returns:
        {"distance": float, "face_box": (gx1,gy1,gx2,gy2), "prob": float}
        or None if no face found in the crop.
    """
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in person_box]

    # Upper portion of the person box = face region
    face_y2 = y1 + int(OWNER_FACE_REGION_FRAC * (y2 - y1))

    # Clamp to frame boundaries
    rx1 = max(0, x1);       ry1 = max(0, y1)
    rx2 = min(W - 1, x2);   ry2 = min(H - 1, face_y2)

    if rx2 - rx1 < MIN_OWNER_FACE_SIZE or ry2 <= ry1:
        return None

    crop = frame[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return None

    faces = owner_recognizer.detect_faces(crop)
    if not faces:
        return None

    # Pick the face with highest MTCNN detection confidence
    best = max(faces, key=lambda f: f["prob"])

    result = owner_recognizer.recognize_tensor(best["face_tensor"])
    dist   = float(result.get("distance", 999.0))

    # Convert face box from crop-local to global frame coordinates
    fx1, fy1, fx2, fy2 = best["face_box"]
    global_face_box = (rx1 + fx1, ry1 + fy1, rx1 + fx2, ry1 + fy2)

    return {"distance": dist, "face_box": global_face_box, "prob": best["prob"]}


def run_owner_recognition_phase(frame):
    """
    Called once per OWNER_RECOGNITION_EVERY frames.
    Iterates over every confirmed person track, searches for the owner's
    face inside their YOLO box, and locks recognized_owner_id.
    """
    global recognized_owner_id

    if not OWNER_RECOGNITION_ENABLED:
        return

    # If owner is already locked and still tracked, nothing to do
    if recognized_owner_id is not None and recognized_owner_id in active_persons:
        return

    best_pid  = None
    best_dist = 999.0

    for pid, pdata in active_persons.items():
        if not pdata.get("confirmed"):
            continue

        result = recognize_owner_in_yolo_box(frame, pdata["bbox"])

        # Reset per-person face data on each recognition pass
        pdata["face_box_global"] = None

        if result is None:
            pdata["owner_distance"]    = 999.0
            pdata["owner_match_streak"] = 0
            continue

        dist = result["distance"]
        pdata["owner_distance"]    = dist
        pdata["face_box_global"]   = result["face_box"]

        if dist < OWNER_DISTANCE_THRESHOLD:
            pdata["owner_match_streak"] = pdata.get("owner_match_streak", 0) + 1
        else:
            pdata["owner_match_streak"] = 0

        if (pdata["owner_match_streak"] >= OWNER_CONFIRM_FRAMES
                and dist < best_dist):
            best_dist = dist
            best_pid  = pid

    if best_pid is not None:
        recognized_owner_id = best_pid
        for pid in active_persons:
            active_persons[pid]["owner_confirmed"] = (pid == recognized_owner_id)
            active_persons[pid]["identity"] = (
                "owner" if pid == recognized_owner_id else "unknown"
            )


# =============================================================================
# MAIN LOOP
# =============================================================================
while True:
    frame = cam.read()
    if frame is None:
        time.sleep(0.01)
        continue

    frame_count += 1

    # ── MOTION GATE ───────────────────────────────────────────────────────────
    # Compare a tiny 80×60 grayscale thumbnail against the previous one.
    # Skip the detection pipeline entirely when the scene has barely changed.
    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray, (80, 60), interpolation=cv2.INTER_AREA)

    if prev_gray_small is not None:
        diff           = cv2.absdiff(gray_small, prev_gray_small)
        changed_ratio  = float(np.count_nonzero(diff > MOTION_PIXEL_DIFF)) / diff.size
        motion_active  = changed_ratio >= MOTION_THRESHOLD
    else:
        changed_ratio  = 1.0
        motion_active  = True

    prev_gray_small = gray_small

    if not motion_active:
        # Scene is static — draw last-known state and skip all heavy processing
        _draw_frame(frame)
        cv2.imshow("Threat Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    # ── PHASE 1 — YOLO: PERSON DETECTION ─────────────────────────────────────
    if frame_count % YOLO_EVERY == 0:
        results    = model(frame)
        detections = results.xyxy[0].cpu().numpy()

    current_person_detections = []
    current_object_detections = []

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cls  = int(cls);  conf = float(conf)
        cname = class_id_to_name(cls)
        if cname is None:
            continue
        box = [int(x1), int(y1), int(x2), int(y2)]

        if cname == "person":
            mid = match_detection_to_tracks(box, active_persons,
                                            iou_threshold=0.20, dist_threshold=300)
            if mid is None:
                if not should_accept_new(cname, conf): continue
            else:
                if not should_keep_existing(cname, conf): continue
            current_person_detections.append(
                {"bbox": box, "conf": conf, "class_name": cname, "source": "yolo"})
        else:
            mid = match_detection_to_tracks(box, active_objects,
                                            iou_threshold=0.20, dist_threshold=180)
            if mid is None:
                if not should_accept_new(cname, conf): continue
            else:
                if not should_keep_existing(cname, conf): continue
            current_object_detections.append(
                {"bbox": box, "conf": conf, "class_name": cname})

    # ── UPDATE PERSON TRACKS ──────────────────────────────────────────────────
    matched_person_ids  = set()
    used_detection_idxs = set()

    # 1) Protect the locked owner track first
    if recognized_owner_id is not None and recognized_owner_id in active_persons:
        owner_box   = active_persons[recognized_owner_id]["bbox"]
        best_idx    = None
        best_iou_v  = -1.0
        best_dist_v = float("inf")

        for i, det in enumerate(current_person_detections):
            ov = iou(det["bbox"], owner_box)
            d  = distance(bbox_center(det["bbox"]), bbox_center(owner_box))
            if ov > best_iou_v or (abs(ov - best_iou_v) < 1e-6 and d < best_dist_v):
                best_iou_v  = ov
                best_dist_v = d
                best_idx    = i

        owner_match_ok = False
        if best_idx is not None:
            det    = current_person_detections[best_idx]
            owner_match_ok = (best_iou_v >= 0.10) or (best_dist_v < 120)
            if owner_match_ok:
                update_person_track(recognized_owner_id,
                                    det["bbox"], det["conf"],
                                    det.get("source", "yolo"))
                matched_person_ids.add(recognized_owner_id)
                used_detection_idxs.add(best_idx)

        if not owner_match_ok:
            p = active_persons[recognized_owner_id]
            p["owner_confirmed"]   = False
            p["identity"]          = "unknown"
            p["owner_match_streak"]= 0
            p["owner_distance"]    = 999.0
            p["owner_label"]       = "UNKNOWN"
            p["face_box_global"]   = None
            recognized_owner_id    = None

    # 2) Match remaining detections to remaining tracks
    for i, det in enumerate(current_person_detections):
        if i in used_detection_idxs:
            continue
        box    = det["bbox"];  conf = det["conf"]
        source = det.get("source", "yolo")
        skip   = {recognized_owner_id} if recognized_owner_id is not None else set()

        pid = match_detection_to_tracks(box, active_persons,
                                        iou_threshold=0.20, dist_threshold=300,
                                        skip_ids=skip)
        if pid is None:
            pid = next_person_id
            next_person_id += 1
            active_persons[pid] = {
                "bbox": box, "conf": conf, "missing": 0,
                "identity": "unknown", "seen_count": 1, "confirmed": True,
                "source": source, "owner_confirmed": False,
                "owner_distance": 999.0, "owner_label": "UNKNOWN",
                "face_box_global": None, "owner_match_streak": 0,
            }
        else:
            update_person_track(pid, box, conf, source)
        matched_person_ids.add(pid)

    # Age out missing persons
    for pid in list(active_persons.keys()):
        if pid not in matched_person_ids:
            active_persons[pid]["missing"] += 1
            if active_persons[pid]["missing"] > PERSON_MISSING_FRAMES:
                del active_persons[pid]
                if recognized_owner_id == pid:
                    recognized_owner_id = None

    if recognized_owner_id is not None and recognized_owner_id not in active_persons:
        recognized_owner_id = None

    # ── PHASE 1b — OWNER RECOGNITION INSIDE YOLO BOX ─────────────────────────
    if (OWNER_RECOGNITION_ENABLED
            and recognized_owner_id is None
            and frame_count % OWNER_RECOGNITION_EVERY == 0):
        run_owner_recognition_phase(frame)

    # Sync identity flags
    for pid in active_persons:
        is_owner = (pid == recognized_owner_id)
        active_persons[pid]["identity"]        = "owner" if is_owner else "unknown"
        active_persons[pid]["owner_confirmed"] = is_owner

    effective_owner_id = recognized_owner_id if recognized_owner_id in active_persons else None

    # ── PHASE 2 — OBJECTS + THREAT ANALYSIS (only if persons present) ─────────
    if active_persons:

        # Update object tracks
        matched_object_ids = set()
        for det in current_object_detections:
            box  = det["bbox"]; conf = det["conf"]; cname = det["class_name"]
            oid  = match_detection_to_tracks(box, active_objects,
                                             iou_threshold=0.20, dist_threshold=180)
            if oid is None:
                oid = next_object_id
                next_object_id += 1
                active_objects[oid] = {
                    "bbox": box, "conf": conf, "class_name": cname,
                    "missing": 0, "seen_count": 1, "confirmed": False,
                }
                object_class_history[oid] = deque(maxlen=5)
                object_class_history[oid].append(cname)
            else:
                ob  = active_objects[oid]["bbox"]
                active_objects[oid]["bbox"]      = [int(0.7*ob[i]+0.3*box[i]) for i in range(4)]
                active_objects[oid]["conf"]      = 0.7*active_objects[oid]["conf"] + 0.3*conf
                active_objects[oid]["missing"]   = 0
                active_objects[oid]["seen_count"] += 1
                maybe_switch_object_class(oid, cname, conf)

            if active_objects[oid]["seen_count"] >= confirm_needed(active_objects[oid]["class_name"]):
                active_objects[oid]["confirmed"] = True
            matched_object_ids.add(oid)

        for oid in list(active_objects.keys()):
            if oid not in matched_object_ids:
                active_objects[oid]["missing"] += 1
                if oid in switch_candidate_history:
                    switch_candidate_history[oid].clear()
                if active_objects[oid]["missing"] > OBJECT_MISSING_FRAMES:
                    del active_objects[oid]
                    object_class_history.pop(oid, None)
                    switch_candidate_history.pop(oid, None)

        threat_analyzer.cleanup_missing_tracks(active_persons.keys())
        weapon_assignments = assign_weapons_to_people(active_persons, active_objects)
        owner_bbox = active_persons[effective_owner_id]["bbox"] if effective_owner_id else None

    else:
        # No persons — skip object/threat analysis, keep last-known objects
        weapon_assignments = {}
        owner_bbox         = None

    # ── DRAW ─────────────────────────────────────────────────────────────────
    def _draw_frame(f):
        for pid, person_data in active_persons.items():
            if not person_data["confirmed"]:
                continue
            pbox = person_data["bbox"]
            x1, y1, x2, y2 = pbox
            assigned_weapons = weapon_assignments.get(pid, [])
            is_real_owner    = (pid == recognized_owner_id)
            identity         = "owner" if is_real_owner else "unknown"

            if active_persons:
                person  = {"id": pid, "bbox": pbox,
                           "identity": identity, "weapons": assigned_weapons}
                context = {"mode": "indoor",
                           "owner_present": effective_owner_id in active_persons,
                           "time_of_day": "day",
                           "owner_bbox": owner_bbox, "owner_id": effective_owner_id}
                threat_level, threat_score, _, debug = threat_analyzer.analyze_person(
                    f, person, context, frame_index=frame_count)
            else:
                threat_level = "SAFE"; threat_score = 0.0; debug = {}

            color = (255, 255, 0) if is_real_owner else (
                (0, 0, 255) if threat_level == "THREAT" else
                (0, 255, 255) if threat_level == "SUSPICIOUS" else (0, 255, 0))

            cv2.rectangle(f, (x1, y1), (x2, y2), color, 3 if is_real_owner else 2)
            label = "OWNER" if is_real_owner else f"ID {pid} | {threat_level} {threat_score:.1f}"
            cv2.putText(f, label, (x1, max(20, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            info = []
            if not is_real_owner:
                info.append(f"threat_score: {threat_score:.1f}")
                info.append(f"ID_score:     {debug.get('identity_score', 0.0):.1f}")
                info.append(f"weapon_score: {debug.get('weapon_score', 0.0):.1f}")
                info.append(f"aggr_score:   {debug.get('aggression_score', 0.0):.2f}")
                info.append(f"aggr_pts:     {debug.get('aggression_points', 0.0):.1f}")
                info.append(f"behavior:     {debug.get('behavior_score', 0.0):.1f}")
                info.append(f"owner_prox:   {debug.get('owner_proximity_score', 0.0):.1f}")
                info.append(f"robot_dist:   {debug.get('robot_distance_score', 0.0):.1f}")

            info.append(f"person_conf:  {person_data['conf']:.2f}")
            info.append(f"source:       {person_data.get('source','yolo')}")
            info.append(f"motion:       {changed_ratio*100:.1f}%")
            dist_v  = person_data.get("owner_distance", 999.0)
            streak  = person_data.get("owner_match_streak", 0)
            info.append(f"owner_dist:   {dist_v:.3f} | streak: {streak}")

            if assigned_weapons:
                wt = ", ".join(f"{w['class_name']}({w['conf']:.2f})"
                               for w in assigned_weapons[:3])
                info.append(f"weapons: {wt}")
            else:
                info.append("weapons: none")

            draw_text_block(f, info, x1, min(f.shape[0]-110, y2+20), color)

            fb = person_data.get("face_box_global")
            if fb is not None:
                cv2.rectangle(f, (fb[0], fb[1]), (fb[2], fb[3]), (200, 200, 0), 1)

        # Draw objects
        for obj in active_objects.values():
            if not obj["confirmed"]:
                continue
            ox1, oy1, ox2, oy2 = obj["bbox"]
            cv2.rectangle(f, (ox1, oy1), (ox2, oy2), (255, 0, 0), 2)
            cv2.putText(f, f"{obj['class_name']} {obj['conf']:.2f}",
                        (ox1, max(20, oy1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        summary = f"OWNER ID {recognized_owner_id}" if recognized_owner_id in active_persons else "NONE"
        cv2.putText(f, f"Owner: {summary}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    _draw_frame(frame)
    cv2.imshow("Threat Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.stop()
cv2.destroyAllWindows()
