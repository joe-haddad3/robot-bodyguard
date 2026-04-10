"""
Bodyguard Robot — Main Threat Detection Loop
=============================================
Runs YOLOv5 + FaceNet (MTCNN) + MediaPipe in a real-time camera loop.
Outputs structured threat state via RobotInterface for robot consumption.

Controls:
    ESC  — quit

First-time setup:
    python enroll_owner.py   (enroll owner face before running this)
"""

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
from robot_interface import RobotInterface


# ===========================================================================
# PLATFORM — set RPI_MODE = True when deploying on Raspberry Pi 5
# ===========================================================================
RPI_MODE = False

# ===========================================================================
# CAMERA + YOLO
# ===========================================================================
CAMERA_INDEX = 0
CAM_W, CAM_H, CAM_FPS = 640, 480, 15
YOLO_EVERY = 2          # run YOLO every N frames

if RPI_MODE:
    CAM_W, CAM_H, CAM_FPS = 320, 240, 10
    YOLO_EVERY = 5

_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_DIR, "best.pt")

# ===========================================================================
# ROBOT OUTPUT
# ===========================================================================
# Change to "serial", "udp", or "file" when running on Raspberry Pi
ROBOT_OUTPUT_MODE  = "print"
ROBOT_SERIAL_PORT  = "/dev/ttyUSB0"   # RPi UART to Arduino/MCU
ROBOT_UDP_HOST     = "192.168.1.100"  # IP of robot motor controller
ROBOT_UDP_PORT     = 5005

# ===========================================================================
# DETECTION CLASSES
# ===========================================================================
CLASS_NAMES = {
    0: "person",
    1: "knife",
    2: "gun",
    3: "baseball_bat",
    4: "hammer",
}
WEAPON_CLASSES = {"knife", "gun", "baseball_bat", "hammer"}

# ===========================================================================
# DETECTION THRESHOLDS
# Raise ENTRY to reduce false detections; lower KEEP to avoid flickering
# ===========================================================================
ENTRY_THRESHOLDS = {
    "person":       0.50,
    "knife":        0.50,
    "gun":          0.65,
    "baseball_bat": 0.45,
    "hammer":       0.40,
}
KEEP_THRESHOLDS = {
    "person":       0.20,
    "knife":        0.35,   # keep tracking after first detection
    "gun":          0.40,
    "baseball_bat": 0.30,
    "hammer":       0.30,
}
CLASS_CONFIRM_FRAMES = {
    "person":       1,
    "knife":        2,      # needs 2 detections before adding to track
    "gun":          2,
    "baseball_bat": 2,
    "hammer":       2,
}
CLASS_SWITCH_CONFIRM_FRAMES = 2
CLASS_PRIORITY = {"gun": 4, "knife": 3, "hammer": 2, "baseball_bat": 1}

PERSON_MISSING_FRAMES = 40
OBJECT_MISSING_FRAMES = 15

# ===========================================================================
# FACE FALLBACK  (Haar cascade — used only to create person tracks)
# ===========================================================================
FACE_FALLBACK_ENABLED = True
FACE_MIN_SIZE = (30, 30) if RPI_MODE else (60, 60)

# ===========================================================================
# OWNER RECOGNITION  (MTCNN + FaceNet)
# ===========================================================================
OWNER_RECOGNITION_ENABLED  = True
OWNER_RECOGNITION_EVERY    = 15 if RPI_MODE else 2
OWNER_DISTANCE_THRESHOLD   = 0.30
OWNER_CONFIRM_FRAMES       = 3
MIN_OWNER_FACE_SIZE        = 30 if RPI_MODE else 60

# ===========================================================================
# DISPLAY COLOURS  (BGR)
# ===========================================================================
C_OWNER  = (0, 255, 255)   # cyan
C_SAFE   = (0, 200, 0)     # green
C_SUSP   = (0, 165, 255)   # orange
C_THREAT = (0, 0, 255)     # red
C_WEAPON = (255, 80, 0)    # blue
C_WHITE  = (255, 255, 255)
C_YELLOW = (0, 255, 255)

LEVEL_COLOR = {"SAFE": C_SAFE, "SUSPICIOUS": C_SUSP, "THREAT": C_THREAT}
LEVEL_RANK  = {"SAFE": 0, "SUSPICIOUS": 1, "THREAT": 2}


# ===========================================================================
# INITIALISATION
# ===========================================================================
print("[INIT] Opening camera...")
cam = LatestFrameCamera(
    src=CAMERA_INDEX, width=CAM_W, height=CAM_H, fps=CAM_FPS
).start()

print("[INIT] Loading YOLO model...")
model = torch.hub.load(
    _DIR, "custom", path=MODEL_PATH, source="local", force_reload=False
)
model.conf = 0.15

print("[INIT] Loading threat analyzer...")
threat_analyzer = EnhancedThreatAnalyzer(rpi_mode=RPI_MODE)

print("[INIT] Loading owner recognizer (MTCNN + FaceNet)...")
owner_recognizer = FaceNetOwnerRecognizer(threshold=OWNER_DISTANCE_THRESHOLD)

print("[INIT] Starting robot interface...")
robot_iface = RobotInterface(
    mode=ROBOT_OUTPUT_MODE,
    serial_port=ROBOT_SERIAL_PORT,
    udp_host=ROBOT_UDP_HOST,
    udp_port=ROBOT_UDP_PORT,
)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("[INIT] All systems ready.  Press ESC to quit.\n")


# ===========================================================================
# TRACKING STATE
# ===========================================================================
next_person_id = 0
next_object_id = 0
active_persons = {}
active_objects = {}
object_class_history   = {}
switch_candidate_history = defaultdict(lambda: deque(maxlen=5))

recognized_owner_id = None
frame_count  = 0
detections   = []

fps_history  = deque(maxlen=30)
_last_t      = time.time()


# ===========================================================================
# GEOMETRY HELPERS
# ===========================================================================
def bbox_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def bbox_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

def bbox_width(box):
    return max(0, box[2] - box[0])

def bbox_height(box):
    return max(0, box[3] - box[1])

def dist2d(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def iou(A, B):
    xA = max(A[0], B[0]); yA = max(A[1], B[1])
    xB = min(A[2], B[2]); yB = min(A[3], B[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    denom = bbox_area(A) + bbox_area(B) - inter
    return inter / denom if denom > 0 else 0.0

def point_in_box(pt, box):
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


# ===========================================================================
# TRACKING HELPERS
# ===========================================================================
def match_to_tracks(det_box, tracks, iou_thr=0.25, dist_thr=250, skip_ids=None):
    if skip_ids is None:
        skip_ids = set()
    best_id, best_score = None, -1
    dc = bbox_center(det_box)

    for tid, t in tracks.items():
        if tid in skip_ids:
            continue
        ov = iou(det_box, t["bbox"])
        if ov >= iou_thr and ov > best_score:
            best_score, best_id = ov, tid

    if best_id is not None:
        return best_id

    best_dist = float("inf")
    for tid, t in tracks.items():
        if tid in skip_ids:
            continue
        d = dist2d(dc, bbox_center(t["bbox"]))
        if d < dist_thr and d < best_dist:
            best_dist, best_id = d, tid

    return best_id


def update_person_track(pid, box, conf, source):
    p = active_persons[pid]
    p["bbox"]       = [int(0.7 * p["bbox"][i] + 0.3 * box[i]) for i in range(4)]
    p["conf"]       = 0.8 * p["conf"] + 0.2 * conf
    p["missing"]    = 0
    p["seen_count"] += 1
    p["confirmed"]  = True
    p["source"]     = source


def new_person_record(box, conf, source):
    return {
        "bbox": box, "conf": conf, "missing": 0,
        "identity": "unknown", "seen_count": 1, "confirmed": True,
        "source": source,
        "owner_confirmed": False, "owner_distance": 999.0,
        "owner_label": "UNKNOWN", "face_box_global": None,
        "owner_match_streak": 0,
        "threat_level": "SAFE", "threat_score": 0.0,
    }


def maybe_switch_class(oid, candidate_class, candidate_conf):
    obj = active_objects[oid]
    current = obj["class_name"]
    if current == candidate_class:
        switch_candidate_history[oid].clear()
        return
    if CLASS_PRIORITY.get(candidate_class, 0) <= CLASS_PRIORITY.get(current, 0):
        switch_candidate_history[oid].clear()
        return
    if candidate_conf < ENTRY_THRESHOLDS.get(candidate_class, 0.50):
        switch_candidate_history[oid].clear()
        return
    switch_candidate_history[oid].append(candidate_class)
    recent = list(switch_candidate_history[oid])[-CLASS_SWITCH_CONFIRM_FRAMES:]
    if len(recent) == CLASS_SWITCH_CONFIRM_FRAMES and all(c == candidate_class for c in recent):
        obj["class_name"] = candidate_class
        switch_candidate_history[oid].clear()


# ===========================================================================
# FACE DETECTION (Haar — for person fallback only, NOT recognition)
# ===========================================================================
def detect_faces_haar(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=FACE_MIN_SIZE
    )


def face_to_person_box(face, frame_shape):
    x, y, w, h = face
    H, W = frame_shape[:2]
    return [
        max(0, int(x - 0.8 * w)),
        max(0, int(y - 0.6 * h)),
        min(W - 1, int(x + w + 0.8 * w)),
        min(H - 1, int(y + h + 2.8 * h)),
    ]


# ===========================================================================
# WEAPON ASSIGNMENT
# ===========================================================================
def make_hand_zone(pbox):
    w = bbox_width(pbox)
    h = bbox_height(pbox)
    return [
        int(pbox[0] - 0.25 * w), int(pbox[1] + 0.18 * h),
        int(pbox[2] + 0.25 * w), int(pbox[1] + 0.78 * h),
    ]


def weapon_match_score(weapon_box, person_box):
    wc = bbox_center(weapon_box)
    score  = max(0.0, 2.0 - dist2d(wc, bbox_center(person_box)) / max(1, bbox_height(person_box)))
    score += 3.0 * iou(weapon_box, person_box)
    if point_in_box(wc, person_box):
        score += 1.5
    if point_in_box(wc, make_hand_zone(person_box)):
        score += 2.0
    return score


def assign_weapons(persons, objects):
    assignments = {pid: [] for pid in persons}
    conf_ppl = {pid: p for pid, p in persons.items() if p.get("confirmed")}
    conf_wpn = {
        oid: o for oid, o in objects.items()
        if o.get("confirmed") and o.get("class_name") in WEAPON_CLASSES
    }
    for oid, obj in conf_wpn.items():
        best_pid, best_s, second_s = None, -1e9, -1e9
        for pid, pdata in conf_ppl.items():
            s = weapon_match_score(obj["bbox"], pdata["bbox"])
            if s > best_s:
                second_s, best_s, best_pid = best_s, s, pid
            elif s > second_s:
                second_s = s
        if best_pid is None:
            continue
        margin = best_s - second_s
        if best_s < 1.2 or margin < 0.35:
            continue
        if conf_ppl[best_pid].get("identity") == "owner" and margin < 0.60:
            continue
        assignments[best_pid].append({
            "class_name": obj["class_name"],
            "conf": obj["conf"],
            "bbox": obj["bbox"],
            "object_id": oid,
        })
    return assignments


# ===========================================================================
# OWNER RECOGNITION  (MTCNN + FaceNet — called only when owner not locked)
# ===========================================================================
def match_faces_to_persons(faces, persons):
    assignments, used = {}, set()
    for face in faces:
        fc = face["center"]
        best_pid, best_s = None, -1e9
        for pid, pdata in persons.items():
            if not pdata.get("confirmed") or pid in used:
                continue
            pbox = pdata["bbox"]
            ph   = max(1.0, bbox_height(pbox))
            s    = 0.0
            if point_in_box(fc, pbox):
                s += 3.0
            s += max(0.0, 2.0 - dist2d(fc, bbox_center(pbox)) / ph)
            s += 2.0 * iou(list(face["face_box"]), pbox)
            upper = [pbox[0], pbox[1], pbox[2], pbox[1] + int(0.55 * bbox_height(pbox))]
            if point_in_box(fc, upper):
                s += 1.0
            if s > best_s:
                best_s, best_pid = s, pid
        if best_pid is not None and best_s > 1.0:
            assignments[best_pid] = face
            used.add(best_pid)
    return assignments


def run_owner_recognition(frame):
    global recognized_owner_id

    for pdata in active_persons.values():
        pdata["face_box_global"] = None
        pdata["owner_distance"]  = 999.0
        pdata["owner_label"]     = "UNKNOWN"

    detected_faces   = owner_recognizer.detect_faces(frame)
    face_assignments = match_faces_to_persons(detected_faces, active_persons)

    best_pid, best_dist = None, 999.0

    for pid, pdata in active_persons.items():
        if not pdata["confirmed"]:
            continue
        face_info = face_assignments.get(pid)
        if face_info is None:
            pdata["owner_match_streak"] = 0
            continue

        pdata["face_box_global"] = face_info["face_box"]
        result = owner_recognizer.recognize_tensor(face_info["face_tensor"])
        dist   = float(result.get("distance", 999.0))
        is_ow  = dist < OWNER_DISTANCE_THRESHOLD

        pdata["owner_distance"]    = dist
        pdata["owner_label"]       = "OWNER" if is_ow else "UNKNOWN"
        pdata["owner_match_streak"] = (
            pdata.get("owner_match_streak", 0) + 1 if is_ow else 0
        )

        if pdata["owner_match_streak"] >= OWNER_CONFIRM_FRAMES and dist < best_dist:
            best_dist, best_pid = dist, pid

    if best_pid is not None:
        recognized_owner_id = best_pid
        for pid in active_persons:
            active_persons[pid]["owner_confirmed"] = (pid == best_pid)
            active_persons[pid]["identity"]        = "owner" if pid == best_pid else "unknown"


# ===========================================================================
# DISPLAY HELPERS
# ===========================================================================
def put(frame, text, x, y, color, scale=0.5, thick=1):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def draw_status_bar(frame, fps, owner_status, system_level, n_threats):
    W = frame.shape[1]
    bar_colors = {"SAFE": (0, 90, 0), "SUSPICIOUS": (0, 110, 180), "THREAT": (0, 0, 160)}
    cv2.rectangle(frame, (0, 0), (W, 32), bar_colors.get(system_level, (0, 90, 0)), -1)
    put(frame, f"FPS:{fps:.0f}", 6, 22, C_WHITE, 0.55, 1)
    put(frame, f"| {system_level}",  80, 22, C_WHITE, 0.6, 2 if system_level != "SAFE" else 1)
    put(frame, f"OWNER:{owner_status}", W // 2 - 70, 22, C_YELLOW, 0.55, 1)
    put(frame, f"THREATS:{n_threats}", W - 120, 22, C_WHITE, 0.55, 1)


def draw_person(frame, pid, pdata, is_owner, weapons):
    pbox = pdata["bbox"]
    x1, y1, x2, y2 = pbox
    lvl   = pdata.get("threat_level", "SAFE")
    score = pdata.get("threat_score", 0.0)

    if is_owner:
        color = C_OWNER
        label = "OWNER"
        thick = 3
    else:
        color = LEVEL_COLOR.get(lvl, C_SAFE)
        label = f"ID{pid} | {lvl} ({score:.0f})"
        thick = 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
    put(frame, label, x1, max(22, y1 - 8), color, 0.6, 2)

    lines = []
    if not is_owner:
        lines.append(f"conf:{pdata['conf']:.2f} src:{pdata.get('source','?')}")
        d = pdata.get("owner_distance", 999)
        if d < 999:
            lines.append(f"face_dist:{d:.3f}  streak:{pdata.get('owner_match_streak',0)}")
        if weapons:
            wlist = " + ".join(f"{w['class_name']}({w['conf']:.2f})" for w in weapons[:2])
            lines.append(f"WEAPON: {wlist}")
    else:
        lines.append(f"conf:{pdata['conf']:.2f}")

    for i, ln in enumerate(lines):
        yy = min(frame.shape[0] - 4, y2 + 16 + i * 16)
        put(frame, ln, x1, yy, color, 0.42, 1)

    fb = pdata.get("face_box_global")
    if fb is not None:
        cv2.rectangle(frame, (fb[0], fb[1]), (fb[2], fb[3]), (200, 200, 0), 1)


# ===========================================================================
# MAIN LOOP
# ===========================================================================
try:
    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.005)
            continue

        # --- FPS ---
        now = time.time()
        fps_history.append(1.0 / max(0.001, now - _last_t))
        _last_t = now
        fps = sum(fps_history) / len(fps_history)

        frame_count += 1

        # ==============================================================
        # STEP 1 — YOLO DETECTION
        # ==============================================================
        if frame_count % YOLO_EVERY == 0:
            results    = model(frame)
            detections = results.xyxy[0].cpu().numpy()

        current_persons, current_objects = [], []

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            cls, conf = int(cls), float(conf)
            cname = CLASS_NAMES.get(cls)
            if cname is None:
                continue
            box = [int(x1), int(y1), int(x2), int(y2)]

            if cname == "person":
                mid = match_to_tracks(box, active_persons, 0.20, 300)
                if mid is None and not (conf >= ENTRY_THRESHOLDS["person"]):
                    continue
                if mid is not None and not (conf >= KEEP_THRESHOLDS["person"]):
                    continue
                current_persons.append({"bbox": box, "conf": conf, "source": "yolo"})
            else:
                mid = match_to_tracks(box, active_objects, 0.20, 180)
                if mid is None and not (conf >= ENTRY_THRESHOLDS.get(cname, 0.50)):
                    continue
                if mid is not None and not (conf >= KEEP_THRESHOLDS.get(cname, 0.30)):
                    continue
                current_objects.append({"bbox": box, "conf": conf, "class_name": cname})

        # ==============================================================
        # STEP 2 — FACE FALLBACK (person detection only)
        # ==============================================================
        if FACE_FALLBACK_ENABLED:
            for face in detect_faces_haar(frame):
                pb = face_to_person_box(face, frame.shape)
                if any(iou(pb, d["bbox"]) > 0.25 for d in current_persons):
                    continue
                current_persons.append({"bbox": pb, "conf": 0.99, "source": "face"})

        # ==============================================================
        # STEP 3 — UPDATE PERSON TRACKS
        # ==============================================================
        matched_pids, used_idxs = set(), set()

        # 3A — lock on confirmed owner first
        if recognized_owner_id is not None and recognized_owner_id in active_persons:
            owner_box = active_persons[recognized_owner_id]["bbox"]
            best_idx, best_iou_v, best_dist_v = None, -1.0, float("inf")
            for i, det in enumerate(current_persons):
                ov = iou(det["bbox"], owner_box)
                d  = dist2d(bbox_center(det["bbox"]), bbox_center(owner_box))
                if ov > best_iou_v:
                    best_iou_v, best_dist_v, best_idx = ov, d, i
                elif abs(ov - best_iou_v) < 1e-6 and d < best_dist_v:
                    best_dist_v, best_idx = d, i

            owner_match_ok = False
            if best_idx is not None:
                owner_match_ok = (best_iou_v >= 0.10) or (best_dist_v < 120)
                if owner_match_ok:
                    det = current_persons[best_idx]
                    update_person_track(recognized_owner_id, det["bbox"], det["conf"], det["source"])
                    matched_pids.add(recognized_owner_id)
                    used_idxs.add(best_idx)

            if not owner_match_ok:
                p = active_persons[recognized_owner_id]
                p["owner_confirmed"] = False
                p["identity"]        = "unknown"
                p["owner_match_streak"] = 0
                p["owner_distance"]  = 999.0
                p["owner_label"]     = "UNKNOWN"
                p["face_box_global"] = None
                recognized_owner_id  = None

        # 3B — match remaining detections
        for i, det in enumerate(current_persons):
            if i in used_idxs:
                continue
            box, conf, src = det["bbox"], det["conf"], det["source"]
            skip = {recognized_owner_id} if recognized_owner_id is not None else set()
            pid  = match_to_tracks(box, active_persons, 0.20, 300, skip_ids=skip)

            if pid is None:
                pid = next_person_id
                next_person_id += 1
                active_persons[pid] = new_person_record(box, conf, src)
            else:
                update_person_track(pid, box, conf, src)

            matched_pids.add(pid)

        # Age out missing persons
        for pid in list(active_persons.keys()):
            if pid not in matched_pids:
                active_persons[pid]["missing"] += 1
                if active_persons[pid]["missing"] > PERSON_MISSING_FRAMES:
                    del active_persons[pid]
                    if recognized_owner_id == pid:
                        recognized_owner_id = None

        if recognized_owner_id not in active_persons:
            recognized_owner_id = None

        # ==============================================================
        # STEP 4 — OWNER RECOGNITION
        # ==============================================================
        if (OWNER_RECOGNITION_ENABLED
                and recognized_owner_id is None
                and frame_count % OWNER_RECOGNITION_EVERY == 0):
            run_owner_recognition(frame)

        effective_owner_id = recognized_owner_id if recognized_owner_id in active_persons else None
        for pid in active_persons:
            is_ow = (pid == effective_owner_id)
            active_persons[pid]["identity"]       = "owner" if is_ow else "unknown"
            active_persons[pid]["owner_confirmed"] = is_ow

        # ==============================================================
        # STEP 5 — UPDATE OBJECT TRACKS
        # ==============================================================
        matched_oids = set()
        for det in current_objects:
            box, conf, cname = det["bbox"], det["conf"], det["class_name"]
            oid = match_to_tracks(box, active_objects, 0.20, 180)

            if oid is None:
                oid = next_object_id
                next_object_id += 1
                active_objects[oid] = {
                    "bbox": box, "conf": conf, "class_name": cname,
                    "missing": 0, "seen_count": 1, "confirmed": False,
                }
                object_class_history[oid] = deque(maxlen=5)
            else:
                o = active_objects[oid]
                o["conf"]       = 0.7 * o["conf"] + 0.3 * conf
                o["bbox"]       = [int(0.7 * o["bbox"][i] + 0.3 * box[i]) for i in range(4)]
                o["missing"]    = 0
                o["seen_count"] += 1
                maybe_switch_class(oid, cname, conf)

            object_class_history[oid].append(active_objects[oid]["class_name"])
            if active_objects[oid]["seen_count"] >= CLASS_CONFIRM_FRAMES.get(active_objects[oid]["class_name"], 1):
                active_objects[oid]["confirmed"] = True

            matched_oids.add(oid)

        for oid in list(active_objects.keys()):
            if oid not in matched_oids:
                active_objects[oid]["missing"] += 1
                switch_candidate_history.get(oid, deque()).clear()
                if active_objects[oid]["missing"] > OBJECT_MISSING_FRAMES:
                    del active_objects[oid]
                    object_class_history.pop(oid, None)
                    switch_candidate_history.pop(oid, None)

        threat_analyzer.cleanup_missing_tracks(active_persons.keys())

        # ==============================================================
        # STEP 6 — WEAPON ASSIGNMENT
        # ==============================================================
        weapon_assignments = assign_weapons(active_persons, active_objects)

        # ==============================================================
        # STEP 7 — THREAT ANALYSIS + DRAW PERSONS
        # ==============================================================
        owner_bbox     = active_persons[effective_owner_id]["bbox"] if effective_owner_id in active_persons else None
        system_level   = "SAFE"
        top_threat     = None

        for pid, pdata in active_persons.items():
            if not pdata["confirmed"]:
                continue

            pbox     = pdata["bbox"]
            is_owner = (pid == effective_owner_id)
            weapons  = weapon_assignments.get(pid, [])

            threat_level, threat_score, _, _ = threat_analyzer.analyze_person(
                frame,
                {"id": pid, "bbox": pbox, "identity": "owner" if is_owner else "unknown", "weapons": weapons},
                {"mode": "indoor", "owner_present": effective_owner_id in active_persons,
                 "owner_bbox": owner_bbox, "owner_id": effective_owner_id},
                frame_index=frame_count,
            )

            pdata["threat_level"] = threat_level
            pdata["threat_score"] = threat_score

            if not is_owner:
                rank = LEVEL_RANK.get(threat_level, 0)
                if rank > LEVEL_RANK.get(system_level, 0):
                    system_level = threat_level
                top_rank = LEVEL_RANK.get(top_threat["level"], 0) if top_threat else -1
                if rank > top_rank or (rank == top_rank and threat_score > (top_threat["score"] if top_threat else 0)):
                    cx, cy = bbox_center(pbox)
                    top_threat = {
                        "id": pid, "level": threat_level, "score": round(threat_score, 1),
                        "cx": int(cx), "cy": int(cy), "bbox": list(pbox),
                        "weapons": [w["class_name"] for w in weapons],
                    }

            draw_person(frame, pid, pdata, is_owner, weapons)

        # ==============================================================
        # STEP 8 — DRAW WEAPONS
        # ==============================================================
        for obj in active_objects.values():
            if not obj["confirmed"]:
                continue
            x1, y1, x2, y2 = obj["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), C_WEAPON, 2)
            put(frame, f"{obj['class_name']} {obj['conf']:.2f}",
                x1, max(20, y1 - 6), C_WEAPON, 0.5, 2)

        # ==============================================================
        # STEP 9 — STATUS BAR + ROBOT OUTPUT
        # ==============================================================
        n_threats  = sum(
            1 for p in active_persons.values()
            if p.get("threat_level") in ("SUSPICIOUS", "THREAT") and p.get("identity") != "owner"
        )
        owner_str  = f"ID{effective_owner_id}" if effective_owner_id is not None else "NOT FOUND"
        draw_status_bar(frame, fps, owner_str, system_level, n_threats)

        # Build owner output
        owner_out = {"detected": False}
        if effective_owner_id is not None and effective_owner_id in active_persons:
            ob = active_persons[effective_owner_id]["bbox"]
            cx, cy = bbox_center(ob)
            owner_out = {"detected": True, "cx": int(cx), "cy": int(cy), "bbox": list(ob)}

        robot_iface.update({
            "ts":         round(now, 3),
            "system":     system_level,
            "owner":      owner_out,
            "top_threat": top_threat,
            "frame_w":    CAM_W,
            "frame_h":    CAM_H,
        })

        cv2.imshow("Bodyguard Robot", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    print("[EXIT] Shutting down...")
    cam.stop()
    robot_iface.close()
    cv2.destroyAllWindows()
    print("[EXIT] Done.")
