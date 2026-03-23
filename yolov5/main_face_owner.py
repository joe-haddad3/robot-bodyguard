import cv2
import torch
import math
import time
from collections import deque, defaultdict

from enhanced_threat_analyzer import EnhancedThreatAnalyzer
from camera_utils import LatestFrameCamera
from face_owner_recognizer import OwnerRecognizer


# -----------------------------
# CAMERA / MODEL SETTINGS
# -----------------------------
CAMERA_INDEX = 0
CAM_W = 640
CAM_H = 480
CAM_FPS = 15
YOLO_EVERY = 2

MODEL_PATH = "best.pt"

cam = LatestFrameCamera(
    src=CAMERA_INDEX,
    width=CAM_W,
    height=CAM_H,
    fps=CAM_FPS
).start()

model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path=MODEL_PATH,
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
# FAST OWNER SETTINGS
# -----------------------------
OWNER_FACE_CHECK_EVERY = 1
OWNER_FACE_MIN_PERSON_AREA = 9000
OWNER_LOCK_MISSING_FRAMES = 45
ALLOW_OWNER_FALLBACK_BY_SIZE = False

FACE_ID_HISTORY_LEN = 4
FACE_OWNER_CONFIRM_VOTES = 1
FACE_UNKNOWN_CONFIRM_VOTES = 4


# -----------------------------
# TRACKING STATE
# -----------------------------
next_person_id = 0
next_object_id = 0

active_persons = {}
active_objects = {}
switch_candidate_history = defaultdict(lambda: deque(maxlen=5))

threat_analyzer = EnhancedThreatAnalyzer()
owner_proxy_id = None

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

owner_recognizer = OwnerRecognizer()

frame_count = 0
detections = []


# -----------------------------
# UTILS
# -----------------------------
def bbox_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def bbox_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    return inter / (bbox_area(boxA) + bbox_area(boxB) - inter)


def match_detection_to_tracks(det_box, tracks):
    best_id = None
    best_score = 0

    for tid, t in tracks.items():
        overlap = iou(det_box, t["bbox"])
        if overlap > 0.2 and overlap > best_score:
            best_score = overlap
            best_id = tid

    return best_id


# -----------------------------
# FACE LOGIC (LOCKED OWNER)
# -----------------------------
def maybe_update_person_identity_from_face(frame, pid, pdata):
    global owner_proxy_id

    if not pdata["confirmed"]:
        return

    # Init safety
    if "face_id_history" not in pdata:
        pdata["face_id_history"] = deque(maxlen=FACE_ID_HISTORY_LEN)
    if "owner_locked" not in pdata:
        pdata["owner_locked"] = False

    # 🔒 HARD LOCK: never change once owner
    if pdata["owner_locked"]:
        pdata["identity"] = "owner"
        owner_proxy_id = pid
        return

    # If owner already exists → everyone else unknown
    if owner_proxy_id is not None and pid != owner_proxy_id:
        pdata["identity"] = "unknown"
        return

    # Frequency control
    if frame_count - pdata.get("last_face_check_frame", 0) < OWNER_FACE_CHECK_EVERY:
        return

    if bbox_area(pdata["bbox"]) < OWNER_FACE_MIN_PERSON_AREA:
        return

    result = owner_recognizer.recognize_person(frame, pdata["bbox"])

    pdata["last_face_check_frame"] = frame_count
    pdata["face_status"] = result.get("status", "unverified")

    if result["status"] == "authorized":
        # 🔥 INSTANT OWNER LOCK
        pdata["identity"] = "owner"
        pdata["identity_name"] = result.get("name", "owner")
        pdata["owner_locked"] = True
        owner_proxy_id = pid

        # Force everyone else unknown
        for other_id, other in active_persons.items():
            if other_id != pid:
                other["identity"] = "unknown"
                other["owner_locked"] = False

        return

    # Unknown (only if no owner yet)
    if owner_proxy_id is None:
        pdata["identity"] = "unknown"


# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    frame = cam.read()
    if frame is None:
        continue

    frame_count += 1

    if frame_count % YOLO_EVERY == 0:
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()

    current_persons = []

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) != 0:
            continue

        box = [int(x1), int(y1), int(x2), int(y2)]
        current_persons.append({"bbox": box, "conf": float(conf)})

    matched_ids = set()

    # Update persons
    for det in current_persons:
        pid = match_detection_to_tracks(det["bbox"], active_persons)

        if pid is None:
            pid = next_person_id
            next_person_id += 1

            active_persons[pid] = {
                "bbox": det["bbox"],
                "conf": det["conf"],
                "missing": 0,
                "identity": "unknown",
                "confirmed": True,
                "last_face_check_frame": 0,
                "face_id_history": deque(maxlen=FACE_ID_HISTORY_LEN),
                "owner_locked": False,
            }
        else:
            active_persons[pid]["bbox"] = det["bbox"]
            active_persons[pid]["missing"] = 0

        matched_ids.add(pid)

    # Remove missing
    for pid in list(active_persons.keys()):
        if pid not in matched_ids:
            active_persons[pid]["missing"] += 1
            if active_persons[pid]["missing"] > 40:
                del active_persons[pid]
                if owner_proxy_id == pid:
                    owner_proxy_id = None

    # Face recognition
    for pid, pdata in active_persons.items():
        maybe_update_person_identity_from_face(frame, pid, pdata)

    # Draw
    for pid, pdata in active_persons.items():
        x1, y1, x2, y2 = pdata["bbox"]

        if pdata["identity"] == "owner":
            color = (0, 255, 255)
            label = "OWNER"
        else:
            color = (0, 255, 0)
            label = f"ID {pid}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Owner Lock System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.stop()
cv2.destroyAllWindows()
