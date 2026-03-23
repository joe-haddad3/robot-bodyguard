import cv2
import torch
import math
import time
import subprocess

from camera_utils import LatestFrameCamera
from enhanced_threat_analyzer import EnhancedThreatAnalyzer


MODEL_PATH = "best.pt"

CAM_W = 640
CAM_H = 480
CAM_FPS = 12

YOLO_SIZE = 320
YOLO_EVERY_N_FRAMES = 3

MAIN_LOOP_SLEEP_SEC = 0.01
SHOW_WINDOW = True
PRINT_STATS_EVERY = 30

ENTRY_THRESHOLDS = {
    "person": 0.22,
    "knife": 0.5,
    "gun": 0.55,
    "baseball_bat": 0.6,
    "hammer": 0.35,
}

KEEP_THRESHOLDS = {
    "person": 0.18,
    "knife": 0.4,
    "gun": 0.40,
    "baseball_bat": 0.30,
    "hammer": 0.25,
}

PERSON_MISSING_FRAMES = 12
OBJECT_MISSING_FRAMES = 5
CONF_HISTORY_LEN = 3


model = torch.hub.load(
    ".",
    "custom",
    path=MODEL_PATH,
    source="local",
    force_reload=False,
)
model.conf = 0.15
model.iou = 0.45

CLASS_NAMES = {
    0: "person",
    1: "knife",
    2: "gun",
    3: "baseball_bat",
    4: "hammer",
}
WEAPON_CLASSES = {"knife", "gun", "baseball_bat", "hammer"}


next_person_id = 0
next_object_id = 0

active_persons = {}
active_objects = {}

threat_analyzer = EnhancedThreatAnalyzer()


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


def class_id_to_name(cls_id):
    return CLASS_NAMES.get(cls_id, None)


def should_accept_new(class_name, conf):
    return conf >= ENTRY_THRESHOLDS.get(class_name, 0.50)


def should_keep_existing(class_name, conf):
    return conf >= KEEP_THRESHOLDS.get(class_name, 0.35)


def update_conf_history(track, conf):
    hist = track["conf_history"]
    hist.append(conf)
    if len(hist) > CONF_HISTORY_LEN:
        hist.pop(0)
    track["conf"] = sum(hist) / len(hist)

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
                1,
            )


def match_detection_to_tracks(det_box, tracks, iou_threshold=0.25, dist_threshold=250):
    best_id = None
    best_score = -1

    det_center = bbox_center(det_box)

    for tid, t in tracks.items():
        overlap = iou(det_box, t["bbox"])
        if overlap >= iou_threshold and overlap > best_score:
            best_score = overlap
            best_id = tid

    if best_id is not None:
        return best_id

    best_dist = float("inf")
    for tid, t in tracks.items():
        d = distance(det_center, bbox_center(t["bbox"]))
        if d < dist_threshold and d < best_dist:
            best_dist = d
            best_id = tid

    return best_id


def choose_owner_proxy():
    if not active_persons:
        return None

    best_id = None
    best_score = -1

    for pid, pdata in active_persons.items():
        area = bbox_area(pdata["bbox"])
        conf = pdata["conf"]
        score = area + 5000 * conf
        if score > best_score:
            best_score = score
            best_id = pid

    return best_id


def temperature_string():
    try:
        return subprocess.getoutput("vcgencmd measure_temp").strip()
    except Exception:
        return "temp=unknown"

print("Loading camera...")

camera = LatestFrameCamera(
    src=0,
    width=CAM_W,
    height=CAM_H,
    fps=CAM_FPS,
)

frame_count = 0
yolo_count = 0
last_yolo_ms = 0.0
start_time = time.time()

while True:
    ok, frame, _ = camera.read()

    if not ok or frame is None:
        time.sleep(0.01)
        continue

    frame_count += 1
    detections = []

    if frame_count % YOLO_EVERY_N_FRAMES == 0:
        t0 = time.time()

        infer_frame = cv2.resize(frame, (YOLO_SIZE, YOLO_SIZE))
        results = model(infer_frame)

        t1 = time.time()
        last_yolo_ms = (t1 - t0) * 1000.0
        yolo_count += 1

        det = results.xyxy[0].cpu().numpy()

        sx = frame.shape[1] / YOLO_SIZE
        sy = frame.shape[0] / YOLO_SIZE

        for row in det:
            x1, y1, x2, y2, conf, cls = row
            class_name = class_id_to_name(int(cls))

            if class_name is None:
                continue

            detections.append({
                "bbox": [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)],
                "conf": float(conf),
                "class_name": class_name,
            })

    for p in active_persons.values():
        p["seen"] = False

    for o in active_objects.values():
        o["seen"] = False

    person_dets = [d for d in detections if d["class_name"] == "person"]
    object_dets = [d for d in detections if d["class_name"] in WEAPON_CLASSES]

    # PERSONS
    for det in person_dets:
        box = det["bbox"]
        conf = det["conf"]
        class_name = det["class_name"]

        matched = match_detection_to_tracks(
            box,
            active_persons,
            iou_threshold=0.20,
            dist_threshold=220,
        )
        if matched is None:
            if not should_accept_new(class_name, conf):
                continue

            matched = next_person_id
            next_person_id += 1

            active_persons[matched] = {
                "id": matched,
                "bbox": box,
                "conf": conf,
                "conf_history": [conf],
                "identity": "unknown",
                "weapons": [],
                "missing": 0,
                "seen": True,
            }
        else:
            current_conf = active_persons[matched]["conf"]
            smoothed_candidate = (current_conf + conf) / 2.0

            if not should_keep_existing(class_name, smoothed_candidate):
                continue

            active_persons[matched]["bbox"] = box
            update_conf_history(active_persons[matched], conf)
            active_persons[matched]["seen"] = True
            active_persons[matched]["missing"] = 0

    for pid in list(active_persons.keys()):
        if not active_persons[pid]["seen"]:
            active_persons[pid]["missing"] += 1
            if active_persons[pid]["missing"] > PERSON_MISSING_FRAMES:
                del active_persons[pid]

    # OBJECTS
    for det in object_dets:
        box = det["bbox"]
        conf = det["conf"]
        class_name = det["class_name"]

        matched = match_detection_to_tracks(
            box,
            active_objects,
            iou_threshold=0.20,
            dist_threshold=180,
        )

        if matched is None:
            if not should_accept_new(class_name, conf):
                continue

            matched = next_object_id
            next_object_id += 1

            active_objects[matched] = {
                "id": matched,
                "bbox": box,
                "conf": conf,
                "conf_history": [conf],
                "class_name": class_name,
                "missing": 0,
                "seen": True,
            }
        else:
            current_class = active_objects[matched]["class_name"]
            current_conf = active_objects[matched]["conf"]
            smoothed_candidate = (current_conf + conf) / 2.0

            if not should_keep_existing(current_class, smoothed_candidate):
                continue

            active_objects[matched]["bbox"] = box
            update_conf_history(active_objects[matched], conf)
            active_objects[matched]["seen"] = True
            active_objects[matched]["missing"] = 0

    for oid in list(active_objects.keys()):
        if not active_objects[oid]["seen"]:
            active_objects[oid]["missing"] += 1
            if active_objects[oid]["missing"] > OBJECT_MISSING_FRAMES:
                del active_objects[oid]

    owner_proxy_id = choose_owner_proxy()
    owner_bbox = active_persons[owner_proxy_id]["bbox"] if owner_proxy_id in active_persons else None

    for pid, pdata in active_persons.items():
        pdata["weapons"] = []
        px1, py1, px2, py2 = pdata["bbox"]

        for _, obj in active_objects.items():
            cx, cy = bbox_center(obj["bbox"])
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                pdata["weapons"].append({
                    "class_name": obj["class_name"],
                    "conf": obj["conf"],
                    "bbox": obj["bbox"],
                })

    for pid, pdata in active_persons.items():
        pdata["identity"] = "owner_proxy" if pid == owner_proxy_id else "unknown"

    num_unknown_people = sum(
        1 for _, p in active_persons.items() if p["identity"] == "unknown"
    )
    for pid, pdata in active_persons.items():
        bbox = pdata["bbox"]
        bbox_h = bbox[3] - bbox[1]

        run_pose = (pid != owner_proxy_id) and (bbox_h >= 120) and (frame_count % 6 == 0)

        context = {
            "mode": "indoor",
            "owner_present": owner_proxy_id is not None,
            "time_of_day": "day",
            "owner_bbox": owner_bbox,
            "owner_id": owner_proxy_id,
            "num_unknown_people": num_unknown_people,
            "run_pose": run_pose,
        }

        threat_level, threat_score, explanation = threat_analyzer.analyze_person(
            frame,
            pdata,
            context,
            frame_index=frame_count,
        )

        x1, y1, x2, y2 = bbox

        if pdata["identity"] == "owner_proxy":
            color = (0, 255, 0)
        elif threat_level == "THREAT":
            color = (0, 0, 255)
        elif threat_level == "SUSPICIOUS":
            color = (0, 165, 255)
        else:
            color = (255, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"ID {pid} {pdata['identity']} {threat_level} {threat_score:.1f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

        draw_text_block(frame, explanation[:3], x1, y2 + 18, color)

    for _, obj in active_objects.items():
        x1, y1, x2, y2 = obj["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(
            frame,
            f"{obj['class_name']} {obj['conf']:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
        )

    if frame_count % PRINT_STATS_EVERY == 0:
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0.0

        print(
            f"frames={frame_count} "
            f"yolo_runs={yolo_count} "
            f"loop_fps={fps:.2f} "
            f"last_yolo_ms={last_yolo_ms:.1f} "
            f"{temperature_string()}"
        )

    if SHOW_WINDOW:
        cv2.imshow("Bodyguard Robot Threat Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    time.sleep(MAIN_LOOP_SLEEP_SEC)

camera.release()
cv2.destroyAllWindows()
