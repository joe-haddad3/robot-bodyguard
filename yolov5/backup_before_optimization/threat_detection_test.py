import cv2
import torch
import math
from collections import deque, defaultdict

from enhanced_threat_analyzer import EnhancedThreatAnalyzer


MODEL_PATH = "best.pt"

model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path=MODEL_PATH,
    force_reload=False
)

model.conf = 0.15


CLASS_NAMES = {
    0: "person",
    1: "knife",
    2: "gun",
    3: "baseball_bat",
    4: "hammer",
}

WEAPON_CLASSES = {"knife", "gun", "baseball_bat", "hammer"}

# Your requested thresholds
ENTRY_THRESHOLDS = {
    "person": 0.20,
    "knife": 0.40,
    "gun": 0.65,
    "baseball_bat": 0.45,
    "hammer": 0.4,
}

# Lower keep thresholds so accepted detections stay alive
KEEP_THRESHOLDS = {
    "person": 0.20,
    "knife": 0.35,
    "gun": 0.50,
    "baseball_bat": 0.35,
    "hammer": 0.35,
}

PERSON_MISSING_FRAMES = 40
OBJECT_MISSING_FRAMES = 15

CLASS_CONFIRM_FRAMES = {
    "person": 1,
    "knife": 1,
    "gun": 2,
    "baseball_bat": 1,
    "hammer": 1,
}

# Exception-based switching settings
CLASS_SWITCH_CONFIRM_FRAMES = 2

# Higher priority means allowed to override lower-priority class if seen consistently
CLASS_PRIORITY = {
    "gun": 4,
    "knife": 3,
    "hammer": 2,
    "baseball_bat": 1,
}

FACE_FALLBACK_ENABLED = True
FACE_MIN_SIZE = (60, 60)

next_person_id = 0
next_object_id = 0

active_persons = {}
active_objects = {}
object_class_history = {}
switch_candidate_history = defaultdict(lambda: deque(maxlen=5))

threat_analyzer = EnhancedThreatAnalyzer()
owner_proxy_id = None

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


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


def match_detection_to_tracks(det_box, tracks, iou_threshold=0.25, dist_threshold=250):
    best_id = None
    best_score = -1

    det_center = bbox_center(det_box)

    for tid, t in tracks.items():
        track_box = t["bbox"]
        overlap = iou(det_box, track_box)
        if overlap >= iou_threshold and overlap > best_score:
            best_score = overlap
            best_id = tid

    if best_id is not None:
        return best_id

    best_dist = float("inf")
    for tid, t in tracks.items():
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
    """
    Default behavior:
      keep the first locked class

    Exception:
      if a higher-priority class is seen strongly for enough frames,
      allow switching.
    """
    obj = active_objects[object_id]
    current_class = obj["class_name"]

    if current_class == candidate_class:
        switch_candidate_history[object_id].clear()
        return

    current_priority = CLASS_PRIORITY.get(current_class, 0)
    candidate_priority = CLASS_PRIORITY.get(candidate_class, 0)

    # Only allow override to a stronger/more important class
    if candidate_priority <= current_priority:
        switch_candidate_history[object_id].clear()
        return

    # Candidate must itself pass entry threshold strongly
    if candidate_conf < ENTRY_THRESHOLDS.get(candidate_class, 0.50):
        switch_candidate_history[object_id].clear()
        return

    switch_candidate_history[object_id].append(candidate_class)

    # Require same candidate repeatedly
    recent = list(switch_candidate_history[object_id])[-CLASS_SWITCH_CONFIRM_FRAMES:]
    if len(recent) == CLASS_SWITCH_CONFIRM_FRAMES and all(c == candidate_class for c in recent):
        obj["class_name"] = candidate_class
        switch_candidate_history[object_id].clear()


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

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
                # For existing object, allow keep-threshold continuation
                if not should_keep_existing(class_name, conf):
                    continue

            current_object_detections.append({
                "bbox": box,
                "conf": conf,
                "class_name": class_name
            })

    # STEP 2: FACE FALLBACK
    if FACE_FALLBACK_ENABLED:
        faces = detect_faces(frame)

        for face in faces:
            pseudo_person_box = face_to_person_box(face, frame.shape)

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

    for det in current_person_detections:
        box = det["bbox"]
        conf = det["conf"]
        source = det.get("source", "yolo")

        person_id = match_detection_to_tracks(
            box,
            active_persons,
            iou_threshold=0.20,
            dist_threshold=300
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
                "source": source
            }
        else:
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

        matched_person_ids.add(person_id)

    for pid in list(active_persons.keys()):
        if pid not in matched_person_ids:
            active_persons[pid]["missing"] += 1
            if active_persons[pid]["missing"] > PERSON_MISSING_FRAMES:
                del active_persons[pid]
                if owner_proxy_id == pid:
                    owner_proxy_id = None

    # STEP 4: OWNER PROXY
    if owner_proxy_id is None or owner_proxy_id not in active_persons:
        owner_proxy_id = choose_owner_proxy()

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

            # First accepted class is fixated here
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

            # Keep first class by default, but allow exception-based switch
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

    # STEP 6: ANALYZE ALL PERSONS
    owner_bbox = active_persons[owner_proxy_id]["bbox"] if owner_proxy_id in active_persons else None

    for pid, person_data in active_persons.items():
        if not person_data["confirmed"]:
            continue

        pbox = person_data["bbox"]
        pc = bbox_center(pbox)

        assigned_weapons = []

        for oid, obj in active_objects.items():
            if not obj["confirmed"]:
                continue

            if obj["class_name"] not in WEAPON_CLASSES:
                continue

            obox = obj["bbox"]
            oc = bbox_center(obox)
            d = distance(pc, oc)

            if d < 220:
                assigned_weapons.append({
                    "class_name": obj["class_name"],
                    "conf": obj["conf"],
                    "bbox": obj["bbox"],
                    "distance": d
                })

        assigned_weapons.sort(key=lambda w: w["distance"])

        identity = "owner_proxy" if pid == owner_proxy_id else "unknown"

        person = {
            "id": pid,
            "bbox": pbox,
            "identity": identity,
            "weapons": assigned_weapons
        }

        context = {
            "mode": "indoor",
            "owner_present": True,
            "time_of_day": "day",
            "num_unknown_people": max(0, len(active_persons) - 1),
            "owner_bbox": owner_bbox,
            "owner_id": owner_proxy_id
        }

        threat_level, threat_score, explanation = threat_analyzer.analyze_person(frame, person, context)

        x1, y1, x2, y2 = pbox

        if pid == owner_proxy_id:
            color = (255, 255, 0)
        else:
            color = (0, 255, 0)
            if threat_level == "SUSPICIOUS":
                color = (0, 255, 255)
            elif threat_level == "THREAT":
                color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if pid == owner_proxy_id else 2)

        role = "OWNER_PROXY" if pid == owner_proxy_id else f"ID {pid}"
        source_txt = person_data.get("source", "yolo")
        main_label = f"{role} | {threat_level} {threat_score:.1f}"

        cv2.putText(
            frame,
            main_label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

        info_lines = [
            f"person_conf: {person_data['conf']:.2f}",
            f"source: {source_txt}"
        ]

        if assigned_weapons:
            weapon_text = ", ".join(
                [f"{w['class_name']}({w['conf']:.2f})" for w in assigned_weapons[:3]]
            )
            info_lines.append(f"weapons: {weapon_text}")
        else:
            info_lines.append("weapons: none")

        if explanation:
            info_lines.extend(explanation[:2])

        draw_text_block(frame, info_lines, x1, min(frame.shape[0] - 50, y2 + 20), color)

    # STEP 7: DRAW OBJECTS
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

    cv2.imshow("Threat Detection Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
