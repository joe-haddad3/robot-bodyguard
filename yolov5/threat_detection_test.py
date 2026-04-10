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
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
RPI_MODE      = False
CAMERA_INDEX  = 0
CAM_W, CAM_H  = 640, 480
CAM_FPS       = 15
YOLO_EVERY    = 2       # run YOLO every N frames

if RPI_MODE:
    CAM_W, CAM_H = 320, 240
    CAM_FPS      = 10
    YOLO_EVERY   = 5

# Motion gate — skip detection entirely when scene barely changed
MOTION_THRESHOLD  = 0.10    # fraction of pixels that must change  (10 %)
MOTION_PIXEL_DIFF = 15      # grey-level change counted as "moved"

# Owner recognition
OWNER_RECOGNITION_EVERY   = 2      # check every N frames (0 = every frame)
OWNER_DISTANCE_THRESHOLD  = 0.45   # cosine distance; strangers ≈ 0.55+
OWNER_CONFIRM_FRAMES      = 2      # consecutive hits to lock owner
OWNER_FACE_REGION_FRAC    = 0.45   # top fraction of YOLO box = face region
MIN_OWNER_FACE_SIZE       = 60     # min crop width (px) to attempt recognition

if RPI_MODE:
    OWNER_RECOGNITION_EVERY = 15
    MIN_OWNER_FACE_SIZE     = 30

# Detection thresholds
ENTRY_THRESHOLDS = {
    "person": 0.50, "knife": 0.70, "gun": 0.65,
    "baseball_bat": 0.90, "hammer": 0.40,
}
KEEP_THRESHOLDS = {
    "person": 0.20, "knife": 0.35, "gun": 0.70,
    "baseball_bat": 0.90, "hammer": 0.30,
}
PERSON_MISSING_FRAMES = 40
OBJECT_MISSING_FRAMES = 15
CLASS_CONFIRM_FRAMES  = {
    "person": 1, "knife": 2, "gun": 2, "baseball_bat": 2, "hammer": 2,
}
CLASS_SWITCH_CONFIRM_FRAMES = 2
CLASS_PRIORITY = {"gun": 4, "knife": 3, "hammer": 2, "baseball_bat": 1}
CLASS_NAMES    = {0: "person", 1: "knife", 2: "gun", 3: "baseball_bat", 4: "hammer"}
WEAPON_CLASSES = {"knife", "gun", "baseball_bat", "hammer"}

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")

# ─────────────────────────────────────────────────────────────────────────────
# INITIALISE MODELS
# ─────────────────────────────────────────────────────────────────────────────
cam = LatestFrameCamera(src=CAMERA_INDEX, width=CAM_W, height=CAM_H, fps=CAM_FPS).start()

model = torch.hub.load(
    os.path.dirname(os.path.abspath(__file__)),
    "custom", path=MODEL_PATH, source="local", force_reload=False,
)
model.conf = 0.15

threat_analyzer  = EnhancedThreatAnalyzer(rpi_mode=RPI_MODE)
owner_recognizer = FaceNetOwnerRecognizer(threshold=OWNER_DISTANCE_THRESHOLD)

# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────
next_person_id           = 0
next_object_id           = 0
active_persons           = {}   # pid  → person record
active_objects           = {}   # oid  → object record
object_class_history     = {}
switch_candidate_history = defaultdict(lambda: deque(maxlen=5))
recognized_owner_id      = None

frame_count      = 0
detections       = []           # last YOLO output
prev_gray_small  = None         # motion gate reference frame
changed_ratio    = 1.0          # last computed motion ratio


# =============================================================================
# HELPERS
# =============================================================================
def bbox_center(b):   return ((b[0]+b[2])/2, (b[1]+b[3])/2)
def bbox_height(b):   return max(0, b[3]-b[1])
def bbox_width(b):    return max(0, b[2]-b[0])
def bbox_area(b):     return max(0, b[2]-b[0]) * max(0, b[3]-b[1])

def dist2d(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def iou(A, B):
    ix1=max(A[0],B[0]); iy1=max(A[1],B[1])
    ix2=min(A[2],B[2]); iy2=min(A[3],B[3])
    iw=max(0,ix2-ix1); ih=max(0,iy2-iy1)
    inter=iw*ih
    if inter==0: return 0.0
    denom=bbox_area(A)+bbox_area(B)-inter
    return inter/denom if denom>0 else 0.0

def point_in_box(pt, b):
    return b[0]<=pt[0]<=b[2] and b[1]<=pt[1]<=b[3]

def draw_text_block(frame, lines, x, y, color):
    for i, line in enumerate(lines):
        yy = y + i*18
        if 0 < yy < frame.shape[0]-5:
            cv2.putText(frame, line, (x, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def match_to_tracks(box, tracks, iou_thr=0.25, dist_thr=250, skip=None):
    skip = skip or set()
    best_id=None; best_iou=-1
    for tid,t in tracks.items():
        if tid in skip: continue
        ov=iou(box,t["bbox"])
        if ov>=iou_thr and ov>best_iou: best_iou=ov; best_id=tid
    if best_id: return best_id
    best_d=float("inf")
    for tid,t in tracks.items():
        if tid in skip: continue
        d=dist2d(bbox_center(box),bbox_center(t["bbox"]))
        if d<dist_thr and d<best_d: best_d=d; best_id=tid
    return best_id

def smooth_box(old, new, a=0.7):
    return [int(a*old[i]+(1-a)*new[i]) for i in range(4)]

def update_person_track(pid, box, conf, source):
    p=active_persons[pid]
    p["bbox"]      = smooth_box(p["bbox"], box)
    p["conf"]      = 0.8*p["conf"] + 0.2*conf
    p["missing"]   = 0
    p["seen_count"]+= 1
    p["confirmed"] = True
    p["source"]    = source

def maybe_switch_class(oid, cname, conf):
    obj=active_objects[oid]; cur=obj["class_name"]
    if cur==cname: switch_candidate_history[oid].clear(); return
    if CLASS_PRIORITY.get(cname,0)<=CLASS_PRIORITY.get(cur,0):
        switch_candidate_history[oid].clear(); return
    if conf<ENTRY_THRESHOLDS.get(cname,0.5):
        switch_candidate_history[oid].clear(); return
    switch_candidate_history[oid].append(cname)
    recent=list(switch_candidate_history[oid])[-CLASS_SWITCH_CONFIRM_FRAMES:]
    if len(recent)==CLASS_SWITCH_CONFIRM_FRAMES and all(c==cname for c in recent):
        obj["class_name"]=cname; switch_candidate_history[oid].clear()

def make_hand_zone(b):
    w=bbox_width(b); h=bbox_height(b)
    return [int(b[0]-0.25*w), int(b[1]+0.18*h),
            int(b[2]+0.25*w), int(b[1]+0.78*h)]

def weapon_match_score(wb, pb):
    wc=bbox_center(wb)
    s  = max(0.0, 2.0 - dist2d(wc,bbox_center(pb))/max(1,bbox_height(pb)))
    s += 3.0*iou(wb,pb)
    if point_in_box(wc,pb):           s += 1.5
    if point_in_box(wc,make_hand_zone(pb)): s += 2.0
    return s

def assign_weapons(persons, objects):
    out = {pid: [] for pid in persons}
    conf_w = {oid:o for oid,o in objects.items()
              if o.get("confirmed") and o.get("class_name") in WEAPON_CLASSES}
    conf_p = {pid:p for pid,p in persons.items() if p.get("confirmed")}
    for oid,obj in conf_w.items():
        best_pid=None; best_s=-1e9; sec_s=-1e9
        for pid,p in conf_p.items():
            s=weapon_match_score(obj["bbox"],p["bbox"])
            if s>best_s: sec_s=best_s; best_s=s; best_pid=pid
            elif s>sec_s: sec_s=s
        if best_pid is None: continue
        margin=best_s-sec_s
        if best_s<1.2 or margin<0.35: continue
        if conf_p[best_pid].get("identity")=="owner" and margin<0.60: continue
        out[best_pid].append({
            "class_name":obj["class_name"],"conf":obj["conf"],
            "bbox":obj["bbox"],"object_id":oid,
        })
    return out


# =============================================================================
# OWNER RECOGNITION — INSIDE YOLO BOX ONLY
# =============================================================================
def recognize_in_yolo_box(frame, person_box):
    """
    Crop the top OWNER_FACE_REGION_FRAC of the YOLO box, run MTCNN inside it,
    embed with FaceNet.  Returns dict or None.
    """
    H, W = frame.shape[:2]
    x1,y1,x2,y2 = [int(v) for v in person_box]
    fy2 = y1 + int(OWNER_FACE_REGION_FRAC*(y2-y1))
    rx1=max(0,x1); ry1=max(0,y1)
    rx2=min(W-1,x2); ry2=min(H-1,fy2)
    if rx2-rx1 < MIN_OWNER_FACE_SIZE or ry2<=ry1: return None

    crop = frame[ry1:ry2, rx1:rx2]
    if crop.size == 0: return None

    faces = owner_recognizer.detect_faces(crop)
    if not faces: return None

    best = max(faces, key=lambda f: f["prob"])
    result = owner_recognizer.recognize_tensor(best["face_tensor"])
    dist   = float(result.get("distance", 999.0))

    fx1,fy1c,fx2,fy2c = best["face_box"]
    gfb = (rx1+fx1, ry1+fy1c, rx1+fx2, ry1+fy2c)
    return {"distance": dist, "face_box": gfb, "prob": best["prob"]}


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
    gray_s = cv2.resize(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        (80, 60), interpolation=cv2.INTER_AREA
    )
    if prev_gray_small is not None:
        diff          = cv2.absdiff(gray_s, prev_gray_small)
        changed_ratio = float(np.count_nonzero(diff > MOTION_PIXEL_DIFF)) / diff.size
    else:
        changed_ratio = 1.0
    prev_gray_small = gray_s

    if changed_ratio < MOTION_THRESHOLD and not active_persons:
        # Nothing moving AND no one being tracked — skip everything
        cv2.putText(frame, "IDLE", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120,120,120), 2)
        cv2.imshow("Threat Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
        continue
    # If persons are already tracked, always continue even if scene is still

    # ── STEP 1 : YOLO — PERSON DETECTION ONLY ────────────────────────────────
    if frame_count % YOLO_EVERY == 0:
        results    = model(frame)
        detections = results.xyxy[0].cpu().numpy()

    current_person_dets = []
    current_object_dets = []

    for det in detections:
        x1,y1,x2,y2,conf,cls = det
        cls=int(cls); conf=float(conf)
        cname = CLASS_NAMES.get(cls)
        if cname is None: continue
        box = [int(x1),int(y1),int(x2),int(y2)]

        if cname == "person":
            mid = match_to_tracks(box, active_persons,
                                  iou_thr=0.20, dist_thr=300)
            if mid is None:
                if conf < ENTRY_THRESHOLDS["person"]: continue
            else:
                if conf < KEEP_THRESHOLDS["person"]: continue
            current_person_dets.append({"bbox":box,"conf":conf,"source":"yolo"})

        else:   # weapons / objects — collected but processed only if non-owners present
            mid = match_to_tracks(box, active_objects,
                                  iou_thr=0.20, dist_thr=180)
            if mid is None:
                if conf < ENTRY_THRESHOLDS.get(cname,0.5): continue
            else:
                if conf < KEEP_THRESHOLDS.get(cname,0.35): continue
            current_object_dets.append({"bbox":box,"conf":conf,"class_name":cname})

    # ── STEP 2 : UPDATE PERSON TRACKS ────────────────────────────────────────
    matched_pids = set()
    used_dets    = set()

    # Protect locked owner first
    if recognized_owner_id is not None and recognized_owner_id in active_persons:
        ob = active_persons[recognized_owner_id]["bbox"]
        bi=None; bv=-1.0; bd=float("inf")
        for i,d in enumerate(current_person_dets):
            ov=iou(d["bbox"],ob)
            dv=dist2d(bbox_center(d["bbox"]),bbox_center(ob))
            if ov>bv or (abs(ov-bv)<1e-6 and dv<bd):
                bv=ov; bd=dv; bi=i
        ok = bi is not None and (bv>=0.10 or bd<120)
        if ok:
            update_person_track(recognized_owner_id,
                                current_person_dets[bi]["bbox"],
                                current_person_dets[bi]["conf"],
                                "yolo")
            matched_pids.add(recognized_owner_id)
            used_dets.add(bi)
        else:
            p = active_persons[recognized_owner_id]
            p["owner_confirmed"]=False; p["identity"]="unknown"
            p["owner_match_streak"]=0;  p["owner_distance"]=999.0
            p["face_box_global"]=None
            recognized_owner_id = None

    # Match remaining detections
    for i,d in enumerate(current_person_dets):
        if i in used_dets: continue
        skip = {recognized_owner_id} if recognized_owner_id else set()
        pid  = match_to_tracks(d["bbox"], active_persons,
                               iou_thr=0.20, dist_thr=300, skip=skip)
        if pid is None:
            pid = next_person_id; next_person_id += 1
            active_persons[pid] = {
                "bbox":d["bbox"], "conf":d["conf"], "missing":0,
                "identity":"unknown", "seen_count":1, "confirmed":True,
                "source":"yolo", "owner_confirmed":False,
                "owner_distance":999.0, "face_box_global":None,
                "owner_match_streak":0,
            }
        else:
            update_person_track(pid, d["bbox"], d["conf"], "yolo")
        matched_pids.add(pid)

    # Age out persons
    for pid in list(active_persons):
        if pid not in matched_pids:
            active_persons[pid]["missing"] += 1
            if active_persons[pid]["missing"] > PERSON_MISSING_FRAMES:
                del active_persons[pid]
                if recognized_owner_id == pid:
                    recognized_owner_id = None

    if recognized_owner_id not in active_persons:
        recognized_owner_id = None

    # ── STEP 3 : IF NO PERSONS — do nothing else ──────────────────────────────
    if not active_persons:
        cv2.putText(frame, "No person", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120,120,120), 2)
        cv2.imshow("Threat Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
        continue

    # ── STEP 4 : FACE RECOGNITION ON EACH PERSON ─────────────────────────────
    # Run every OWNER_RECOGNITION_EVERY frames when owner not yet locked
    if recognized_owner_id is None and frame_count % OWNER_RECOGNITION_EVERY == 0:
        best_pid=None; best_dist=999.0
        for pid, pdata in active_persons.items():
            if not pdata["confirmed"]: continue
            result = recognize_in_yolo_box(frame, pdata["bbox"])
            pdata["face_box_global"] = None
            if result is None:
                pdata["owner_distance"]     = 999.0
                pdata["owner_match_streak"] = 0
                continue
            dist = result["distance"]
            pdata["owner_distance"]   = dist
            pdata["face_box_global"]  = result["face_box"]
            if dist < OWNER_DISTANCE_THRESHOLD:
                pdata["owner_match_streak"] = pdata.get("owner_match_streak",0) + 1
            else:
                pdata["owner_match_streak"] = 0
            if pdata["owner_match_streak"] >= OWNER_CONFIRM_FRAMES and dist < best_dist:
                best_dist = dist; best_pid = pid
        if best_pid is not None:
            recognized_owner_id = best_pid

    # Sync identity
    for pid in active_persons:
        is_owner = (pid == recognized_owner_id)
        active_persons[pid]["identity"]        = "owner" if is_owner else "unknown"
        active_persons[pid]["owner_confirmed"] = is_owner

    # ── STEP 5 : DECIDE WHAT TO RUN PER PERSON ───────────────────────────────
    #  • OWNER  → draw OWNER box + distance, skip ALL analysis
    #  • OTHERS → run weapons, aggression, behavior, threat scoring

    non_owner_pids = [pid for pid in active_persons
                      if active_persons[pid]["confirmed"]
                      and pid != recognized_owner_id]

    # Only update objects / run threat analysis when there are non-owners
    weapon_assignments = {pid: [] for pid in active_persons}
    threat_results     = {}

    if non_owner_pids:
        # Update object tracks
        matched_oids = set()
        for d in current_object_dets:
            box=d["bbox"]; conf=d["conf"]; cname=d["class_name"]
            oid=match_to_tracks(box,active_objects,iou_thr=0.20,dist_thr=180)
            if oid is None:
                oid=next_object_id; next_object_id+=1
                active_objects[oid]={
                    "bbox":box,"conf":conf,"class_name":cname,
                    "missing":0,"seen_count":1,"confirmed":False,
                }
                object_class_history[oid]=deque(maxlen=5)
                object_class_history[oid].append(cname)
            else:
                ob=active_objects[oid]["bbox"]
                active_objects[oid]["bbox"]=[int(0.7*ob[i]+0.3*box[i]) for i in range(4)]
                active_objects[oid]["conf"]=0.7*active_objects[oid]["conf"]+0.3*conf
                active_objects[oid]["missing"]=0
                active_objects[oid]["seen_count"]+=1
                maybe_switch_class(oid,cname,conf)
            if active_objects[oid]["seen_count"] >= CLASS_CONFIRM_FRAMES.get(cname,1):
                active_objects[oid]["confirmed"]=True
            matched_oids.add(oid)

        for oid in list(active_objects):
            if oid not in matched_oids:
                active_objects[oid]["missing"]+=1
                switch_candidate_history[oid].clear()
                if active_objects[oid]["missing"] > OBJECT_MISSING_FRAMES:
                    del active_objects[oid]
                    object_class_history.pop(oid,None)
                    switch_candidate_history.pop(oid,None)

        threat_analyzer.cleanup_missing_tracks(active_persons.keys())
        weapon_assignments = assign_weapons(active_persons, active_objects)

        owner_bbox = (active_persons[recognized_owner_id]["bbox"]
                      if recognized_owner_id in active_persons else None)

        for pid in non_owner_pids:
            pdata = active_persons[pid]
            person  = {"id":pid,"bbox":pdata["bbox"],
                       "identity":"unknown",
                       "weapons":weapon_assignments.get(pid,[])}
            context = {"mode":"indoor",
                       "owner_present": recognized_owner_id in active_persons,
                       "time_of_day":"day",
                       "owner_bbox":owner_bbox,
                       "owner_id":recognized_owner_id}
            level, score, _, debug = threat_analyzer.analyze_person(
                frame, person, context, frame_index=frame_count)
            threat_results[pid] = (level, score, debug)

    else:
        # Only owner(s) in frame — clear stale object tracks gradually
        for oid in list(active_objects):
            active_objects[oid]["missing"] += 1
            if active_objects[oid]["missing"] > OBJECT_MISSING_FRAMES:
                del active_objects[oid]
                object_class_history.pop(oid,None)
                switch_candidate_history.pop(oid,None)

    # ── DRAW ─────────────────────────────────────────────────────────────────
    for pid, pdata in active_persons.items():
        if not pdata["confirmed"]: continue
        x1,y1,x2,y2 = pdata["bbox"]
        is_owner = (pid == recognized_owner_id)

        if is_owner:
            # OWNER — yellow box, minimal info
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),3)
            cv2.putText(frame,"OWNER",(x1,max(20,y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,0),2)
            dist_v = pdata.get("owner_distance",999.0)
            cv2.putText(frame,f"dist:{dist_v:.3f}",(x1,max(40,y1+16)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,0),1)
            fb = pdata.get("face_box_global")
            if fb:
                cv2.rectangle(frame,(fb[0],fb[1]),(fb[2],fb[3]),(200,200,0),1)

        else:
            level, score, debug = threat_results.get(pid, ("SAFE", 0.0, {}))
            color = ((0,0,255) if level=="THREAT" else
                     (0,255,255) if level=="SUSPICIOUS" else (0,255,0))
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,f"ID {pid} | {level} {score:.1f}",
                        (x1,max(20,y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.65,color,2)

            info = [
                f"threat_score: {score:.1f}",
                f"ID_score:     {debug.get('identity_score',0):.1f}",
                f"weapon_score: {debug.get('weapon_score',0):.1f}",
                f"aggr_score:   {debug.get('aggression_score',0):.2f}",
                f"aggr_pts:     {debug.get('aggression_points',0):.1f}",
                f"behavior:     {debug.get('behavior_score',0):.1f}",
                f"owner_prox:   {debug.get('owner_proximity_score',0):.1f}",
                f"robot_dist:   {debug.get('robot_distance_score',0):.1f}",
                f"person_conf:  {pdata['conf']:.2f}",
                f"owner_dist:   {pdata.get('owner_distance',999):.3f}"
                f" | streak:{pdata.get('owner_match_streak',0)}",
            ]
            wlist = weapon_assignments.get(pid, [])
            if wlist:
                info.append("weapons: " + ", ".join(
                    f"{w['class_name']}({w['conf']:.2f})" for w in wlist[:3]))
            else:
                info.append("weapons: none")
            draw_text_block(frame, info, x1, min(frame.shape[0]-130, y2+20), color)

            fb = pdata.get("face_box_global")
            if fb:
                cv2.rectangle(frame,(fb[0],fb[1]),(fb[2],fb[3]),(200,200,0),1)

    # Draw confirmed weapons
    for obj in active_objects.values():
        if not obj["confirmed"]: continue
        ox1,oy1,ox2,oy2 = obj["bbox"]
        cv2.rectangle(frame,(ox1,oy1),(ox2,oy2),(255,0,0),2)
        cv2.putText(frame,f"{obj['class_name']} {obj['conf']:.2f}",
                    (ox1,max(20,oy1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)

    # HUD
    owner_txt = (f"OWNER ID {recognized_owner_id}"
                 if recognized_owner_id in active_persons else "OWNER: searching")
    cv2.putText(frame, owner_txt, (10,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    cv2.putText(frame, f"motion:{changed_ratio*100:.1f}%", (10,48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

    cv2.imshow("Threat Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.stop()
cv2.destroyAllWindows()
