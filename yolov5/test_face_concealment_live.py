"""
Standalone face-concealment test.

Logic mirrored from the main pipeline:
  - MTCNN tries to find a face in the upper 45% of each detected person box.
  - No face found for FACE_CONCEALED_MIN_FRAMES consecutive frames → SUSPICIOUS.
  - Face found → SAFE.

Owner-present condition is simulated with the 'o' key toggle:
  Press 'o' to toggle "owner in frame" on/off.
  Concealment is only flagged when owner is simulated as present.

Controls:
  o   — toggle owner-present simulation
  esc — quit
"""

import cv2
import numpy as np
import torch
import os
from types import SimpleNamespace
from collections import defaultdict

from camera_utils import make_camera
from mask_detector import MaskDetector

# ── Minimal YOLO person detector ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
model = torch.hub.load(
    os.path.dirname(os.path.abspath(__file__)),
    "custom",
    path=MODEL_PATH,
    source="local",
    force_reload=False,
)
model.conf = 0.35

# ── Face-concealment settings (match main pipeline) ──────────────────────────
FACE_CONCEALED_MIN_FRAMES = 4   # consecutive concealed frames before flagging
MIN_BOX_HEIGHT = 120            # skip tiny/distant boxes

# ── State ─────────────────────────────────────────────────────────────────────
mask_detector = MaskDetector()
concealed_streak = defaultdict(int)   # object_id → streak count
owner_simulated  = True               # toggled with 'o'

CAM_W, CAM_H, CAM_FPS = 640, 480, 30
WINDOW = "Face Concealment Test  (o=toggle owner  esc=quit)"


def draw_label(frame, text, x, y, bg_color, text_color=(255, 255, 255)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x, y - th - 6), (x + tw + 8, y + 4), bg_color, -1)
    cv2.putText(frame, text, (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                text_color, 2, cv2.LINE_AA)


def main():
    global owner_simulated

    cam = make_camera(src=0, width=CAM_W, height=CAM_H, fps=CAM_FPS).start()
    frame_idx = 0

    while True:
        frame = cam.read()
        if frame is None:
            continue

        fh, fw = frame.shape[:2]
        display = frame.copy()

        # ── YOLO person detection ─────────────────────────────────────────────
        results  = model(frame)
        dets     = results.xyxy[0].cpu().numpy()
        persons  = []
        for det in dets:
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 0 and conf >= 0.35:
                persons.append([int(x1), int(y1), int(x2), int(y2)])

        # ── Per-person concealment check ──────────────────────────────────────
        active_ids = set()
        for pid, (x1, y1, x2, y2) in enumerate(persons):
            active_ids.add(pid)
            box_h = y2 - y1

            level = "SAFE"
            label_text = "SAFE"
            box_color  = (0, 200, 0)

            if not owner_simulated:
                # No owner in frame → always SAFE (match production rule)
                concealed_streak[pid] = 0
                label_text = "SAFE (no owner)"
            elif not mask_detector.enabled:
                label_text = "SAFE (detector off)"
            else:
                # Crop upper 45% of person box — where the face is
                face_y2   = y1 + int(box_h * 0.45)
                face_crop = frame[max(0, y1):min(fh, face_y2), max(0, x1):min(fw, x2)]

                if face_crop.size > 0:
                    result = mask_detector.detect(face_crop, person_box_height=box_h)

                    if result.get("mask"):
                        concealed_streak[pid] += 1
                    else:
                        concealed_streak[pid] = 0

                    streak = concealed_streak[pid]

                    if streak >= FACE_CONCEALED_MIN_FRAMES:
                        level      = "SUSPICIOUS"
                        label_text = f"SUSPICIOUS — FACE CONCEALED ({streak}f)"
                        box_color  = (0, 165, 255)  # orange
                    elif streak > 0:
                        label_text = f"VISIBLE? streak={streak}/{FACE_CONCEALED_MIN_FRAMES}"
                        box_color  = (0, 220, 220)
                    else:
                        conf_val = result.get("confidence", 0.0)
                        label_text = f"SAFE — face visible ({conf_val:.2f})"

            cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)
            draw_label(display, label_text, x1, max(20, y1 - 8), box_color)

        # Expire streaks for persons no longer detected
        for pid in list(concealed_streak.keys()):
            if pid not in active_ids:
                del concealed_streak[pid]

        # ── HUD ───────────────────────────────────────────────────────────────
        owner_color = (0, 200, 0) if owner_simulated else (80, 80, 80)
        draw_label(display, f"Owner in frame: {'YES' if owner_simulated else 'NO (press o)'}",
                   10, 30, owner_color)
        draw_label(display, f"Persons: {len(persons)}  frame: {frame_idx}",
                   10, 62, (40, 40, 40))

        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('o'):
            owner_simulated = not owner_simulated
            print(f"[TEST] Owner simulated: {owner_simulated}")

        frame_idx += 1

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
