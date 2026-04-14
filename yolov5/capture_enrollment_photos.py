"""
Capture enrollment photos for owner recognition.

Opens the camera, detects your face with MTCNN, and auto-saves face crops
every ~1.5 seconds when a good face is visible. You just need to sit/stand
in front of the camera and slowly turn your head — the script does the rest.

Photos are saved to:  face_db/enrollment_photos/

After capturing, run:
    python enroll_owner.py

Controls:
    Q / ESC  — quit

Tips for good enrollment:
    - Aim for 60-80 photos total
    - Spend ~5 seconds at each angle: straight, slight left, slight right,
      slight up, slight down, 45° left, 45° right
    - Do one pass in normal lighting, one pass near a window or lamp
    - Remove glasses then put them back on if you wear them
    - The more variety, the more robust recognition becomes
"""

import os
import sys
import time
import cv2
import numpy as np
from PIL import Image

try:
    from facenet_pytorch import MTCNN
except ImportError:
    print("[ERROR] facenet-pytorch is not installed.")
    print("  Run: pip install facenet-pytorch")
    sys.exit(1)

SAVE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_db", "enrollment_photos")
CAMERA_INDEX   = 0
CAPTURE_EVERY  = 1.5   # seconds between auto-captures
MIN_FACE_CONF  = 0.92  # MTCNN confidence gate — only save high-quality detections
MIN_FACE_PX    = 80    # minimum face width/height in pixels


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Count existing photos so we don't overwrite a previous session
    existing = [f for f in os.listdir(SAVE_DIR) if f.endswith(".jpg")]
    photo_idx = len(existing)
    print(f"[capture] Found {photo_idx} existing photos in {SAVE_DIR}")

    print("[capture] Loading MTCNN...")
    mtcnn = MTCNN(
        image_size=160,
        margin=30,          # larger margin = more context around face
        keep_all=False,     # only the closest/largest face
        min_face_size=MIN_FACE_PX,
        thresholds=[0.6, 0.7, 0.7],
        post_process=False, # we want the raw PIL crop, not the normalised tensor
        device="cpu",
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}.")
        sys.exit(1)

    last_capture = 0.0
    print(f"\n[capture] Auto-capturing every {CAPTURE_EVERY}s when a face is visible.")
    print(f"[capture] Slowly vary your head angle. Q/ESC to stop.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img   = Image.fromarray(frame_rgb)

        boxes, probs = mtcnn.detect(pil_img)

        display     = frame.copy()
        face_visible = False
        best_box    = None
        best_prob   = 0.0

        if boxes is not None:
            for box, prob in zip(boxes, probs or []):
                if prob is None or prob < MIN_FACE_CONF:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box]
                w = x2 - x1; h = y2 - y1
                if w < MIN_FACE_PX or h < MIN_FACE_PX:
                    continue
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, f"{prob:.2f}", (x1, max(14, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                if prob > best_prob:
                    best_prob = prob
                    best_box  = (x1, y1, x2, y2)
                face_visible = True

        now  = time.time()
        ready_to_capture = face_visible and (now - last_capture) >= CAPTURE_EVERY

        # Status bar
        if face_visible:
            bar_color = (0, 200, 0) if ready_to_capture else (0, 180, 255)
            cooldown  = max(0.0, CAPTURE_EVERY - (now - last_capture))
            status    = (f"Photos: {photo_idx}  |  Capturing in {cooldown:.1f}s"
                         if not ready_to_capture else
                         f"Photos: {photo_idx}  |  Capturing...")
        else:
            bar_color = (0, 0, 200)
            status    = f"Photos: {photo_idx}  |  No face — move closer or improve lighting"

        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2)
        cv2.putText(display, "Q/ESC = stop", (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Auto-capture
        if ready_to_capture and best_box is not None:
            x1, y1, x2, y2 = best_box
            # Add padding around face box
            pad = 20
            h_img, w_img = frame.shape[:2]
            fx1 = max(0, x1 - pad); fy1 = max(0, y1 - pad)
            fx2 = min(w_img, x2 + pad); fy2 = min(h_img, y2 + pad)
            face_crop = frame[fy1:fy2, fx1:fx2]

            if face_crop.size > 0:
                fname = os.path.join(SAVE_DIR, f"face_{photo_idx:04d}.jpg")
                cv2.imwrite(fname, face_crop)
                photo_idx += 1
                last_capture = now
                print(f"[capture] Saved photo {photo_idx}  (conf={best_prob:.3f})")

                # Flash green border to confirm
                flash = display.copy()
                cv2.rectangle(flash, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 10)
                cv2.imshow("Capture Enrollment Photos", flash)
                cv2.waitKey(120)

        cv2.imshow("Capture Enrollment Photos", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[capture] Done. {photo_idx} photos saved to:\n  {SAVE_DIR}")
    print(f"\nNext step: python enroll_owner.py")


if __name__ == "__main__":
    main()
