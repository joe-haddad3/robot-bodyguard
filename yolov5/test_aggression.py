"""
Standalone aggression score test.

Opens the camera, detects faces with Haar, and runs AggressionAnalyzer
on each face crop in real time.  Shows face_score and aggression_score
overlaid on the frame so you can see how they respond to expressions
and hand gestures — completely independent of the threat pipeline.

Usage:
    python test_aggression.py

Press Q or ESC to quit.
"""

import sys
import os
import time
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggression_analyzer import AggressionAnalyzer

CAMERA_INDEX = 0
FACE_MIN_SIZE = (60, 60)

# One shared instance — fine here because this is a single-person test.
analyzer = AggressionAnalyzer()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"[ERROR] Cannot open camera {CAMERA_INDEX}")
    sys.exit(1)

print("[OK] Camera open — point it at your face and try angry/neutral/happy expressions")
print("[OK] Press Q/ESC to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=FACE_MIN_SIZE
    )

    display = frame.copy()

    if len(faces) > 0:
        # Pick the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Crop with a small padding
        pad = 20
        fH, fW = frame.shape[:2]
        cx1 = max(0, x - pad);  cy1 = max(0, y - pad)
        cx2 = min(fW, x + w + pad); cy2 = min(fH, y + h + pad)
        crop = frame[cy1:cy2, cx1:cx2]

        if crop.size > 0:
            timestamp_ms = int(time.time() * 1000)
            result = analyzer.analyze(crop, timestamp_ms)

            face_score       = result.get("face_score", 0.0)
            aggression_score = result.get("aggression_score", 0.0)
            dbg              = result.get("debug", {})

            brow_down   = dbg.get("brow_down", 0.0)
            mouth_close = dbg.get("mouth_close", 0.0)
            eye_squint  = dbg.get("eye_squint", 0.0)
            combo       = dbg.get("brow_squint_combo", 0.0)

            # Colour the box: green → orange → red as anger rises
            if face_score < 0.15:
                color = (0, 220, 0)       # green  — calm
            elif face_score < 0.22:
                color = (0, 200, 255)     # orange — approaching threshold
            else:
                color = (0, 0, 220)       # red    — ANGRY (threshold 0.22 crossed)

            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

            # Overlay each contributing signal so you can tune thresholds
            lines = [
                f"face_score:   {face_score:.3f}",
                f"aggr_score:   {aggression_score:.3f}",
                f"brow_frown:   {brow_down:.3f}",
                f"mouth_close:  {mouth_close:.3f}",
                f"eye_squint:   {eye_squint:.3f}",
                f"brow+squint:  {combo:.3f}",
            ]
            for i, line in enumerate(lines):
                cv2.putText(display, line,
                            (x, max(14 + i * 16, y - 14 + i * 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

            # Progress bar for face_score
            bar_w = int(w * min(1.0, face_score))
            cv2.rectangle(display, (x, y + h + 4), (x + bar_w, y + h + 10), color, -1)
            cv2.rectangle(display, (x, y + h + 4), (x + w,     y + h + 10), (80, 80, 80), 1)

            print(f"face={face_score:.3f}  aggr={aggression_score:.3f}  "
                  f"brow={brow_down:.3f}  mouth_close={mouth_close:.3f}  "
                  f"squint={eye_squint:.3f}  combo={combo:.3f}")
    else:
        cv2.putText(display, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

    cv2.imshow("Aggression Score Test", display)
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord("q")):
        break

cap.release()
cv2.destroyAllWindows()
