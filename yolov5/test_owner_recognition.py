"""
Standalone owner recognition test.

Opens the camera and runs MTCNN + FaceNet on every detected face.
Shows the cosine distance and OWNER / UNKNOWN verdict in real time.

Usage:
    python test_owner_recognition.py

Press Q or ESC to quit.
"""

import sys
import os
import cv2
import numpy as np
from PIL import Image

# ── Load recognizer ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from owner_recognizer_facenet import FaceNetOwnerRecognizer, cosine_distance, l2_normalize

FACENET_STRICT_THRESHOLD = 0.30   # must match threat_detection_test.py
CAMERA_INDEX = 0

recognizer = FaceNetOwnerRecognizer(threshold=FACENET_STRICT_THRESHOLD)
if not recognizer.enabled:
    print("[ERROR] No owner database found. Run enroll_owner.py first.")
    sys.exit(1)

# ── Load MTCNN ───────────────────────────────────────────────────────────────
try:
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(
        image_size=160, margin=20, keep_all=True,
        min_face_size=40, thresholds=[0.6, 0.7, 0.7],
        post_process=False, device="cpu",
    )
    print("[OK] MTCNN loaded")
except ImportError:
    print("[ERROR] facenet-pytorch not installed. Run: pip install facenet-pytorch")
    sys.exit(1)

# ── Camera ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"[ERROR] Cannot open camera {CAMERA_INDEX}")
    sys.exit(1)

print(f"[OK] Camera open — threshold={FACENET_STRICT_THRESHOLD}  (enrolled: {len(recognizer.owner_embeddings)} samples)")
print("[OK] Press Q/ESC to quit\n")

import torch

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes, probs = mtcnn.detect(pil_img)

    display = frame.copy()

    if boxes is not None:
        for i, (box, prob) in enumerate(zip(boxes, probs if probs is not None else [])):
            if prob is None or prob < 0.85:
                continue

            x1, y1, x2, y2 = [int(v) for v in box]

            # Crop face and embed
            pad = 15
            h, w = frame.shape[:2]
            fx1 = max(0, x1 - pad); fy1 = max(0, y1 - pad)
            fx2 = min(w, x2 + pad); fy2 = min(h, y2 + pad)
            face_bgr = frame[fy1:fy2, fx1:fx2]
            if face_bgr.size == 0:
                continue

            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            emb = recognizer.embed_face_rgb(face_rgb)

            mean_dist = (
                cosine_distance(emb, recognizer.mean_embedding)
                if recognizer.mean_embedding is not None else 999.0
            )
            sample_dists = [cosine_distance(emb, ref) for ref in recognizer.owner_embeddings]
            best_dist = min(sample_dists) if sample_dists else 999.0

            is_owner = (mean_dist < FACENET_STRICT_THRESHOLD and
                        best_dist < FACENET_STRICT_THRESHOLD)

            label   = "OWNER" if is_owner else "UNKNOWN"
            color   = (0, 220, 0) if is_owner else (0, 0, 220)

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, label,
                        (x1, max(20, y1 - 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(display,
                        f"mean={mean_dist:.3f}  best={best_dist:.3f}  thr={FACENET_STRICT_THRESHOLD}",
                        (x1, max(40, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            print(f"Face {i}: {label}  mean_dist={mean_dist:.3f}  best_dist={best_dist:.3f}")

    cv2.imshow("Owner Recognition Test", display)
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord("q")):
        break

cap.release()
cv2.destroyAllWindows()
