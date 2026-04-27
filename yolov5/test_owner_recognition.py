"""
Standalone owner recognition test.

Opens the webcam, detects faces with MTCNN, embeds each aligned face with
FaceNet, and shows OWNER / UNKNOWN in real time.

Usage:
    python test_owner_recognition.py

Press Q or ESC to quit.
"""

import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from owner_recognizer_facenet import FaceNetOwnerRecognizer, cosine_distance, l2_normalize


FACENET_STRICT_THRESHOLD = 0.30
CAMERA_INDEX = 0
CAM_W = 1280
CAM_H = 720
CAM_FPS = 30
MIN_FACE_PROB = 0.85


def embed_aligned_face_tensor(recognizer, face_tensor):
    """Embed an MTCNN-aligned face tensor using the loaded FaceNet backend."""
    if face_tensor is None or not recognizer.enabled:
        return np.zeros(512, dtype=np.float32)

    face_np = face_tensor.detach().cpu().numpy()
    if face_np.ndim == 3:
        face_np = face_np[np.newaxis, ...]
    face_np = face_np.astype(np.float32)

    if recognizer._session is not None:
        emb = recognizer._session.run(None, {recognizer._input_name: face_np})[0][0]
    else:
        with torch.no_grad():
            emb = recognizer._pt_model(torch.from_numpy(face_np)).cpu().numpy()[0]

    return l2_normalize(emb)


def classify_embedding(recognizer, emb):
    mean_dist = (
        cosine_distance(emb, recognizer.mean_embedding)
        if recognizer.mean_embedding is not None else 999.0
    )
    sample_dists = [cosine_distance(emb, ref) for ref in recognizer.owner_embeddings]
    best_dist = min(sample_dists) if sample_dists else 999.0
    is_owner = mean_dist < FACENET_STRICT_THRESHOLD and best_dist < FACENET_STRICT_THRESHOLD
    return is_owner, mean_dist, best_dist


def main():
    recognizer = FaceNetOwnerRecognizer(threshold=FACENET_STRICT_THRESHOLD)
    if not recognizer.enabled:
        print("[ERROR] No owner database found. Run enroll_owner.py first.")
        return 1

    try:
        from facenet_pytorch import MTCNN
    except ImportError:
        print("[ERROR] facenet-pytorch is not installed. Run: pip install facenet-pytorch")
        return 1

    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        keep_all=True,
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.7],
        post_process=True,
        device="cpu",
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}.")
        return 1

    print(
        f"[OK] Owner recognition test running "
        f"(threshold={FACENET_STRICT_THRESHOLD}, samples={len(recognizer.owner_embeddings)})"
    )
    print("[OK] Press Q or ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        boxes, probs = mtcnn.detect(pil_img)
        face_tensors = None
        if boxes is not None and len(boxes) > 0:
            face_tensors = mtcnn.extract(pil_img, boxes, save_path=None)

        if boxes is not None and probs is not None and face_tensors is not None:
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob is None or prob < MIN_FACE_PROB or i >= len(face_tensors):
                    continue

                emb = embed_aligned_face_tensor(recognizer, face_tensors[i])
                is_owner, mean_dist, best_dist = classify_embedding(recognizer, emb)

                x1, y1, x2, y2 = [int(v) for v in box]
                label = "OWNER" if is_owner else "UNKNOWN"
                color = (0, 220, 0) if is_owner else (0, 0, 220)

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    display,
                    label,
                    (x1, max(22, y1 - 26)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )
                cv2.putText(
                    display,
                    f"mean={mean_dist:.3f} best={best_dist:.3f} thr={FACENET_STRICT_THRESHOLD}",
                    (x1, max(42, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )

        cv2.imshow("Owner Recognition Test", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
