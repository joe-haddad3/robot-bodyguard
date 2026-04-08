"""
Run this script ONCE to enroll the owner's face.
It opens the camera, captures face samples using MTCNN (face alignment),
and saves face_db/owner_embeddings.npy for FaceNetOwnerRecognizer.

Usage:
    python enroll_owner.py

Controls:
    SPACE  - capture a sample (aim for 30-50 samples, vary your head angle)
    S      - save and exit
    ESC    - exit without saving

Tips for best recognition:
    - Capture from multiple angles (left, right, up, down, straight)
    - Vary lighting if possible
    - Aim for 40+ samples
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_db")
SAVE_PATH = os.path.join(SAVE_DIR, "owner_embeddings.npy")
CAMERA_INDEX = 0
TARGET_SAMPLES = 40


def l2_normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def main():
    print("[enroll_owner] Loading MTCNN + FaceNet models...")

    # MTCNN aligns faces to standard eye positions before embedding.
    # keep_all=False: only take the largest/closest face per frame.
    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        keep_all=False,
        min_face_size=80,
        thresholds=[0.6, 0.7, 0.7],
        post_process=True,
        device="cpu",
    )
    model = InceptionResnetV1(pretrained="vggface2").eval()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}. Try changing CAMERA_INDEX.")
        sys.exit(1)

    embeddings = []
    print(f"\n[enroll_owner] Camera open.")
    print(f"[enroll_owner] SPACE=capture  S=save  ESC=quit")
    print(f"[enroll_owner] Aim for {TARGET_SAMPLES} samples — vary your head angle!\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # Detect face bounding boxes for display
        boxes, probs = mtcnn.detect(pil_img)

        display = frame.copy()
        face_found = False

        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob is None or prob < 0.85:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    display, f"conf:{prob:.2f}",
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )
                face_found = True

        color = (0, 255, 255) if face_found else (0, 0, 255)
        status = f"Samples: {len(embeddings)}/{TARGET_SAMPLES}  |  SPACE=capture  S=save  ESC=quit"
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if not face_found:
            cv2.putText(display, "No face detected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Enroll Owner", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("[enroll_owner] Cancelled.")
            break

        elif key == ord("s"):
            if len(embeddings) == 0:
                print("[WARN] No samples captured yet.")
            else:
                os.makedirs(SAVE_DIR, exist_ok=True)
                arr = np.array(embeddings)
                np.save(SAVE_PATH, arr)
                print(f"\n[enroll_owner] Saved {len(arr)} embeddings to {SAVE_PATH}")

                mean_raw = arr.mean(axis=0)
                mean_norm = mean_raw / (np.linalg.norm(mean_raw) + 1e-10)
                mean_path = os.path.join(SAVE_DIR, "owner_mean_embedding.npy")
                np.save(mean_path, mean_norm)
                print(f"[enroll_owner] Saved mean embedding to {mean_path}")
            break

        elif key == ord(" "):
            # Get MTCNN-aligned face tensor
            face_tensor = mtcnn(pil_img)

            if face_tensor is None:
                print("[WARN] No face detected — move closer or improve lighting.")
                continue

            # face_tensor: (3, 160, 160) — already aligned and normalized
            with torch.no_grad():
                emb = model(face_tensor.unsqueeze(0)).cpu().numpy()[0]

            emb = l2_normalize(emb)
            embeddings.append(emb)
            print(f"[OK] Captured sample {len(embeddings)}")

            # Flash green to confirm capture
            flash = display.copy()
            cv2.rectangle(flash, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 8)
            cv2.imshow("Enroll Owner", flash)
            cv2.waitKey(150)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
