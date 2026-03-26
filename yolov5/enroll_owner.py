"""
Run this script ONCE to enroll the owner's face.
It opens the camera, captures face samples, and saves
face_db/owner_embeddings.npy for FaceNetOwnerRecognizer.

Usage:
    python enroll_owner.py

Controls:
    SPACE  - capture a sample (aim for 30-50 samples)
    S      - save and exit
    ESC    - exit without saving
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_db")
SAVE_PATH = os.path.join(SAVE_DIR, "owner_embeddings.npy")
CAMERA_INDEX = 0     # change if your camera is on a different index
TARGET_SAMPLES = 40
FACE_SIZE = (160, 160)


def l2_normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def embed(model, face_rgb):
    face_img = Image.fromarray(face_rgb).resize(FACE_SIZE)
    face_np = np.asarray(face_img).astype(np.float32) / 255.0
    face_np = (face_np - 0.5) / 0.5
    tensor = torch.tensor(face_np).permute(2, 0, 1).unsqueeze(0).float()
    with torch.no_grad():
        emb = model(tensor).cpu().numpy()[0]
    return l2_normalize(emb)


def main():
    print("[enroll_owner] Loading FaceNet model...")
    model = InceptionResnetV1(pretrained="vggface2").eval()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}. Try changing CAMERA_INDEX.")
        sys.exit(1)

    embeddings = []
    print(f"\n[enroll_owner] Camera open. Press SPACE to capture, S to save, ESC to quit.")
    print(f"[enroll_owner] Aim for {TARGET_SAMPLES} samples.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        display = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        status = f"Samples: {len(embeddings)}/{TARGET_SAMPLES}  |  SPACE=capture  S=save  ESC=quit"
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
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
            if len(faces) == 0:
                print("[WARN] No face detected — move closer or improve lighting.")
                continue

            x, y, w, h = faces[0]
            H, W = frame.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            emb = embed(model, rgb_crop)
            embeddings.append(emb)
            print(f"[OK] Captured sample {len(embeddings)}")

            # Flash green to confirm
            flash = display.copy()
            cv2.rectangle(flash, (x1, y1), (x2, y2), (0, 255, 0), 6)
            cv2.imshow("Enroll Owner", flash)
            cv2.waitKey(150)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
