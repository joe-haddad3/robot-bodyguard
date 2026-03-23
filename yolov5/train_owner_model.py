import os
import cv2
import pickle
import mediapipe as mp
import numpy as np

OWNER_DIR = "owner_faces"
MODEL_PATH = "owner_lbph_model.yml"
LABELS_PATH = "owner_lbph_labels.pkl"

FACE_SIZE = (160, 160)
MIN_DET_CONF = 0.5


class MediaPipeFaceCropper:
    def __init__(self):
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=MIN_DET_CONF
        )

    def crop_largest_face(self, image_bgr):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)

        if not results.detections:
            return None

        H, W = image_bgr.shape[:2]
        best_box = None
        best_area = -1

        for det in results.detections:
            rel = det.location_data.relative_bounding_box
            x = int(rel.xmin * W)
            y = int(rel.ymin * H)
            w = int(rel.width * W)
            h = int(rel.height * H)

            x = max(0, x)
            y = max(0, y)
            w = max(1, w)
            h = max(1, h)

            x2 = min(W, x + w)
            y2 = min(H, y + h)

            area = (x2 - x) * (y2 - y)
            if area > best_area:
                best_area = area
                best_box = (x, y, x2, y2)

        if best_box is None:
            return None

        x1, y1, x2, y2 = best_box
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        return crop


def load_training_faces():
    cropper = MediaPipeFaceCropper()
    images = []
    labels = []
    label_map = {}
    next_label = 0

    if not os.path.isdir(OWNER_DIR):
        raise FileNotFoundError(f"Missing folder: {OWNER_DIR}")

    for person_name in sorted(os.listdir(OWNER_DIR)):
        person_dir = os.path.join(OWNER_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        label_map[next_label] = person_name

        for fname in sorted(os.listdir(person_dir)):
            path = os.path.join(person_dir, fname)

            if not os.path.isfile(path):
                continue

            img = cv2.imread(path)
            if img is None:
                print(f"[WARN] Could not read {path}")
                continue

            face_crop = cropper.crop_largest_face(img)
            if face_crop is None:
                print(f"[WARN] No face found in {path}")
                continue

            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, FACE_SIZE)
            gray = cv2.equalizeHist(gray)

            images.append(gray)
            labels.append(next_label)
            print(f"[OK] Added training face: {path}")

        next_label += 1

    return images, np.array(labels, dtype=np.int32), label_map
    
def main():
    images, labels, label_map = load_training_faces()

    if len(images) == 0:
        raise RuntimeError("No valid training faces found.")

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )
    recognizer.train(images, labels)

    recognizer.save(MODEL_PATH)

    with open(LABELS_PATH, "wb") as f:
        pickle.dump(label_map, f)

    print(f"[OK] Saved model to {MODEL_PATH}")
    print(f"[OK] Saved labels to {LABELS_PATH}")
    print(f"[INFO] Label map: {label_map}")


if __name__ == "__main__":
    main()
