import os
import cv2
import time
import pickle
import mediapipe as mp


MODEL_PATH = "owner_lbph_model.yml"
LABELS_PATH = "owner_lbph_labels.pkl"

FACE_SIZE = (160, 160)
LBPH_THRESHOLD = 80.0
SNAPSHOT_DIR = "face_snapshots"

os.makedirs(SNAPSHOT_DIR, exist_ok=True)


class OwnerRecognizer:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
        if not os.path.exists(LABELS_PATH):
            raise FileNotFoundError(f"Missing labels: {LABELS_PATH}")

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(MODEL_PATH)

        with open(LABELS_PATH, "rb") as f:
            self.label_map = pickle.load(f)

        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )

        self.last_unknown_snapshot_time = 0
        self.unknown_snapshot_cooldown = 5.0

    def _detect_largest_face_in_roi(self, roi_bgr):
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb)

        if not results.detections:
            return None

        H, W = roi_bgr.shape[:2]
        best = None
        best_area = -1

        for det in results.detections:
            rel = det.location_data.relative_bounding_box
            x = int(rel.xmin * W)
            y = int(rel.ymin * H)
            w = int(rel.width * W)
            h = int(rel.height * H)

            x = max(0, x)
            y = max(0, y)
            x2 = min(W, x + max(1, w))
            y2 = min(H, y + max(1, h))

            area = (x2 - x) * (y2 - y)
            if area > best_area:
                best_area = area
                best = (x, y, x2, y2)

        return best

    def recognize_person(self, frame_bgr, person_box):
        x1, y1, x2, y2 = person_box
        H, W = frame_bgr.shape[:2]
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W - 1, x2)
        y2 = min(H - 1, y2)

        if x2 <= x1 or y2 <= y1:
            return {
                "status": "invalid_roi",
                "name": None,
                "score": None,
                "face_box_global": None
            }

        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return {
                "status": "empty_roi",
                "name": None,
                "score": None,
                "face_box_global": None
            }

        face_box_local = self._detect_largest_face_in_roi(roi)
        if face_box_local is None:
            return {
                "status": "no_face",
                "name": None,
                "score": None,
                "face_box_global": None
            }

        fx1, fy1, fx2, fy2 = face_box_local
        face_crop = roi[fy1:fy2, fx1:fx2]

        if face_crop.size == 0:
            return {
                "status": "bad_face_crop",
                "name": None,
                "score": None,
                "face_box_global": None
            }

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, FACE_SIZE)
        gray = cv2.equalizeHist(gray)

        label_id, score = self.recognizer.predict(gray)

        global_face_box = (x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2)

        if score <= LBPH_THRESHOLD:
            return {
                "status": "authorized",
                "name": self.label_map.get(label_id, "owner"),
                "score": score,
                "face_box_global": global_face_box
            }

        return {
            "status": "unknown",
            "name": None,
            "score": score,
            "face_box_global": global_face_box
        }

    def maybe_save_unknown_snapshot(self, frame_bgr, person_box, prefix="unknown"):
        now = time.time()
        if now - self.last_unknown_snapshot_time < self.unknown_snapshot_cooldown:
            return

        x1, y1, x2, y2 = person_box
        H, W = frame_bgr.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W - 1, x2)
        y2 = min(H - 1, y2)

        if x2 <= x1 or y2 <= y1:
            return

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(SNAPSHOT_DIR, f"{prefix}_{ts}.jpg")
        cv2.imwrite(out_path, crop)
        self.last_unknown_snapshot_time = now
