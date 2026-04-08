import os
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

_DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "face_db", "owner_embeddings.npy"
)
_DEFAULT_MEAN_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "face_db", "owner_mean_embedding.npy"
)


def l2_normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-10:
        return v
    return v / n


def cosine_distance(a, b):
    return 1.0 - float(np.dot(a, b))


class FaceNetOwnerRecognizer:
    def __init__(self, db_path=None, threshold=0.30):
        if db_path is None:
            db_path = _DEFAULT_DB_PATH
        self.threshold = threshold

        # MTCNN: detects faces AND aligns them (eyes at fixed positions).
        # This is critical — FaceNet accuracy drops heavily without alignment.
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            keep_all=True,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            post_process=True,
            device="cpu",
        )

        self.enabled = os.path.exists(db_path)

        if not self.enabled:
            print(
                f"[FaceNetOwnerRecognizer] WARNING: Database not found at '{db_path}'.\n"
                "  Owner recognition is DISABLED.\n"
                "  Run enroll_owner.py first to enable it."
            )
            self.owner_embeddings = np.empty((0, 512), dtype=np.float32)
            self.mean_embedding = None
            self.model = None
            return

        self.owner_embeddings = np.load(db_path)

        if os.path.exists(_DEFAULT_MEAN_PATH):
            raw_mean = np.load(_DEFAULT_MEAN_PATH)
        else:
            raw_mean = self.owner_embeddings.mean(axis=0)

        self.mean_embedding = l2_normalize(raw_mean)
        self.model = InceptionResnetV1(pretrained="vggface2").eval()

        print(
            f"[FaceNetOwnerRecognizer] Loaded {len(self.owner_embeddings)} embeddings. "
            f"Threshold: {threshold}"
        )

    # ------------------------------------------------------------------
    # FACE DETECTION (called from main loop — full frame)
    # ------------------------------------------------------------------

    def detect_faces(self, frame_bgr):
        """
        Detect and align all faces in a full BGR frame using MTCNN.
        Returns list of dicts: {face_box, face_tensor, center, prob}
        face_tensor is pre-aligned (3, 160, 160) — ready for FaceNet.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        boxes, probs = self.mtcnn.detect(pil_img)
        if boxes is None:
            return []

        face_tensors = self.mtcnn(pil_img)
        if face_tensors is None:
            return []

        if face_tensors.ndim == 3:
            face_tensors = face_tensors.unsqueeze(0)

        H, W = frame_bgr.shape[:2]
        results = []

        for i, (box, prob) in enumerate(zip(boxes, probs)):
            if prob is None or prob < 0.85:
                continue
            if i >= len(face_tensors):
                break

            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(W, int(box[2]))
            y2 = min(H, int(box[3]))

            if x2 <= x1 or y2 <= y1:
                continue

            results.append({
                "face_box": (x1, y1, x2, y2),
                "face_tensor": face_tensors[i],
                "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                "prob": float(prob),
            })

        return results

    # ------------------------------------------------------------------
    # RECOGNITION
    # ------------------------------------------------------------------

    def recognize_tensor(self, face_tensor):
        """Recognize from a pre-aligned MTCNN face tensor (3, 160, 160)."""
        if not self.enabled or self.model is None:
            return {"is_owner": False, "label": "UNKNOWN", "distance": 999.0}
        with torch.no_grad():
            emb = self.model(face_tensor.unsqueeze(0)).cpu().numpy()[0]
        emb = l2_normalize(emb)
        return self._score(emb)

    def recognize(self, face_rgb):
        """Recognize from a raw RGB crop (MTCNN will align internally)."""
        if not self.enabled or self.model is None:
            return {"is_owner": False, "label": "UNKNOWN", "distance": 999.0}
        pil_img = Image.fromarray(face_rgb)
        face_tensor = self.mtcnn(pil_img)
        if face_tensor is None:
            return {"is_owner": False, "label": "UNKNOWN", "distance": 999.0}
        if face_tensor.ndim == 4:
            face_tensor = face_tensor[0]
        with torch.no_grad():
            emb = self.model(face_tensor.unsqueeze(0)).cpu().numpy()[0]
        emb = l2_normalize(emb)
        return self._score(emb)

    def _score(self, emb):
        mean_dist = cosine_distance(emb, self.mean_embedding)
        sample_dists = [cosine_distance(emb, ref) for ref in self.owner_embeddings]
        best_sample_dist = min(sample_dists) if sample_dists else 999.0
        is_owner = (mean_dist < self.threshold) and (best_sample_dist < self.threshold)
        return {
            "is_owner": is_owner,
            "label": "OWNER" if is_owner else "UNKNOWN",
            "distance": float(mean_dist),
        }
