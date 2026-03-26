import os
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1

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
    def __init__(self, db_path=None, threshold=0.90):
        if db_path is None:
            db_path = _DEFAULT_DB_PATH
        self.threshold = threshold
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

        # Load mean embedding if it exists, otherwise compute it
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

    def embed_face_rgb(self, face_rgb):
        if not self.enabled:
            return np.zeros(512, dtype=np.float32)
        face_img = Image.fromarray(face_rgb).resize((160, 160))
        face_np = np.asarray(face_img).astype(np.float32) / 255.0
        face_np = (face_np - 0.5) / 0.5
        face_tensor = torch.tensor(face_np).permute(2, 0, 1).unsqueeze(0).float()

        with torch.no_grad():
            emb = self.model(face_tensor).cpu().numpy()[0]

        return l2_normalize(emb)

    def recognize(self, face_rgb):
        if not self.enabled:
            return {"is_owner": False, "label": "UNKNOWN", "distance": 999.0}

        emb = self.embed_face_rgb(face_rgb)

        # Primary: distance to mean embedding (stable, fast)
        mean_dist = cosine_distance(emb, self.mean_embedding)

        # Secondary: best match across all stored samples
        sample_dists = [cosine_distance(emb, ref) for ref in self.owner_embeddings]
        best_sample_dist = min(sample_dists) if sample_dists else 999.0

        # Both must pass — mean_dist is the reported distance
        is_owner = (mean_dist < self.threshold) and (best_sample_dist < self.threshold)

        return {
            "is_owner": is_owner,
            "label": "OWNER" if is_owner else "UNKNOWN",
            "distance": float(mean_dist),
        }
