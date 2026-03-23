import os
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1


def l2_normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-10:
        return v
    return v / n


def cosine_distance(a, b):
    return 1.0 - float(np.dot(a, b))


class FaceNetOwnerRecognizer:
    def __init__(self, db_path="face_db/owner_embeddings.npy", threshold=0.90):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")

        self.owner_embeddings = np.load(db_path)
        self.threshold = threshold
        self.model = InceptionResnetV1(pretrained="vggface2").eval()

    def embed_face_rgb(self, face_rgb):
        face_img = Image.fromarray(face_rgb).resize((160, 160))
        face_np = np.asarray(face_img).astype(np.float32) / 255.0
        face_np = (face_np - 0.5) / 0.5
        face_tensor = torch.tensor(face_np).permute(2, 0, 1).unsqueeze(0).float()

        with torch.no_grad():
            emb = self.model(face_tensor).cpu().numpy()[0]

        return l2_normalize(emb)

    def recognize(self, face_rgb):
        emb = self.embed_face_rgb(face_rgb)
        dists = [cosine_distance(emb, ref) for ref in self.owner_embeddings]
        best_dist = min(dists) if dists else 999.0
        is_owner = best_dist < self.threshold

        return {
            "is_owner": is_owner,
            "label": "OWNER" if is_owner else "UNKNOWN",
            "distance": float(best_dist),
        }
