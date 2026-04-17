# It embeds faces with FaceNet and compares them to enrolled owner embeddings using cosine_distance

import os
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1

_BASE = os.path.dirname(os.path.abspath(__file__))
_DB_DIR = os.path.join(_BASE, "face_db")

_DEFAULT_DB_PATH   = os.path.join(_DB_DIR, "owner_embeddings.npy")
_DEFAULT_MEAN_PATH = os.path.join(_DB_DIR, "owner_mean_embedding.npy")
_ONNX_PATH         = os.path.join(_DB_DIR, "facenet.onnx")


# ------------------------------------------------------------------
# Math helpers
# ------------------------------------------------------------------

def l2_normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def cosine_distance(a, b):
    return 1.0 - float(np.dot(a, b))


# ------------------------------------------------------------------
# One-time ONNX export
# ------------------------------------------------------------------

def _export_onnx(onnx_path):
    """
    Export InceptionResnetV1 (VGGFace2 weights) to ONNX.
    Runs once and saves to face_db/facenet.onnx.
    Takes ~20 s on first call.
    """
    print("[FaceNetOwnerRecognizer] Exporting to ONNX — one-time, ~20 s …")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    pt_model = InceptionResnetV1(pretrained="vggface2").eval()
    dummy = torch.zeros(1, 3, 160, 160)
    torch.onnx.export(
        pt_model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=True,
    )
    print(f"[FaceNetOwnerRecognizer] ONNX saved → {onnx_path}")


# ------------------------------------------------------------------
# Recognizer
# ------------------------------------------------------------------

class FaceNetOwnerRecognizer:
    """
    Compares a live face crop against stored owner embeddings.

    Usage:
        recognizer = FaceNetOwnerRecognizer(threshold=0.30)
        result = recognizer.recognize(face_rgb_numpy)
        # result = {"is_owner": bool, "label": str, "distance": float}
    """

    recognizer_type = "facenet"  # distinguishes from CLIPOwnerRecognizer

    def __init__(self, db_path=None, threshold=0.30):
        if db_path is None:
            db_path = _DEFAULT_DB_PATH

        self.threshold = threshold
        self.enabled   = os.path.exists(db_path)
        self._session  = None   # onnxruntime session
        self._pt_model = None   # PyTorch fallback

        if not self.enabled:
            print(
                f"[FaceNetOwnerRecognizer] WARNING: no database at '{db_path}'.\n"
                "  Run enroll_owner.py first to enable owner recognition."
            )
            self.owner_embeddings = np.empty((0, 512), dtype=np.float32)
            self.mean_embedding   = None
            return

        # Load stored embeddings
        self.owner_embeddings = np.load(db_path)

        raw_mean = (
            np.load(_DEFAULT_MEAN_PATH)
            if os.path.exists(_DEFAULT_MEAN_PATH)
            else self.owner_embeddings.mean(axis=0)
        )
        self.mean_embedding = l2_normalize(raw_mean)

        # --- Try ONNX Runtime first ---
        try:
            import onnxruntime as ort

            if not os.path.exists(_ONNX_PATH):
                _export_onnx(_ONNX_PATH)

            self._session    = ort.InferenceSession(
                _ONNX_PATH,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name

            print(
                f"[FaceNetOwnerRecognizer] ONNX Runtime loaded "
                f"({len(self.owner_embeddings)} embeddings, threshold={threshold})"
            )

        except ImportError:
            # --- Fall back to PyTorch ---
            print(
                "[FaceNetOwnerRecognizer] onnxruntime not installed — using PyTorch (slower).\n"
                "  For RPi, strongly recommend: pip install onnxruntime"
            )
            self._pt_model = InceptionResnetV1(pretrained="vggface2").eval()
            print(
                f"[FaceNetOwnerRecognizer] PyTorch loaded "
                f"({len(self.owner_embeddings)} embeddings, threshold={threshold})"
            )

    # ------------------------------------------------------------------

    def embed_face_rgb(self, face_rgb):
        """
        Compute an L2-normalised 512-D embedding for a face crop.

        Args:
            face_rgb: H×W×3 numpy array, RGB, any size (will be resized to 160×160)

        Returns:
            numpy array (512,) float32
        """
        if not self.enabled:
            return np.zeros(512, dtype=np.float32)

        # Resize → normalise to [-1, 1]
        face_img = Image.fromarray(face_rgb).resize((160, 160))
        face_np  = np.asarray(face_img, dtype=np.float32) / 255.0
        face_np  = (face_np - 0.5) / 0.5                         # → [-1, 1]
        face_np  = face_np.transpose(2, 0, 1)[np.newaxis]        # → (1, 3, 160, 160)

        if self._session is not None:
            emb = self._session.run(None, {self._input_name: face_np})[0][0]
        else:
            with torch.no_grad():
                t   = torch.from_numpy(face_np)
                emb = self._pt_model(t).cpu().numpy()[0]

        return l2_normalize(emb)

    # ------------------------------------------------------------------

    def recognize(self, face_rgb):
        """
        Decide whether the face matches the enrolled owner.

        Both the mean embedding and best individual sample must be below
        the threshold — this dual check reduces false positives significantly.

        Returns:
            dict: {"is_owner": bool, "label": str, "distance": float}
        """
        if not self.enabled:
            return {"is_owner": False, "label": "UNKNOWN", "distance": 999.0}

        emb = self.embed_face_rgb(face_rgb)

        mean_dist        = cosine_distance(emb, self.mean_embedding)
        sample_dists     = [cosine_distance(emb, ref) for ref in self.owner_embeddings]
        best_sample_dist = min(sample_dists) if sample_dists else 999.0

        is_owner = (mean_dist < self.threshold) and (best_sample_dist < self.threshold)

        return {
            "is_owner": is_owner,
            "label":    "OWNER" if is_owner else "UNKNOWN",
            "distance": float(mean_dist),
        }

    def recognize_crop(self, bgr_crop, face_cascade=None, min_face_size=20, face_expand=0.15):
        """
        Unified interface: detect face in a BGR person crop, then run FaceNet.

        Strategy (same as run_owner_recognition_in_yolo_boxes):
          1. Haar cascade in upper 55 % of crop
          2. Fall back to whole crop if no face found

        Returns:
            dict: {"is_owner": bool, "label": str, "distance": float, "embedding": ndarray|None}
        """
        import cv2

        if not self.enabled:
            return {"is_owner": False, "label": "UNKNOWN", "distance": 999.0, "embedding": None}

        h_crop = bgr_crop.shape[0]
        upper  = bgr_crop[:max(1, int(h_crop * 0.55))]

        face_crop_rgb = None

        # Attempt 1: Haar cascade inside upper crop
        if face_cascade is not None:
            gray  = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(min_face_size, min_face_size))
            if len(faces) > 0:
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                px = int(fw * face_expand)
                py = int(fh * face_expand)
                fc = upper[
                    max(0, fy - py):min(upper.shape[0], fy + fh + py),
                    max(0, fx - px):min(upper.shape[1], fx + fw + px),
                ]
                if fc.size > 0 and min(fc.shape[:2]) >= min_face_size:
                    face_crop_rgb = cv2.cvtColor(fc, cv2.COLOR_BGR2RGB)

        # Attempt 2: full upper crop as face input
        if face_crop_rgb is None and min(upper.shape[:2]) >= min_face_size:
            face_crop_rgb = cv2.cvtColor(upper, cv2.COLOR_BGR2RGB)

        if face_crop_rgb is None:
            return {"is_owner": False, "label": "UNKNOWN", "distance": 999.0, "embedding": None}

        emb = self.embed_face_rgb(face_crop_rgb)

        mean_dist        = cosine_distance(emb, self.mean_embedding) if self.mean_embedding is not None else 999.0
        sample_dists     = [cosine_distance(emb, ref) for ref in self.owner_embeddings]
        best_sample_dist = min(sample_dists) if sample_dists else 999.0

        is_owner = (mean_dist < self.threshold) and (best_sample_dist < self.threshold)

        return {
            "is_owner": is_owner,
            "label":    "OWNER" if is_owner else "UNKNOWN",
            "distance": float(mean_dist),
            "embedding": emb,
        }
