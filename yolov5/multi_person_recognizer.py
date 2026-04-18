"""
Multi-person FaceNet recognizer.

Drop-in replacement for FaceNetOwnerRecognizer that supports multiple
enrolled persons (owner + family members).

Database layout:
    face_db/
        people.json          ← registry of all enrolled persons
        embeddings/
            owner_1.npy      ← per-person embedding matrices (N×512)
            family_1.npy
            ...

Usage:
    recognizer = MultiPersonRecognizer(threshold=0.32)
    result = recognizer.identify(face_rgb_numpy)
    # result = {"person_id": str, "name": str, "role": str,
    #            "distance": float, "matched": bool}

    # Legacy shim (backward compat with code expecting is_owner / label):
    result = recognizer.recognize(face_rgb_numpy)
    # result = {"is_owner": bool, "label": str, "distance": float, "person_id": str}
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1

_BASE           = os.path.dirname(os.path.abspath(__file__))
_DB_DIR         = os.path.join(_BASE, "face_db")
_REGISTRY_PATH  = os.path.join(_DB_DIR, "people.json")
_EMBEDDINGS_DIR = os.path.join(_DB_DIR, "embeddings")
_ONNX_PATH      = os.path.join(_DB_DIR, "facenet.onnx")


# ------------------------------------------------------------------
# Math helpers
# ------------------------------------------------------------------

def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - float(np.dot(a, b))


# ------------------------------------------------------------------
# ONNX export (one-time)
# ------------------------------------------------------------------

def _export_onnx(onnx_path: str):
    print("[MultiPersonRecognizer] Exporting FaceNet to ONNX — one-time, ~20 s ...")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    pt_model = InceptionResnetV1(pretrained="vggface2").eval()
    dummy = torch.zeros(1, 3, 160, 160)
    torch.onnx.export(
        pt_model, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        opset_version=11, do_constant_folding=True,
    )
    print(f"[MultiPersonRecognizer] ONNX saved → {onnx_path}")


# ------------------------------------------------------------------
# Recognizer
# ------------------------------------------------------------------

class MultiPersonRecognizer:
    """
    Identifies a face crop against all enrolled persons.

    Matching strategy (dual-gate, same principle as original FaceNetOwnerRecognizer):
        combined_score = (mean_embedding_distance + best_sample_distance) / 2
    A person matches only if combined_score < threshold.

    This is more robust than single-gate checks and reduces false positives.
    """

    UNKNOWN_RESULT = {
        "person_id": "UNKNOWN",
        "name":      "Unknown",
        "role":      "unknown",
        "distance":  999.0,
        "matched":   False,
    }

    def __init__(self, threshold: float = 0.32):
        self.threshold = threshold

        self._session:         object | None           = None   # onnxruntime.InferenceSession
        self._pt_model:        object | None           = None   # InceptionResnetV1
        self._input_name:      str                     = "input"

        self._registry:         list[dict]             = []
        self._embeddings:       dict[str, np.ndarray]  = {}   # person_id → (N, 512)
        self._mean_embeddings:  dict[str, np.ndarray]  = {}   # person_id → (512,)

        os.makedirs(_DB_DIR, exist_ok=True)
        os.makedirs(_EMBEDDINGS_DIR, exist_ok=True)

        self._load_model()
        self.reload_database()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        try:
            import onnxruntime as ort
            if not os.path.exists(_ONNX_PATH):
                _export_onnx(_ONNX_PATH)
            self._session    = ort.InferenceSession(
                _ONNX_PATH, providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            print("[MultiPersonRecognizer] ONNX Runtime loaded.")
        except ImportError:
            print(
                "[MultiPersonRecognizer] onnxruntime not installed — using PyTorch (slower).\n"
                "  pip install onnxruntime  for faster inference on RPi."
            )
            self._pt_model = InceptionResnetV1(pretrained="vggface2").eval()
            print("[MultiPersonRecognizer] PyTorch model loaded.")

    # ------------------------------------------------------------------
    # Database management
    # ------------------------------------------------------------------

    def reload_database(self):
        """Re-read people.json and all embedding .npy files from disk."""
        self._registry = []
        self._embeddings.clear()
        self._mean_embeddings.clear()

        if not os.path.exists(_REGISTRY_PATH):
            print("[MultiPersonRecognizer] No people.json — empty database.")
            return

        with open(_REGISTRY_PATH, "r") as f:
            self._registry = json.load(f)

        loaded = 0
        for person in self._registry:
            pid  = person["person_id"]
            path = os.path.join(_EMBEDDINGS_DIR, f"{pid}.npy")
            if os.path.exists(path):
                embs = np.load(path)                          # (N, 512)
                self._embeddings[pid]      = embs
                self._mean_embeddings[pid] = l2_normalize(embs.mean(axis=0))
                loaded += 1
            else:
                print(f"[MultiPersonRecognizer] WARNING: embedding file missing for '{pid}'")

        print(
            f"[MultiPersonRecognizer] Database loaded: "
            f"{loaded}/{len(self._registry)} persons, "
            f"threshold={self.threshold}"
        )

    def list_persons(self) -> list[dict]:
        """Return a copy of the registry list."""
        return list(self._registry)

    def is_enrolled(self, person_id: str) -> bool:
        return any(p["person_id"] == person_id for p in self._registry)

    # ------------------------------------------------------------------
    # Embedding inference
    # ------------------------------------------------------------------

    def embed_face_rgb(self, face_rgb: np.ndarray) -> np.ndarray:
        """
        Compute an L2-normalised 512-D embedding for a face crop.

        Args:
            face_rgb: H×W×3 uint8 numpy array (RGB), any size.
        Returns:
            numpy array shape (512,) float32.
        """
        face_img = Image.fromarray(face_rgb).resize((160, 160))
        face_np  = np.asarray(face_img, dtype=np.float32) / 255.0
        face_np  = (face_np - 0.5) / 0.5                          # → [-1, 1]
        face_np  = face_np.transpose(2, 0, 1)[np.newaxis]         # → (1, 3, 160, 160)

        if self._session is not None:
            emb = self._session.run(None, {self._input_name: face_np})[0][0]
        else:
            with torch.no_grad():
                t   = torch.from_numpy(face_np)
                emb = self._pt_model(t).cpu().numpy()[0]

        return l2_normalize(emb)

    # ------------------------------------------------------------------
    # Identification
    # ------------------------------------------------------------------

    def identify(self, face_rgb: np.ndarray) -> dict:
        """
        Identify the closest enrolled person from a face crop.

        Args:
            face_rgb: H×W×3 uint8 numpy array (RGB)

        Returns:
            dict with keys: person_id, name, role, distance, matched
        """
        if not self._registry:
            return self.UNKNOWN_RESULT.copy()

        emb          = self.embed_face_rgb(face_rgb)
        best_score   = float("inf")
        best_person  = None

        for person in self._registry:
            pid = person["person_id"]
            if pid not in self._mean_embeddings:
                continue

            mean_dist   = cosine_distance(emb, self._mean_embeddings[pid])
            sample_dist = min(
                cosine_distance(emb, ref) for ref in self._embeddings[pid]
            )
            # Dual-gate combined score
            combined = (mean_dist + sample_dist) / 2.0

            if combined < best_score:
                best_score  = combined
                best_person = person

        if best_person is None or best_score >= self.threshold:
            return self.UNKNOWN_RESULT.copy()

        return {
            "person_id": best_person["person_id"],
            "name":      best_person["name"],
            "role":      best_person["role"],
            "distance":  round(best_score, 4),
            "matched":   True,
        }

    def identify_from_embedding(self, emb: np.ndarray) -> dict:
        """
        Identify from a pre-computed L2-normalized 512-D embedding.
        Avoids re-running FaceNet when the embedding is already available.
        """
        if not self._registry:
            return self.UNKNOWN_RESULT.copy()

        best_score  = float("inf")
        best_person = None

        for person in self._registry:
            pid = person["person_id"]
            if pid not in self._mean_embeddings:
                continue

            mean_dist   = cosine_distance(emb, self._mean_embeddings[pid])
            sample_dist = min(
                cosine_distance(emb, ref) for ref in self._embeddings[pid]
            )
            combined = (mean_dist + sample_dist) / 2.0

            if combined < best_score:
                best_score  = combined
                best_person = person

        if best_person is None or best_score >= self.threshold:
            return self.UNKNOWN_RESULT.copy()

        return {
            "person_id": best_person["person_id"],
            "name":      best_person["name"],
            "role":      best_person["role"],
            "distance":  round(best_score, 4),
            "matched":   True,
        }

    def identify_crop(
        self,
        bgr_crop:       np.ndarray,
        face_cascade:   object | None = None,
        min_face_size:  int   = 20,
        face_expand:    float = 0.15,
    ) -> dict:
        """
        Detect face inside a BGR person crop, then identify.
        Drop-in replacement for FaceNetOwnerRecognizer.recognize_crop().

        Returns the same dict as identify() plus an 'embedding' key.
        """
        import cv2

        h_crop = bgr_crop.shape[0]
        upper  = bgr_crop[:max(1, int(h_crop * 0.55))]
        face_crop_rgb = None

        if face_cascade is not None:
            gray  = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, 1.1, 3, minSize=(min_face_size, min_face_size)
            )
            if len(faces):
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                px = int(fw * face_expand)
                py = int(fh * face_expand)
                fc = upper[
                    max(0, fy - py):min(upper.shape[0], fy + fh + py),
                    max(0, fx - px):min(upper.shape[1], fx + fw + px),
                ]
                if fc.size > 0 and min(fc.shape[:2]) >= min_face_size:
                    face_crop_rgb = cv2.cvtColor(fc, cv2.COLOR_BGR2RGB)

        if face_crop_rgb is None and min(upper.shape[:2]) >= min_face_size:
            face_crop_rgb = cv2.cvtColor(upper, cv2.COLOR_BGR2RGB)

        if face_crop_rgb is None:
            result = self.UNKNOWN_RESULT.copy()
            result["embedding"] = None
            return result

        emb    = self.embed_face_rgb(face_crop_rgb)
        result = self.identify_from_embedding(emb)
        result["embedding"] = emb
        return result

    # ------------------------------------------------------------------
    # Legacy backward-compatibility shim
    # ------------------------------------------------------------------

    def recognize(self, face_rgb: np.ndarray) -> dict:
        """
        Shim for code that still calls .recognize() expecting
        {is_owner, label, distance}.

        Maps multi-person result to the original FaceNetOwnerRecognizer shape.
        """
        result   = self.identify(face_rgb)
        is_owner = result["matched"] and result["role"] == "owner"
        return {
            "is_owner":  is_owner,
            "label":     result["name"] if result["matched"] else "UNKNOWN",
            "distance":  result["distance"],
            "person_id": result["person_id"],
        }

    def recognize_crop(self, bgr_crop, face_cascade=None, min_face_size=20, face_expand=0.15):
        """Shim: same signature as FaceNetOwnerRecognizer.recognize_crop()."""
        result   = self.identify_crop(bgr_crop, face_cascade, min_face_size, face_expand)
        is_owner = result["matched"] and result["role"] == "owner"
        return {
            "is_owner":  is_owner,
            "label":     result["name"] if result["matched"] else "UNKNOWN",
            "distance":  result["distance"],
            "person_id": result["person_id"],
            "role":      result.get("role", "unknown"),
            "embedding": result.get("embedding"),
        }
