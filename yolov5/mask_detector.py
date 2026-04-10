"""
Face mask detector — uses a pretrained image classifier from HuggingFace.

Model downloads automatically on first run (~100 MB, cached after that).

Requirements:
    pip install transformers

If transformers is not installed, mask detection is silently disabled
and the rest of the system continues working normally.
"""

import os
import cv2
import numpy as np
from PIL import Image

_MODEL_ID = "Bingsu/acanet-mask-detector"   # ViT-based, verified on HuggingFace

_pipe    = None
_enabled = False


def _load_model():
    global _pipe, _enabled
    try:
        from transformers import pipeline
        print("[MaskDetector] Loading model (first run downloads ~100 MB)...")
        _pipe = pipeline(
            "image-classification",
            model=_MODEL_ID,
            device=-1,      # CPU
            top_k=2,
        )
        _enabled = True
        print("[MaskDetector] Ready.")
    except ImportError:
        print("[MaskDetector] 'transformers' not installed.")
        print("  Run:  pip install transformers")
        print("  Mask detection will be disabled until then.")
    except Exception as e:
        print(f"[MaskDetector] Could not load model: {e}")
        print("  Mask detection disabled.")


class MaskDetector:
    """
    Detects whether a person is wearing a face mask.

    Usage:
        detector = MaskDetector()
        result   = detector.detect(face_bgr_crop)
        # result = {"mask": True/False, "confidence": 0.97, "label": "MASK"/"NO_MASK"}
    """

    def __init__(self):
        _load_model()
        self.enabled = _enabled

    def detect(self, face_bgr):
        """
        Parameters
        ----------
        face_bgr : np.ndarray  BGR image of the face / upper-body region

        Returns
        -------
        dict with keys:
            mask        : bool   — True if mask detected
            confidence  : float  — classifier confidence (0-1)
            label       : str    — "MASK", "NO_MASK", or "UNKNOWN"
        """
        _UNKNOWN = {"mask": False, "confidence": 0.0, "label": "UNKNOWN"}

        if not self.enabled or _pipe is None:
            return _UNKNOWN
        if face_bgr is None or face_bgr.size == 0:
            return _UNKNOWN

        try:
            rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            results = _pipe(pil)          # [{"label":..., "score":...}, ...]
            top     = results[0]
            label   = top["label"].lower()
            conf    = float(top["score"])

            # Label strings vary by model; handle both "mask"/"with_mask" etc.
            is_mask = ("mask" in label) and ("without" not in label) and ("no" not in label)

            return {
                "mask":       is_mask,
                "confidence": conf,
                "label":      "MASK" if is_mask else "NO_MASK",
            }
        except Exception as e:
            return _UNKNOWN
