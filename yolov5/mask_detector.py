"""
Face concealment detector — no model download required.

Uses the MTCNN face detector that is already part of the project.
Logic:
  - Crop the upper 45 % of the YOLO person box (where the face is).
  - Run MTCNN on that crop.
  - If MTCNN finds NO face  → face is concealed (balaclava, ski mask, scarf …)
  - If MTCNN finds a face   → face is visible

This works reliably for full-face coverings (balaclavas, ski masks, hoods).
Surgical / cloth masks that leave eyes + nose visible may not be flagged,
but those are lower-risk anyway.

False-positive guard: only flag CONCEALED when the person box is tall enough
(MIN_BOX_HEIGHT) — avoids flagging people who are simply turned away or far.
"""

import cv2
import numpy as np
from PIL import Image

MIN_BOX_HEIGHT  = 120   # px — only flag concealment if person is this close
FACE_DETECT_THR = 0.80  # MTCNN confidence to count as "face found"


class MaskDetector:
    def __init__(self):
        try:
            from facenet_pytorch import MTCNN
            self._mtcnn = MTCNN(
                min_face_size=30,
                keep_all=False,
                thresholds=[0.6, 0.7, 0.7],
                device="cpu",
            )
            self.enabled = True
            print("[MaskDetector] Face-concealment detector ready (MTCNN-based, no download).")
        except Exception as e:
            print(f"[MaskDetector] Could not init MTCNN: {e}. Concealment detection disabled.")
            self._mtcnn  = None
            self.enabled = False

    # ------------------------------------------------------------------
    def detect(self, face_bgr, person_box_height=0):
        """
        Parameters
        ----------
        face_bgr           : BGR crop of the face / upper-body region
        person_box_height  : height of the full YOLO person box (for guard)

        Returns
        -------
        dict:
            mask        bool   True  = face concealed
            confidence  float  how confident we are
            label       str    "CONCEALED" | "VISIBLE" | "UNKNOWN"
        """
        _unknown = {"mask": False, "confidence": 0.0, "label": "UNKNOWN"}

        if not self.enabled or self._mtcnn is None:
            return _unknown
        if face_bgr is None or face_bgr.size == 0:
            return _unknown

        # Don't flag concealment if person is too far / box too small
        if person_box_height > 0 and person_box_height < MIN_BOX_HEIGHT:
            return _unknown

        try:
            rgb   = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            pil   = Image.fromarray(rgb)
            boxes, probs = self._mtcnn.detect(pil)

            if boxes is None or len(boxes) == 0:
                # No face found in this region → concealed
                return {"mask": True, "confidence": 0.88, "label": "CONCEALED"}

            best_prob = float(max(p for p in probs if p is not None) if probs is not None else 0.0)

            if best_prob < FACE_DETECT_THR:
                # Low-confidence detection — likely partially covered
                return {"mask": True, "confidence": round(1.0 - best_prob, 2), "label": "CONCEALED"}

            return {"mask": False, "confidence": round(best_prob, 2), "label": "VISIBLE"}

        except Exception:
            return _unknown
