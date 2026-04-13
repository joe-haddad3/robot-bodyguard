"""
CLIP-based Owner Recognizer with Gallery System

Uses OpenAI CLIP ViT-B/32 to generate whole-body visual embeddings that are
robust to pose, orientation, and clothing. Works with front, back, and side views.

Gallery-based matching: maintains multiple reference embeddings (front/back/side)
and matches incoming faces/bodies against the gallery using cosine similarity.

Advantages over FaceNet:
  - Works with any body part (not just faces)
  - Robust to pose/orientation changes
  - Side/back view recognition
  - Handles occlusions better

Install CLIP:
    pip install git+https://github.com/openai/CLIP.git
"""

import os
import numpy as np
import torch
from PIL import Image
import clip

_BASE = os.path.dirname(os.path.abspath(__file__))
_DB_DIR = os.path.join(_BASE, "face_db")

_CLIP_GALLERY_PATH = os.path.join(_DB_DIR, "clip_owner_gallery.npy")
_CLIP_METADATA_PATH = os.path.join(_DB_DIR, "clip_owner_metadata.txt")


def l2_normalize(v):
    """L2 normalize a vector."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def cosine_similarity(a, b):
    """Compute cosine similarity between two normalized vectors."""
    a_norm = l2_normalize(a)
    b_norm = l2_normalize(b)
    return float(np.dot(a_norm, b_norm))


class CLIPOwnerRecognizer:
    """
    Gallery-based owner recognition using CLIP embeddings.

    Key features:
      - Maintains a gallery of owner embeddings (front, back, side views)
      - Cosine similarity matching against gallery
      - Robust to pose/orientation changes
      - Single gallery threshold (e.g., 0.82)
    """

    recognizer_type = "clip"  # distinguishes from FaceNetOwnerRecognizer

    def __init__(self, threshold=0.82):
        """
        Initialize CLIP owner recognizer.
        
        Args:
            threshold: Cosine similarity threshold for owner match (0.0-1.0).
                      Default 0.82 is strict but reliable.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[CLIPOwnerRecognizer] Loading CLIP ViT-B/32 on {self.device}")
        
        try:
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        except Exception as e:
            print(f"[CLIPOwnerRecognizer] CLIP loading failed: {e}")
            print("[CLIPOwnerRecognizer] CLIP may not be installed. Install with:")
            print("  pip install git+https://github.com/openai/CLIP.git")
            self.enabled = False
            return
        
        self.enabled = True
        self.threshold = threshold
        self.owner_gallery = []  # List of 512-D numpy arrays
        
        # Load persisted gallery if it exists
        self._load_gallery()
    
    def _load_gallery(self):
        """Load owner gallery from disk if it exists."""
        if os.path.exists(_CLIP_GALLERY_PATH):
            try:
                gallery_data = np.load(_CLIP_GALLERY_PATH)  # shape (N, 512) float32
                if gallery_data.ndim == 1:
                    gallery_data = gallery_data[np.newaxis]  # single embedding edge case
                self.owner_gallery = [gallery_data[i] for i in range(len(gallery_data))]
                print(f"[CLIPOwnerRecognizer] Loaded gallery with {len(self.owner_gallery)} embeddings")
            except Exception as e:
                print(f"[CLIPOwnerRecognizer] Failed to load gallery: {e}")
                self.owner_gallery = []
    
    def _save_gallery(self):
        """Save owner gallery to disk."""
        try:
            os.makedirs(_DB_DIR, exist_ok=True)
            if not self.owner_gallery:
                return
            # Stack into (N, 512) float32 array — avoids pickle + object-dtype fragility
            gallery_arr = np.stack(self.owner_gallery).astype(np.float32)
            np.save(_CLIP_GALLERY_PATH, gallery_arr)
            print(f"[CLIPOwnerRecognizer] Saved gallery with {len(self.owner_gallery)} embeddings")
        except Exception as e:
            print(f"[CLIPOwnerRecognizer] Failed to save gallery: {e}")
    
    def embed_image(self, image_pil):
        """
        Generate CLIP embedding from a PIL image or numpy array.
        
        Args:
            image_pil: PIL Image or numpy BGR array
            
        Returns:
            512-D numpy array (L2-normalized) or None on error
        """
        if not self.enabled:
            return None
        
        try:
            # Convert numpy array to PIL if needed
            if isinstance(image_pil, np.ndarray):
                # Assume BGR format, convert to RGB
                image_pil = Image.fromarray(image_pil[..., ::-1])
            
            # Preprocess and embed
            image_tensor = self.preprocess(image_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.encode_image(image_tensor)
            
            # Normalize and convert to numpy
            embedding = embedding.cpu().numpy().flatten()
            embedding = l2_normalize(embedding)
            
            return embedding.astype(np.float32)
        
        except Exception as e:
            print(f"[CLIPOwnerRecognizer] Embedding failed: {e}")
            return None
    
    def add_to_gallery(self, embedding, label=""):
        """
        Add an embedding to the owner gallery.
        
        Args:
            embedding: 512-D numpy array
            label: Optional description (e.g., "front", "back", "side")
        """
        if not self.enabled or embedding is None:
            return
        
        embedding = l2_normalize(embedding).astype(np.float32)
        self.owner_gallery.append(embedding)
        print(f"[CLIPOwnerRecognizer] Added to gallery (now {len(self.owner_gallery)} items) [{label}]")
        self._save_gallery()
    
    def clear_gallery(self):
        """Clear all owner gallery embeddings."""
        self.owner_gallery = []
        if os.path.exists(_CLIP_GALLERY_PATH):
            os.remove(_CLIP_GALLERY_PATH)
        print("[CLIPOwnerRecognizer] Gallery cleared")
    
    def recognize(self, image_pil):
        """
        Recognize if an image contains the owner.
        
        Args:
            image_pil: PIL Image or numpy BGR array
            
        Returns:
            dict: {
                "is_owner": bool,
                "label": "OWNER" or "UNKNOWN",
                "similarity": float (max similarity to gallery),
                "distance": float (1 - similarity, for backwards compatibility)
            }
        """
        if not self.enabled or not self.owner_gallery:
            return {
                "is_owner": False,
                "label": "UNKNOWN",
                "similarity": 0.0,
                "distance": 1.0
            }
        
        embedding = self.embed_image(image_pil)
        if embedding is None:
            return {
                "is_owner": False,
                "label": "UNKNOWN",
                "similarity": 0.0,
                "distance": 1.0
            }
        
        # Compute similarities against all gallery embeddings
        similarities = [cosine_similarity(embedding, ref) for ref in self.owner_gallery]
        max_similarity = max(similarities) if similarities else 0.0
        
        is_owner = max_similarity >= self.threshold
        
        return {
            "is_owner": is_owner,
            "label": "OWNER" if is_owner else "UNKNOWN",
            "similarity": float(max_similarity),
            "distance": 1.0 - float(max_similarity)  # Backwards compatibility
        }
    
    def match_embedding(self, embedding):
        """
        Match a pre-computed embedding against the gallery.
        
        Args:
            embedding: 512-D numpy array
            
        Returns:
            dict: {"is_owner": bool, "similarity": float}
        """
        if not self.enabled or not self.owner_gallery:
            return {"is_owner": False, "similarity": 0.0}
        
        embedding = l2_normalize(embedding)
        similarities = [cosine_similarity(embedding, ref) for ref in self.owner_gallery]
        max_similarity = max(similarities) if similarities else 0.0
        
        return {
            "is_owner": max_similarity >= self.threshold,
            "similarity": float(max_similarity)
        }
    
    def get_gallery_size(self):
        """Return number of embeddings in gallery."""
        return len(self.owner_gallery)
    
    def recognize_crop(self, bgr_crop):
        """
        Recognize owner from a full-body BGR numpy crop.
        CLIP works on the whole body — no face required.

        Returns:
            dict: {"is_owner": bool, "label": str, "similarity": float, "distance": float,
                   "embedding": ndarray | None}
        """
        result = self.recognize(bgr_crop)
        emb = self.embed_image(bgr_crop) if result.get("is_owner") else None
        result["embedding"] = emb
        return result

    def set_threshold(self, threshold):
        """Update similarity threshold."""
        self.threshold = threshold
        print(f"[CLIPOwnerRecognizer] Threshold updated to {threshold:.2f}")
