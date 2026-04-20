"""
Flask enrollment + control API for the AI Bodyguard robot.
Runs as a daemon thread inside threat_detection_test.py.

Endpoints:
    GET    /api/v1/health
    GET    /api/v1/status
    POST   /api/v1/start
    POST   /api/v1/stop
    POST   /api/v1/mode
    GET    /api/v1/alerts
    POST   /api/v1/enroll           enroll trusted member or threat
    GET    /api/v1/enrolled         list all enrolled persons
    DELETE /api/v1/enroll/<id>      delete person
    GET    /api/v1/threats          list threats
    DELETE /api/v1/threats/<id>     delete threat
"""

import base64
import io
import json
import os
import threading

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

import shared_state
from multi_person_recognizer import (
    MultiPersonRecognizer,
    _DB_DIR, _REGISTRY_PATH, _EMBEDDINGS_DIR,
)

app = Flask(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

MIN_IMAGES = 5   # minimum images required for trusted-member enrollment
MIN_VALID  = 1   # minimum valid face crops required

# ── Recognizer used for embedding during enrollment ────────────────────────────
# Loaded once at server startup; separate from the detection loop's recognizer.

_embed_recognizer: MultiPersonRecognizer | None = None
_embed_lock = threading.Lock()


def _get_embed_recognizer() -> MultiPersonRecognizer:
    global _embed_recognizer
    with _embed_lock:
        if _embed_recognizer is None:
            _embed_recognizer = MultiPersonRecognizer()
        return _embed_recognizer


# ── Live recognizer reference (injected by threat_detection_test.py) ──────────
# Used ONLY to call reload_database() so the detection loop sees new enrollments.

_live_recognizer: object | None = None
_live_lock = threading.Lock()


def set_live_recognizer(recognizer) -> None:
    global _live_recognizer
    with _live_lock:
        _live_recognizer = recognizer


def _reload_live_recognizer() -> None:
    with _live_lock:
        rec = _live_recognizer
    if rec is not None:
        rec.reload_database()


# ── MTCNN face detector (lazy-loaded) ─────────────────────────────────────────

_mtcnn = None
_mtcnn_lock = threading.Lock()


def _get_mtcnn():
    global _mtcnn
    with _mtcnn_lock:
        if _mtcnn is None:
            try:
                from facenet_pytorch import MTCNN
                _mtcnn = MTCNN(
                    image_size=160, margin=20,
                    keep_all=False, post_process=False,
                )
                print("[EnrollmentServer] MTCNN loaded.")
            except Exception as e:
                print(f"[EnrollmentServer] MTCNN unavailable ({e}), using Haar fallback.")
                _mtcnn = False
        return _mtcnn if _mtcnn is not False else None


# ── Image helpers ──────────────────────────────────────────────────────────────

def _decode_image(b64: str) -> np.ndarray | None:
    try:
        data = base64.b64decode(b64)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.asarray(img, dtype=np.uint8)
    except Exception:
        return None


def _is_blurry(img_rgb: np.ndarray, threshold: float = 100.0) -> bool:
    import cv2
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < threshold


def _detect_face(img_rgb: np.ndarray) -> np.ndarray | None:
    """Return a 160×160 RGB face crop, or None."""
    import cv2
    mtcnn = _get_mtcnn()

    if mtcnn is not None:
        try:
            face_tensor = mtcnn(Image.fromarray(img_rgb))
            if face_tensor is not None:
                return face_tensor.permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)
        except Exception:
            pass

    # Haar fallback
    haar = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(haar)
    gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
    if len(faces):
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        crop = img_rgb[fy:fy + fh, fx:fx + fw]
        return np.asarray(Image.fromarray(crop).resize((160, 160)), dtype=np.uint8)

    return None


def _embed_images(b64_images: list, skip_blur: bool = False) -> tuple:
    """
    Decode base64 images → valid 512-D embeddings.
    Returns (embeddings_list, stats_dict).
    """
    stats = {"valid": 0, "no_face": 0, "blurry": 0, "decode_error": 0}
    embeddings = []
    recognizer = _get_embed_recognizer()

    for b64 in b64_images:
        img = _decode_image(b64)
        if img is None:
            stats["decode_error"] += 1
            continue

        if not skip_blur and _is_blurry(img):
            stats["blurry"] += 1
            continue

        face = _detect_face(img)
        if face is None:
            stats["no_face"] += 1
            continue

        try:
            emb = recognizer.embed_face_rgb(face)
            embeddings.append(emb)
            stats["valid"] += 1
        except Exception:
            stats["no_face"] += 1

    return embeddings, stats


# ── Database helpers ───────────────────────────────────────────────────────────

def _load_registry() -> list:
    if not os.path.exists(_REGISTRY_PATH):
        return []
    with open(_REGISTRY_PATH, "r") as f:
        return json.load(f)


def _save_registry(registry: list) -> None:
    os.makedirs(_DB_DIR, exist_ok=True)
    with open(_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


# ── Control endpoints ──────────────────────────────────────────────────────────

@app.route("/api/v1/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/api/v1/status", methods=["GET"])
def status():
    return jsonify(shared_state.get_status()), 200


@app.route("/api/v1/start", methods=["POST"])
def start_detection():
    shared_state.set_running(True)
    return jsonify({"ok": True, "message": "Detection started"}), 200


@app.route("/api/v1/stop", methods=["POST"])
def stop_detection():
    shared_state.set_running(False)
    return jsonify({"ok": True, "message": "Detection stopped"}), 200


@app.route("/api/v1/mode", methods=["POST"])
def set_mode():
    data = request.get_json(force=True, silent=True) or {}
    mode = str(data.get("mode", "HOME")).upper()
    if mode not in ("HOME", "AWAY"):
        return jsonify({"error": "mode must be HOME or AWAY"}), 400
    shared_state.set_mode(mode)
    return jsonify({"ok": True, "mode": mode}), 200


@app.route("/api/v1/alerts", methods=["GET"])
def get_alerts():
    limit = int(request.args.get("limit", 20))
    return jsonify(shared_state.get_alerts(limit)), 200


# ── Enrollment endpoints ───────────────────────────────────────────────────────

@app.route("/api/v1/enroll", methods=["POST"])
def enroll_person():
    data      = request.get_json(force=True, silent=True) or {}
    person_id = str(data.get("person_id", "")).strip()
    name      = str(data.get("name", "")).strip()
    role      = str(data.get("role", "family_member")).strip().lower()
    images    = data.get("images", [])

    errors = []
    if not person_id:
        errors.append("person_id is required")
    if not name:
        errors.append("name is required")
    if not isinstance(images, list) or len(images) == 0:
        errors.append("images list is required")
    elif role != "threat" and len(images) < MIN_IMAGES:
        errors.append(
            f"At least {MIN_IMAGES} images required for trusted members (got {len(images)})"
        )
    if errors:
        return jsonify({"errors": errors}), 422

    embeddings, stats = _embed_images(images, skip_blur=(role == "threat"))
    min_valid = 1 if role == "threat" else MIN_VALID

    if stats["valid"] < min_valid:
        return jsonify({
            "errors": [
                f"Not enough valid face images "
                f"(valid={stats['valid']}, required={min_valid})"
            ],
            "stats": stats,
        }), 422

    os.makedirs(_EMBEDDINGS_DIR, exist_ok=True)
    np.save(os.path.join(_EMBEDDINGS_DIR, f"{person_id}.npy"), np.stack(embeddings))

    registry = _load_registry()
    registry = [p for p in registry if p["person_id"] != person_id]
    registry.append({"person_id": person_id, "name": name, "role": role})
    _save_registry(registry)

    _reload_live_recognizer()

    return jsonify({
        "ok":        True,
        "person_id": person_id,
        "name":      name,
        "role":      role,
        "stats":     stats,
    }), 200


@app.route("/api/v1/enrolled", methods=["GET"])
def list_enrolled():
    registry = _load_registry()
    return jsonify([
        {"person_id": p["person_id"], "name": p["name"], "role": p["role"]}
        for p in registry
    ]), 200


@app.route("/api/v1/enroll/<person_id>", methods=["DELETE"])
def delete_person(person_id: str):
    registry     = _load_registry()
    original_len = len(registry)
    registry     = [p for p in registry if p["person_id"] != person_id]

    if len(registry) == original_len:
        return jsonify({"error": f"Person '{person_id}' not found"}), 404

    _save_registry(registry)

    emb_path = os.path.join(_EMBEDDINGS_DIR, f"{person_id}.npy")
    if os.path.exists(emb_path):
        os.remove(emb_path)

    _reload_live_recognizer()
    return jsonify({"ok": True, "deleted": person_id}), 200


# ── Threat endpoints ───────────────────────────────────────────────────────────

@app.route("/api/v1/threats", methods=["GET"])
def list_threats():
    threats = [
        p for p in _load_registry()
        if p.get("role", "").lower() == "threat"
    ]
    return jsonify([
        {"threat_id": p["person_id"], "name": p["name"]}
        for p in threats
    ]), 200


@app.route("/api/v1/threats/<threat_id>", methods=["DELETE"])
def delete_threat(threat_id: str):
    registry     = _load_registry()
    original_len = len(registry)
    registry     = [
        p for p in registry
        if not (p["person_id"] == threat_id and p.get("role", "").lower() == "threat")
    ]

    if len(registry) == original_len:
        return jsonify({"error": f"Threat '{threat_id}' not found"}), 404

    _save_registry(registry)

    emb_path = os.path.join(_EMBEDDINGS_DIR, f"{threat_id}.npy")
    if os.path.exists(emb_path):
        os.remove(emb_path)

    _reload_live_recognizer()
    return jsonify({"ok": True, "deleted": threat_id}), 200


@app.route("/api/v1/threats", methods=["POST"])
def enroll_threat_legacy():
    """Legacy POST /api/v1/threats — kept for backwards compatibility."""
    data      = request.get_json(force=True, silent=True) or {}
    threat_id = str(data.get("threat_id", "")).strip()
    name      = str(data.get("name", "")).strip()
    images    = data.get("images", [])

    if not threat_id or not name:
        return jsonify({"error": "threat_id and name are required"}), 422

    embeddings, stats = _embed_images(images, skip_blur=True)

    if stats["valid"] < 1:
        return jsonify({"error": "No valid face images found", "stats": stats}), 422

    os.makedirs(_EMBEDDINGS_DIR, exist_ok=True)
    np.save(os.path.join(_EMBEDDINGS_DIR, f"{threat_id}.npy"), np.stack(embeddings))

    registry = _load_registry()
    registry = [p for p in registry if p["person_id"] != threat_id]
    registry.append({"person_id": threat_id, "name": name, "role": "threat"})
    _save_registry(registry)

    _reload_live_recognizer()
    return jsonify({"ok": True}), 200
