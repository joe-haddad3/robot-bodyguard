"""
Face Enrollment Flask Server — runs on Raspberry Pi.

Receives enrollment requests from the mobile app, processes images through
MTCNN + FaceNet, and stores per-person embeddings.

Endpoints:
    POST   /api/v1/enroll              Enroll or re-enroll a person
    GET    /api/v1/enrolled            List all enrolled persons
    DELETE /api/v1/enroll/<person_id>  Remove a person from the database
    GET    /api/v1/health              Liveness check

Start:
    python enrollment_server.py [--host 0.0.0.0] [--port 5000]

Database layout after enrollment:
    face_db/
        people.json
        embeddings/
            owner_1.npy
            family_1.npy
            ...
"""

import argparse
import base64
import io
import json
import logging
import os
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request, Response
import shared_state
from PIL import Image

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_BASE           = os.path.dirname(os.path.abspath(__file__))
_DB_DIR         = os.path.join(_BASE, "face_db")
_EMBEDDINGS_DIR = os.path.join(_DB_DIR, "embeddings")
_REGISTRY_PATH  = os.path.join(_DB_DIR, "people.json")
_ONNX_PATH      = os.path.join(_DB_DIR, "facenet.onnx")

os.makedirs(_DB_DIR, exist_ok=True)
os.makedirs(_EMBEDDINGS_DIR, exist_ok=True)

# Lazy-loaded singletons (avoids 15-20 s cold startup on every restart)
_mtcnn:      object | None = None
_recognizer: object | None = None

MIN_IMAGES   = 10
MAX_IMAGES   = 100
MIN_VALID    = 5
BLUR_THRESH  = 80.0   # Laplacian variance threshold


# ------------------------------------------------------------------
# Lazy model loaders
# ------------------------------------------------------------------

def _get_mtcnn():
    global _mtcnn
    if _mtcnn is None:
        from facenet_pytorch import MTCNN
        _mtcnn = MTCNN(
            image_size=160,
            margin=20,
            keep_all=False,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            post_process=True,
            device="cpu",
        )
        log.info("MTCNN loaded.")
    return _mtcnn


def _get_recognizer():
    global _recognizer
    if _recognizer is None:
        from multi_person_recognizer import MultiPersonRecognizer
        _recognizer = MultiPersonRecognizer()
    return _recognizer


# ------------------------------------------------------------------
# Registry helpers
# ------------------------------------------------------------------

def _load_registry() -> list[dict]:
    if not os.path.exists(_REGISTRY_PATH):
        return []
    with open(_REGISTRY_PATH, "r") as f:
        return json.load(f)


def _save_registry(registry: list[dict]):
    with open(_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def _upsert_person(
    registry: list[dict],
    person_id: str,
    name: str,
    role: str,
) -> list[dict]:
    """Insert or update a person entry in the registry list."""
    entry = {
        "person_id":      person_id,
        "name":           name,
        "role":           role,
        "embedding_file": f"embeddings/{person_id}.npy",
        "enrolled_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for i, p in enumerate(registry):
        if p["person_id"] == person_id:
            registry[i] = entry
            return registry
    registry.append(entry)
    return registry


# ------------------------------------------------------------------
# Image quality helpers
# ------------------------------------------------------------------

def _decode_b64_image(b64_str: str) -> Image.Image | None:
    """Decode a base64 string (with optional data-URI prefix) → PIL RGB image."""
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        raw = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        log.debug(f"Image decode failed: {exc}")
        return None


def _is_blurry(pil_img: Image.Image) -> bool:
    """Laplacian variance blur check. Returns True when image is too blurry."""
    gray    = np.array(pil_img.convert("L"), dtype=np.uint8)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(lap_var) < BLUR_THRESH


def _face_too_small(pil_img: Image.Image, min_fraction: float = 0.10) -> bool:
    """True if the image is too small to contain a usable face."""
    w, h    = pil_img.size
    min_dim = min(w, h)
    return min_dim < 80 or min_dim < (max(w, h) * min_fraction)


# ------------------------------------------------------------------
# Core embedding pipeline
# ------------------------------------------------------------------

def _embed_images(images_b64: list[str]) -> tuple[list[np.ndarray], dict]:
    """
    Decode → quality-check → detect face → embed each image.

    Returns:
        embeddings: list of L2-normalised (512,) float32 arrays
        stats:      {"total", "decode_err", "blurry", "no_face", "valid"}
    """
    import torch
    from facenet_pytorch import InceptionResnetV1

    mtcnn      = _get_mtcnn()
    stats      = {"total": len(images_b64), "decode_err": 0,
                  "blurry": 0, "no_face": 0, "valid": 0}
    embeddings = []

    # Build embedding model (ONNX preferred)
    session    = None
    pt_model   = None
    input_name = "input"

    try:
        import onnxruntime as ort
        if not os.path.exists(_ONNX_PATH):
            log.info("Exporting FaceNet to ONNX (one-time ~20 s) ...")
            tmp = InceptionResnetV1(pretrained="vggface2").eval()
            dummy = torch.zeros(1, 3, 160, 160)
            torch.onnx.export(
                tmp, dummy, _ONNX_PATH,
                input_names=["input"], output_names=["output"],
                opset_version=11, do_constant_folding=True,
            )
        session    = ort.InferenceSession(_ONNX_PATH, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        log.info("Embedding via ONNX Runtime.")
    except ImportError:
        pt_model = InceptionResnetV1(pretrained="vggface2").eval()
        log.info("Embedding via PyTorch.")

    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-10 else v

    for idx, b64 in enumerate(images_b64):
        img = _decode_b64_image(b64)
        if img is None:
            stats["decode_err"] += 1
            continue

        if _is_blurry(img):
            stats["blurry"] += 1
            log.debug(f"  img[{idx}]: blurry → skipped")
            continue

        face_tensor = mtcnn(img)
        if face_tensor is None:
            stats["no_face"] += 1
            log.debug(f"  img[{idx}]: no face → skipped")
            continue

        # Multiple faces: mtcnn(keep_all=False) already picks the dominant one,
        # but if for some reason tensor has unexpected shape, skip.
        if face_tensor.ndim != 3:
            stats["no_face"] += 1
            continue

        if session is not None:
            face_np = face_tensor.unsqueeze(0).numpy()
            emb     = session.run(None, {input_name: face_np})[0][0]
        else:
            with torch.no_grad():
                emb = pt_model(face_tensor.unsqueeze(0)).cpu().numpy()[0]

        embeddings.append(_normalize(emb))
        stats["valid"] += 1

    return embeddings, stats


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.post("/api/v1/enroll")
def enroll_person():
    """
    Enroll or re-enroll a person.

    Request JSON body:
    {
        "person_id": "family_1",          // unique identifier
        "name":      "Sara",              // display name
        "role":      "family_member",     // "owner" | "family_member"
        "images":    ["<base64>", ...]    // 10–100 JPEG/PNG images
    }

    Response:
    {
        "success":           true,
        "person_id":         "family_1",
        "name":              "Sara",
        "role":              "family_member",
        "samples_processed": 42,
        "stats":             { "total": 48, "blurry": 3, "no_face": 3, "valid": 42 },
        "message":           "Successfully enrolled Sara with 42 face samples."
    }
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"success": False, "message": "Request body must be JSON."}), 400

    person_id = str(body.get("person_id", "")).strip()
    name      = str(body.get("name", "")).strip()
    role      = str(body.get("role", "")).strip()
    images    = body.get("images", [])

    # ---- Validation ----
    errors = []
    if not person_id:
        errors.append("person_id is required.")
    if not name:
        errors.append("name is required.")
    if role not in ("owner", "family_member"):
        errors.append("role must be 'owner' or 'family_member'.")
    if not isinstance(images, list) or len(images) == 0:
        errors.append("images list is required.")
    elif len(images) < MIN_IMAGES:
        errors.append(f"Too few images ({len(images)}). Minimum {MIN_IMAGES} required.")
    elif len(images) > MAX_IMAGES:
        errors.append(f"Too many images ({len(images)}). Maximum {MAX_IMAGES} accepted.")
    # Reject IDs with path-traversal characters
    if person_id and any(c in person_id for c in ("/", "\\", "..", "\x00")):
        errors.append("person_id contains invalid characters.")

    if errors:
        return jsonify({"success": False, "message": " ".join(errors)}), 422

    log.info(
        f"Enrollment request: person_id={person_id!r}  name={name!r}  "
        f"role={role!r}  images={len(images)}"
    )

    # ---- Check disk space (rough: 1 MB headroom per person) ----
    try:
        stat = os.statvfs(_DB_DIR)
        free_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
        if free_mb < 50:
            return jsonify({
                "success": False,
                "message": f"Insufficient disk space ({free_mb:.0f} MB free). Need 50 MB.",
            }), 507
    except AttributeError:
        pass  # os.statvfs not available on Windows dev machine

    # ---- Process images ----
    try:
        embeddings, stats = _embed_images(images)
    except Exception as exc:
        log.error(f"Embedding pipeline error: {exc}", exc_info=True)
        return jsonify({"success": False, "message": f"Processing error: {exc}"}), 500

    log.info(
        f"  Processing done: valid={stats['valid']}  blurry={stats['blurry']}  "
        f"no_face={stats['no_face']}  decode_err={stats['decode_err']}"
    )

    if stats["valid"] < MIN_VALID:
        return jsonify({
            "success": False,
            "message": (
                f"Only {stats['valid']} usable face images out of {stats['total']}. "
                f"Blurry: {stats['blurry']}, No face detected: {stats['no_face']}. "
                f"Please re-capture with better lighting and ensure your face is visible."
            ),
            "stats": stats,
        }), 422

    # ---- Persist embeddings ----
    emb_array = np.array(embeddings, dtype=np.float32)   # (N, 512)
    emb_path  = os.path.join(_EMBEDDINGS_DIR, f"{person_id}.npy")

    # If re-enrolling, append to existing embeddings (keeps history, improves accuracy)
    if os.path.exists(emb_path):
        existing  = np.load(emb_path)
        emb_array = np.concatenate([existing, emb_array], axis=0)
        log.info(
            f"  Re-enrollment: appended {stats['valid']} samples "
            f"(total now: {len(emb_array)})"
        )

    np.save(emb_path, emb_array)

    # ---- Update registry ----
    registry = _load_registry()
    registry = _upsert_person(registry, person_id, name, role)
    _save_registry(registry)

    # ---- Reload live recognizer if running ----
    if _recognizer is not None:
        _recognizer.reload_database()

    log.info(f"Enrolled '{person_id}' ({name}) — {stats['valid']} new samples.")

    return jsonify({
        "success":           True,
        "person_id":         person_id,
        "name":              name,
        "role":              role,
        "samples_processed": stats["valid"],
        "total_samples":     int(len(emb_array)),
        "stats":             stats,
        "message":           f"Successfully enrolled {name} with {stats['valid']} face samples.",
    }), 200


@app.get("/api/v1/enrolled")
def list_enrolled():
    """Return all enrolled persons with their sample counts."""
    registry = _load_registry()
    result   = []

    for person in registry:
        pid       = person["person_id"]
        emb_path  = os.path.join(_EMBEDDINGS_DIR, f"{pid}.npy")
        count = 0
        if os.path.exists(emb_path):
            try:
                count = int(np.load(emb_path).shape[0])
            except Exception:
                pass
        result.append({
            "person_id":    pid,
            "name":         person["name"],
            "role":         person["role"],
            "sample_count": count,
            "enrolled_at":  person.get("enrolled_at"),
        })

    return jsonify(result), 200


@app.delete("/api/v1/enroll/<person_id>")
def delete_person(person_id: str):
    """Remove a person and their embeddings permanently."""
    if any(c in person_id for c in ("/", "\\", "..", "\x00")):
        return jsonify({"success": False, "message": "Invalid person_id."}), 400

    registry = _load_registry()
    before   = len(registry)
    registry = [p for p in registry if p["person_id"] != person_id]

    if len(registry) == before:
        return jsonify({
            "success": False,
            "message": f"person_id '{person_id}' not found.",
        }), 404

    _save_registry(registry)

    emb_path = os.path.join(_EMBEDDINGS_DIR, f"{person_id}.npy")
    if os.path.exists(emb_path):
        os.remove(emb_path)

    if _recognizer is not None:
        _recognizer.reload_database()

    log.info(f"Deleted person: {person_id}")
    return jsonify({"success": True, "message": f"Deleted '{person_id}'"}), 200


@app.get("/api/v1/health")
def health():
    registry = _load_registry()
    return jsonify({
        "status":          "ok",
        "timestamp":       time.time(),
        "enrolled_count":  len(registry),
    }), 200


# ------------------------------------------------------------------
# Detection / alert endpoints (data written by threat_detection_test.py)
# ------------------------------------------------------------------

@app.get("/api/v1/status")
def get_status():
    """Current detection state: mode, owner_present, alert_count, running."""
    return jsonify(shared_state.get_status()), 200


@app.post("/api/v1/start")
def start_detection():
    """Tell the detection loop to start (camera + YOLO begin)."""
    shared_state.set_running(True)
    log.info("Detection started by app.")
    return jsonify({"success": True, "status": "running"}), 200


@app.post("/api/v1/stop")
def stop_detection():
    """Tell the detection loop to pause (camera keeps open, processing stops)."""
    shared_state.set_running(False)
    log.info("Detection stopped by app.")
    return jsonify({"success": True, "status": "stopped"}), 200


@app.get("/api/v1/alerts")
def get_alerts():
    """Recent detection events (newest first).  ?limit=N (default 20, max 50)."""
    try:
        limit = min(50, max(1, int(request.args.get("limit", 20))))
    except (TypeError, ValueError):
        limit = 20
    return jsonify(shared_state.get_alerts(limit)), 200


@app.post("/api/v1/mode")
def set_mode():
    """Switch security mode.  Body: {"mode": "HOME" | "AWAY"}"""
    body = request.get_json(silent=True) or {}
    mode = str(body.get("mode", "HOME")).upper()
    if mode not in ("HOME", "AWAY"):
        return jsonify({"success": False, "message": "mode must be HOME or AWAY"}), 400
    shared_state.set_mode(mode)
    log.info(f"Security mode set to {mode}")
    return jsonify({"success": True, "mode": mode}), 200


@app.get("/api/v1/stream")
def mjpeg_stream():
    """MJPEG live stream of the annotated detection frame."""
    def generate():
        while True:
            frame = shared_state.get_frame()
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
            time.sleep(0.1)   # ~10 fps max to the app

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Bodyguard — Face Enrollment Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Bind port (default: 5000)")
    args = parser.parse_args()

    log.info(f"Starting enrollment server on http://{args.host}:{args.port}")
    log.info(f"Database directory: {_DB_DIR}")
    app.run(host=args.host, port=args.port, debug=False, threaded=False)
