"""
Thread-safe shared state between the detection loop and the Flask API server.

The detection loop writes to this module; Flask endpoints read from it.
Both run in the same Python process (Flask in a daemon thread).
"""
import threading
import time
from collections import deque

_lock          = threading.Lock()
_alerts        = deque(maxlen=50)
_mode          = "HOME"    # "HOME" | "AWAY"  (set by app)
_owner_present = False
_latest_jpeg   = b""       # most-recent annotated frame as JPEG bytes
_running       = False     # detection loop active only after app sends /start


# ── Writers (called from the detection loop) ──────────────────────────────────

def add_alert(level: str, person_id: str, name: str,
              explanation: list, track_id: int) -> None:
    """Push a new detection event into the alert queue (newest first)."""
    with _lock:
        _alerts.appendleft({
            "id":          str(int(time.time() * 1000)),
            "level":       level,           # "THREAT" | "SUSPICIOUS"
            "person_id":   person_id,
            "name":        name,
            "explanation": explanation,
            "track_id":    track_id,
            "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })


def set_owner_present(flag: bool) -> None:
    global _owner_present
    with _lock:
        _owner_present = bool(flag)


def set_running(flag: bool) -> None:
    global _running
    with _lock:
        _running = bool(flag)


def is_running() -> bool:
    with _lock:
        return _running


def set_mode(m: str) -> None:
    global _mode
    with _lock:
        _mode = m.upper()


def update_frame(jpeg_bytes: bytes) -> None:
    global _latest_jpeg
    with _lock:
        _latest_jpeg = jpeg_bytes


# ── Readers (called from Flask request threads) ────────────────────────────────

def get_status() -> dict:
    with _lock:
        return {
            "mode":          _mode,
            "owner_present": _owner_present,
            "alert_count":   len(_alerts),
            "running":       _running,
        }


def get_alerts(limit: int = 20) -> list:
    with _lock:
        return list(_alerts)[:limit]


def get_frame() -> bytes:
    with _lock:
        return _latest_jpeg


def get_mode() -> str:
    with _lock:
        return _mode
