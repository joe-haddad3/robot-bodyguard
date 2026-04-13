import sys
import cv2
import threading


def _open_capture(src):
    """Try V4L2 backend first on Linux (Raspberry Pi), fall back to default."""
    if sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(src)


class _Picamera2Camera:
    """
    Picamera2-backed camera for Raspberry Pi 5 CSI ribbon-cable sensors.
    Gives lower latency than V4L2 and direct numpy array output (no BGR→RGB needed).
    Only instantiated when picamera2 is importable; LatestFrameCamera is the fallback.
    """

    def __init__(self, width=640, height=480, fps=15):
        from picamera2 import Picamera2  # imported lazily so desktop builds don't need it
        self._picam  = Picamera2()
        cfg = self._picam.create_preview_configuration(
            main={"format": "BGR888", "size": (width, height)},
        )
        self._picam.configure(cfg)
        self.frame   = None
        self.lock    = threading.Lock()
        self.running = False
        self._thread = None

    def start(self):
        self._picam.start()
        self.running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        return self

    def _update(self):
        while self.running:
            arr = self._picam.capture_array()   # BGR888 — already in OpenCV format
            with self.lock:
                self.frame = arr

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._picam.stop()


class LatestFrameCamera:
    def __init__(self, src=0, width=640, height=480, fps=15):
        self.cap = _open_capture(src)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self._thread = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self.update, daemon=True)
        self._thread.start()
        return self

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()


def make_camera(src=0, width=640, height=480, fps=15):
    """
    Factory: return the best available camera for the current platform.

    On Raspberry Pi 5 with a CSI ribbon-cable sensor, Picamera2 is tried first
    (lower latency, direct BGR array output).  Falls back to LatestFrameCamera
    (V4L2 / OpenCV VideoCapture) on any failure or on non-Linux systems.
    """
    if sys.platform.startswith("linux"):
        try:
            cam = _Picamera2Camera(width=width, height=height, fps=fps)
            print("[Camera] Picamera2 available — using CSI camera path")
            return cam
        except Exception as exc:
            print(f"[Camera] Picamera2 unavailable ({exc}), falling back to V4L2/OpenCV")
    return LatestFrameCamera(src=src, width=width, height=height, fps=fps)
