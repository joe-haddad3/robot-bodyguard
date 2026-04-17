# The camera backend: make_camera(...) chooses which camera wrapper to use, 
# and LatestFrameCamera is the OpenCV threaded fallback that always keeps the newest frame.
import sys
import cv2
import threading # Instead of waiting for the camera every time you need a frame, the camera keeps reading frames in the background all the time. That reduces lag.


def _open_capture(src): # Goal: Try V4L2 backend first on Linux (Raspberry Pi), fall back to default.
    """Try V4L2 backend first on Linux (Raspberry Pi), fall back to default."""
    if sys.platform.startswith("linux"):  #checks if you're on linux
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
    """
    Threaded OpenCV camera wrapper that always keeps only the newest frame.

    Optimized for the project's USB webcam setup:
      - OpenCV/V4L2 path
      - MJPG request for webcams like the Logitech C270
      - buffer size 1 to reduce latency
      - background reader thread so the main loop sees the newest frame
    """

    def __init__(self, src=0, width=640, height=480, fps=15):
        self.cap = _open_capture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {src}")

        # Logitech USB webcams like the C270 often need MJPG to reliably honor 720p30.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
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

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        print(f"[Camera] OpenCV webcam ready: {actual_w}x{actual_h} @ {actual_fps:.1f} fps")
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
    """Return the project's webcam-optimized latest-frame OpenCV camera."""
    return LatestFrameCamera(src=src, width=width, height=height, fps=fps)
