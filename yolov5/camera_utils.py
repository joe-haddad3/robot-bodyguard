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
