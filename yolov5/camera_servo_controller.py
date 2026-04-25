import serial
import time


class CameraServoController:
    def __init__(
		self,
        port="/dev/ttyUSB0",
        baud=115200
    ):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        time.sleep(2.0)

        self.pan = 90
        self.tilt = 90

        self.pan_min = 10
        self.pan_max = 170
        self.tilt_min = 60
        self.tilt_max = 120

        self.scan_on = False

    def _send(self, msg):
        full = msg + "\n"
        self.ser.write(full.encode())
        self.ser.flush()
        print("SEND:", msg)

    def start_scan(self):
        if not self.scan_on:
            self._send("SCAN:1")
            self.scan_on = True

    def stop_scan(self):
        if self.scan_on:
            self._send("SCAN:0")
            self.scan_on = False

    def center(self):
        self._send("CENTER")
        self.pan = 90
        self.tilt = 90

    def track_target(self, cx, cy, frame_w, frame_h):
        self.stop_scan()

        center_x = frame_w // 2
        center_y = frame_h // 2

        err_x = cx - center_x
        err_y = cy - center_y

        if err_x > 30:
            self.pan -= 2
        elif err_x < -30:
            self.pan += 2

        if err_y > 25:
            self.tilt += 2
        elif err_y < -25:
            self.tilt -= 2

        self.pan = max(self.pan_min, min(self.pan_max, self.pan))
        self.tilt = max(self.tilt_min, min(self.tilt_max, self.tilt))

        self._send(f"PT:{self.pan},{self.tilt}")

    def snap_to_target(self, cx, cy, frame_w, frame_h):
        """
        Snap to an absolute estimated angle for the target, ignoring self.pan/tilt.
        Use when transitioning from scan to tracking so the camera goes directly to
        the threat rather than jumping to 90° (the stale Python state after scanning).
        """
        self.stop_scan()
        err_x = cx - frame_w // 2
        err_y = cy - frame_h // 2
        pan_half  = self.pan_max - 90   # 80°
        tilt_half = self.tilt_max - 90  # 30°
        self.pan  = 90 - int(err_x / (frame_w / 2) * pan_half)
        self.tilt = 90 + int(err_y / (frame_h / 2) * tilt_half)
        self.pan  = max(self.pan_min,  min(self.pan_max,  self.pan))
        self.tilt = max(self.tilt_min, min(self.tilt_max, self.tilt))
        self._send(f"PT:{self.pan},{self.tilt}")

    def read_position_updates(self):
        """Drain any POS:pan=X,tilt=Y lines the Arduino sent (e.g. during scan)
        and keep self.pan / self.tilt in sync with hardware."""
        try:
            while self.ser.in_waiting > 0:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if line.startswith("POS:"):
                    for part in line[4:].split(","):
                        if part.startswith("pan="):
                            self.pan = int(part[4:])
                        elif part.startswith("tilt="):
                            self.tilt = int(part[5:])
        except Exception:
            pass

    def close(self):
        self.ser.close()
