import serial
import time
import json


class UGVBaseController:
    def __init__(self, port="/dev/ttyUSB1", baud=115200):
        self.ser = serial.Serial(port, baud, timeout=0.5)
        time.sleep(1.0)
        # Disable ESP32 heartbeat so motors aren't killed between our commands
        self._send({"T": 136, "cmd": 100000})

    def _send(self, cmd_dict):
        msg = json.dumps(cmd_dict) + "\n"
        self.ser.write(msg.encode())
        self.ser.flush()

    def turn(self, direction, speed=0.4):
        """Turn in place. direction: 'left' or 'right'. speed: 0.0–1.0."""
        if direction == "right":
            self._send({"T": 1, "L": speed, "R": -speed})
        else:
            self._send({"T": 1, "L": -speed, "R": speed})

    def turn_timed(self, direction, duration, speed=0.4):
        """Turn in place for `duration` seconds, re-sending command every 200ms to keep heartbeat alive."""
        deadline = time.time() + duration
        if direction == "right":
            cmd = {"T": 1, "L": speed, "R": -speed}
        else:
            cmd = {"T": 1, "L": -speed, "R": speed}
        while time.time() < deadline:
            self._send(cmd)
            remaining = deadline - time.time()
            time.sleep(min(0.2, remaining) if remaining > 0 else 0)
        self.stop()

    def stop(self):
        self._send({"T": 1, "L": 0, "R": 0})

    def request_feedback(self):
        """Ask the ESP32 to send back base-info JSON — used as a 'done' signal."""
        self._send({"T": 130})

    def read_response(self, timeout=2.0):
        """Block until any JSON line arrives from the ESP32, or timeout expires."""
        deadline = time.time() + timeout
        buf = ""
        while time.time() < deadline:
            if self.ser.in_waiting > 0:
                buf += self.ser.read(self.ser.in_waiting).decode("utf-8", errors="ignore")
                if "\n" in buf:
                    return True
            time.sleep(0.01)
        return False

    def close(self):
        self.ser.close()
