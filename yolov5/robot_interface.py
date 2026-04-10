"""
RobotInterface — sends AI threat state to the robot hardware every frame.

JSON format sent to robot:
{
    "ts": 1700000000.123,
    "system": "SAFE",          # SAFE | SUSPICIOUS | THREAT
    "owner": {
        "detected": true,
        "cx": 320,             # pixel center X  (use for follow logic)
        "cy": 240,             # pixel center Y
        "bbox": [x1,y1,x2,y2]
    },
    "top_threat": {            # null if no threat
        "id": 2,
        "level": "THREAT",
        "score": 27.5,
        "cx": 200, "cy": 350,
        "bbox": [x1,y1,x2,y2],
        "weapons": ["knife"]
    },
    "frame_w": 640,
    "frame_h": 480
}

Output modes:
  "print"  — logs SUSPICIOUS/THREAT events to console (default / debug)
  "serial" — sends JSON line over UART  (Arduino / RPi UART)
  "udp"    — sends JSON over UDP socket (networked robot controller)
  "file"   — writes latest state to a JSON file  (poll from another process)
"""

import json
import time


LEVEL_RANK = {"SAFE": 0, "SUSPICIOUS": 1, "THREAT": 2}


class RobotInterface:
    """
    Serialises threat state and forwards it to the robot hardware.

    Parameters
    ----------
    mode         : "print" | "serial" | "udp" | "file"
    serial_port  : e.g. "/dev/ttyUSB0" or "COM3"
    serial_baud  : default 115200
    udp_host     : IP of robot controller
    udp_port     : UDP port
    file_path    : path written in "file" mode
    send_every_n : throttle — send at most every N frames
                   (THREAT / level-change always sent immediately)
    """

    def __init__(
        self,
        mode="print",
        serial_port=None,
        serial_baud=115200,
        udp_host="127.0.0.1",
        udp_port=5005,
        file_path="/tmp/robot_state.json",
        send_every_n=3,
    ):
        self.mode = mode
        self.send_every_n = send_every_n
        self._counter = 0
        self._last_level = "SAFE"
        self._ser = None
        self._sock = None
        self._file_path = file_path

        if mode == "serial":
            try:
                import serial
                self._ser = serial.Serial(
                    serial_port or "/dev/ttyUSB0", serial_baud, timeout=0.1
                )
                print(f"[RobotInterface] Serial → {serial_port} @ {serial_baud} baud")
            except Exception as e:
                print(f"[RobotInterface] Serial failed ({e}). Falling back to print.")
                self.mode = "print"

        elif mode == "udp":
            import socket
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._addr = (udp_host, udp_port)
            print(f"[RobotInterface] UDP → {udp_host}:{udp_port}")

        elif mode == "file":
            print(f"[RobotInterface] File → {file_path}")

        elif mode == "print":
            print("[RobotInterface] Console mode (SUSPICIOUS/THREAT events only)")

    # ------------------------------------------------------------------
    def update(self, state: dict):
        """Call once per frame with the current threat state dict."""
        self._counter += 1
        level = state.get("system", "SAFE")
        level_changed = level != self._last_level
        self._last_level = level

        # Send immediately on THREAT or level change; throttle otherwise
        if not (level_changed or level == "THREAT" or
                self._counter % self.send_every_n == 0):
            return

        msg = json.dumps(state, separators=(",", ":")) + "\n"

        if self.mode == "serial" and self._ser:
            try:
                self._ser.write(msg.encode("ascii"))
            except Exception as e:
                print(f"[RobotInterface] Serial write error: {e}")

        elif self.mode == "udp" and self._sock:
            try:
                self._sock.sendto(msg.encode("ascii"), self._addr)
            except Exception as e:
                print(f"[RobotInterface] UDP error: {e}")

        elif self.mode == "file":
            try:
                with open(self._file_path, "w") as f:
                    f.write(msg)
            except Exception as e:
                print(f"[RobotInterface] File error: {e}")

        elif self.mode == "print":
            if level != "SAFE":
                top = state.get("top_threat")
                owner = state.get("owner", {})
                ow = (f"owner@({owner['cx']},{owner['cy']})"
                      if owner.get("detected") else "owner:MISSING")
                th = ""
                if top:
                    w = top.get("weapons", [])
                    th = (f"  | THREAT id={top['id']} score={top['score']:.1f}"
                          f" pos=({top['cx']},{top['cy']})"
                          f" weapons={w or 'none'}")
                print(f"[ROBOT] {level} | {ow}{th}")

    # ------------------------------------------------------------------
    def close(self):
        if self._ser:
            self._ser.close()
        if self._sock:
            self._sock.close()
