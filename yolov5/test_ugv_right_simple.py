import json
import time

import serial


PORT = "/dev/ttyUSB0"
BAUD = 115200


def send(ser, cmd):
    msg = json.dumps(cmd) + "\n"
    ser.write(msg.encode("utf-8"))
    ser.flush()
    print("SENT:", msg.strip())


def main():
    print(f"Opening {PORT} at {BAUD}...")
    ser = serial.Serial(PORT, BAUD, timeout=1.0)
    time.sleep(1.5)

    try:
        # Keep the base from auto-stopping while we test.
        send(ser, {"T": 136, "cmd": 100000})
        time.sleep(0.2)

        # Turn right in place.
        send(ser, {"T": 1, "L": 0.4, "R": -0.4})
        time.sleep(1.0)

        # Stop.
        send(ser, {"T": 1, "L": 0, "R": 0})
        print("Done.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
