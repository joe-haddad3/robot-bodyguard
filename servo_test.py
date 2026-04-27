"""
Standalone servo test — no project dependencies.
Run: python servo_test.py
Press Ctrl+C to exit early.
"""
import serial
import time

PORT = "/dev/ttyUSB0"
BAUD = 115200

def send(ser, msg):
    ser.write((msg + "\n").encode())
    ser.flush()
    print(f"  >> {msg}")
    time.sleep(0.05)

def read_responses(ser, duration=0.3):
    deadline = time.time() + duration
    while time.time() < deadline:
        if ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                print(f"  << {line}")

print(f"Connecting to {PORT} at {BAUD} baud...")
try:
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
except Exception as e:
    print(f"ERROR: could not open {PORT} — {e}")
    raise SystemExit(1)

time.sleep(2.0)  # wait for Arduino to boot
print("Connected.\n")

steps = [
    ("Center",        "CENTER"),
    ("Pan left",      "PT:40,90"),
    ("Pan right",     "PT:140,90"),
    ("Center",        "CENTER"),
    ("Tilt up",       "PT:90,65"),
    ("Tilt down",     "PT:90,115"),
    ("Center",        "CENTER"),
    ("Scan on",       "SCAN:1"),
    ("(scanning 3s)", None),
    ("Scan off",      "SCAN:0"),
    ("Center",        "CENTER"),
]

try:
    for label, cmd in steps:
        print(label)
        if cmd is not None:
            send(ser, cmd)
        read_responses(ser, duration=3.0 if label == "(scanning 3s)" else 0.4)
        if cmd not in (None, "SCAN:1"):
            time.sleep(1.0)
except KeyboardInterrupt:
    print("\nInterrupted — centering and closing.")
    send(ser, "CENTER")

ser.close()
print("Done.")
