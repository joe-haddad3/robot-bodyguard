import serial
import time
import json

PORT = "/dev/ttyUSB1"
BAUD = 115200

print(f"Connecting to {PORT} at {BAUD} baud...")
ser = serial.Serial(PORT, BAUD, timeout=1.0)
time.sleep(1.5)
print("Connected.")

def send(cmd):
    msg = json.dumps(cmd) + "\n"
    ser.write(msg.encode())
    ser.flush()
    print(f"  SENT: {msg.strip()}")

def drain(label, seconds=1.0):
    deadline = time.time() + seconds
    lines = []
    while time.time() < deadline:
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    lines.append(line)
                    print(f"  RECV [{label}]: {line}")
            except Exception as e:
                print(f"  READ ERROR: {e}")
        time.sleep(0.01)
    if not lines:
        print(f"  RECV [{label}]: (nothing)")
    return lines

print("\n--- Step 1: Disable heartbeat (100 s timeout) ---")
send({"T": 136, "cmd": 100000})
drain("heartbeat_ack", 1.0)

print("\n--- Step 2: Request base info (sanity check ESP32 is alive) ---")
send({"T": 130})
drain("base_info", 1.5)

print("\n--- Step 3: Turn LEFT for 2 seconds ---")
deadline = time.time() + 2.0
while time.time() < deadline:
    send({"T": 1, "L": -0.4, "R": 0.4})
    time.sleep(0.2)

print("\n--- Step 4: Stop ---")
send({"T": 1, "L": 0, "R": 0})
drain("after_stop", 0.5)

print("\n--- Step 5: Turn RIGHT for 2 seconds ---")
deadline = time.time() + 2.0
while time.time() < deadline:
    send({"T": 1, "L": 0.4, "R": -0.4})
    time.sleep(0.2)

print("\n--- Step 6: Stop ---")
send({"T": 1, "L": 0, "R": 0})
drain("after_stop2", 0.5)

print("\n--- Step 7: Higher speed test (0.8) — LEFT 1.5 s ---")
deadline = time.time() + 1.5
while time.time() < deadline:
    send({"T": 1, "L": -0.8, "R": 0.8})
    time.sleep(0.2)

send({"T": 1, "L": 0, "R": 0})
drain("after_highspeed", 0.5)

ser.close()
print("\nDone. Check if robot turned at any step.")
