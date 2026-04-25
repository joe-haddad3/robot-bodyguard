"""
Step-by-step ESP32 / UGV base diagnostic.
Run this ALONE (stop threat_detection_test.py first so no other process holds the ports).
"""
import serial
import serial.tools.list_ports
import time
import json
import sys

# ── Step 1: list every serial port the OS can see ─────────────────────────────
print("=" * 60)
print("STEP 1 — all serial ports visible to the OS")
print("=" * 60)
ports = list(serial.tools.list_ports.comports())
if not ports:
    print("  (none found — check USB cables)")
else:
    for p in ports:
        print(f"  {p.device:20s}  hwid={p.hwid}  desc={p.description}")

# ── Step 2: passive listen on every ttyUSB* port ──────────────────────────────
# ESP32 typically broadcasts a status JSON every second at startup.
# We listen for 3 s on each port without sending anything.
import glob
candidates = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
print(f"\n{'='*60}")
print(f"STEP 2 — passive listen (3 s each): {candidates}")
print(f"{'='*60}")

for port in candidates:
    print(f"\n  >> Listening on {port} @ 115200 ...")
    try:
        s = serial.Serial(port, 115200, timeout=0.5)
        time.sleep(0.5)
        deadline = time.time() + 3.0
        got_any = False
        while time.time() < deadline:
            if s.in_waiting:
                raw = s.read(s.in_waiting)
                try:
                    txt = raw.decode("utf-8", errors="replace").strip()
                except Exception:
                    txt = repr(raw)
                print(f"     RAW: {txt}")
                got_any = True
            time.sleep(0.1)
        if not got_any:
            print(f"     (silence — nothing received)")
        s.close()
    except Exception as e:
        print(f"     OPEN ERROR: {e}")

# ── Step 3: probe each port with {"T":130} and wait for reply ─────────────────
print(f"\n{'='*60}")
print(f"STEP 3 — send {{\"T\":130}} to each port and wait for reply")
print(f"{'='*60}")

found_port = None
for port in candidates:
    print(f"\n  >> Probing {port} ...")
    try:
        s = serial.Serial(port, 115200, timeout=1.0)
        time.sleep(1.0)
        msg = json.dumps({"T": 130}) + "\n"
        s.write(msg.encode())
        s.flush()
        print(f"     SENT: {msg.strip()}")
        deadline = time.time() + 2.0
        got_any = False
        buf = ""
        while time.time() < deadline:
            if s.in_waiting:
                buf += s.read(s.in_waiting).decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if line:
                        print(f"     RECV: {line}")
                        got_any = True
            time.sleep(0.05)
        if got_any:
            print(f"  *** FOUND ESP32 on {port} ***")
            found_port = port
        else:
            print(f"     (no reply)")
        s.close()
    except Exception as e:
        print(f"     ERROR: {e}")

# ── Step 4: if we found the port, send a short turn command ───────────────────
print(f"\n{'='*60}")
if found_port:
    print(f"STEP 4 — turning test on {found_port}")
    print(f"{'='*60}")
    s = serial.Serial(found_port, 115200, timeout=1.0)
    time.sleep(1.0)

    def send(cmd):
        msg = json.dumps(cmd) + "\n"
        s.write(msg.encode())
        s.flush()
        print(f"  SENT: {msg.strip()}")

    send({"T": 136, "cmd": 100000})   # disable heartbeat
    time.sleep(0.3)
    print("  Turning LEFT at speed 0.8 for 3 s (re-sending every 200 ms)...")
    deadline = time.time() + 3.0
    while time.time() < deadline:
        send({"T": 1, "L": -0.8, "R": 0.8})
        time.sleep(0.2)
    send({"T": 1, "L": 0, "R": 0})
    print("  Stopped.")
    s.close()
else:
    print("STEP 4 — SKIPPED (no ESP32 found on any port)")
    print(f"{'='*60}")
    print("Possible causes:")
    print("  1. ESP32 not powered or USB cable is charge-only (no data lines)")
    print("  2. Wrong baud rate (try 9600 or 57600)")
    print("  3. ESP32 firmware crashed / not flashed")
    print("  4. Another process (threat_detection_test.py) is holding the port")
