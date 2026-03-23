import serial
import time

port = "/dev/ttyAMA0"   # Pi 4 usually /dev/serial0
# For Pi 5, Waveshare's example uses /dev/ttyAMA0
baud = 115200

ser = serial.Serial(port, baudrate=baud, timeout=1)
time.sleep(2)

def send_json(s):
    ser.write((s + "\n").encode("utf-8"))
    time.sleep(0.3)
    while ser.in_waiting:
        print(ser.readline().decode("utf-8", errors="ignore").strip())

# 1) turn on echo
send_json('{"T":1,"L":0.2,"R":0.2}')

# 2) harmless test on OLED


# 3) restore normal OLED screen if you want
# send_json('{"T":-3}')

# 4) movement test
# send_json('{"T":1,"L":0.2,"R":0.2}')
# time.sleep(2)
# send_json('{"T":1,"L":0,"R":0}')

ser.close()
