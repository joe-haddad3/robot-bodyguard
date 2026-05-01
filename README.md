# Robot Bodyguard — AI-Powered Autonomous Security Robot

An intelligent real-time security system that uses computer vision and AI to detect threats, recognize the owner, identify weapons, and send protective commands to robot hardware.

---

## What It Does

The robot continuously watches its environment through a camera and classifies every person it sees into one of three threat levels:

| Level | Score | Meaning |
|---|---|---|
| `SAFE` | ≤ 5 | Owner or non-threatening person |
| `SUSPICIOUS` | 5 – 15 | Unknown person with concerning behavior |
| `THREAT` | > 15 | Armed or actively aggressive intruder |

When a threat is detected, the system sends a structured JSON command over Serial (UART), UDP, or a file pipe to the robot hardware, which then takes protective action.

---

## Key Features

- **Person & Weapon Detection** — YOLOv5 detects people, guns, knives, hammers, and baseball bats in real time
- **Multi-Person Tracking** — ByteTrack keeps a persistent identity for each person across frames, even through brief occlusions
- **Owner Recognition (Dual System)**
  - *CLIP gallery* — robust to pose changes and partial occlusion (cosine similarity ≥ 0.82)
  - *FaceNet* — face-crop embedding for precise identity matching (L2 distance ≤ 0.32)
- **Behavior Analysis** — MediaPipe pose landmarks detect punches, swings, lunges, arm extensions, and wrist-velocity spikes
- **Aggression Scoring** — Facial blendshapes (anger, tension) and hand gestures contribute to the per-person threat score
- **Face Concealment Detection** — MTCNN detects balaclavas, ski masks, and scarves
- **Camera Servo Control** — Pan/tilt servo follows the owner by default, switches to the top threat on detection
- **REST Enrollment API** — Flask server lets you enroll new faces over HTTP without stopping the system
- **Raspberry Pi 5 Optimized** — Automatic RPi detection, frame-skip scheduling, ONNX runtime, and lighter pose models

---

## Tech Stack

| Layer | Library |
|---|---|
| Object detection | YOLOv5 + PyTorch |
| Multi-object tracking | ByteTrack (via Ultralytics) |
| Pose & face landmarks | MediaPipe |
| Owner recognition | CLIP (OpenAI) + FaceNet (facenet-pytorch) |
| Face detection | MTCNN |
| Hardware control | PySerial (UART) / UDP sockets |
| Enrollment API | Flask |
| Language | Python 3.10+ |

---

## Project Structure

```
robot-bodyguard/
├── yolov5/
│   ├── threat_detection_test.py      # Main entry point — camera loop + threat pipeline
│   ├── enhanced_threat_analyzer.py   # Combines all signals into a single threat score
│   ├── behavior_analyzer.py          # Pose-based action detection (punch, swing, lunge…)
│   ├── aggression_analyzer.py        # Face expression + hand gesture scoring
│   ├── multi_person_recognizer.py    # FaceNet multi-person identification
│   ├── clip_owner_recognizer.py      # CLIP gallery-based owner recognition
│   ├── enhanced_tracking.py          # ByteTrack + Kalman motion prediction
│   ├── mask_detector.py              # Face concealment detection (MTCNN)
│   ├── robot_interface.py            # Serializes threat state → hardware (Serial/UDP/File)
│   ├── camera_servo_controller.py    # Pan/tilt servo control
│   ├── ugv_base_controller.py        # Robot base movement commands
│   ├── enrollment_server.py          # Flask REST API for live enrollment
│   ├── enroll_owner_clip.py          # CLI: enroll owner via CLIP (recommended)
│   ├── enroll_owner.py               # CLI: enroll owner via FaceNet (legacy)
│   ├── capture_enrollment_photos.py  # Capture raw enrollment photos
│   ├── shared_state.py               # Thread-safe global state
│   ├── camera_utils.py               # Camera init utilities
│   ├── run_test.bat                   # Windows launcher
│   └── test_*.py                     # Individual module test scripts
└── owner_faces/                       # Enrolled owner images (local, not committed)
```

---

## Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/joe-haddad3/robot-bodyguard.git
cd robot-bodyguard/yolov5

pip install -r requirements.txt
pip install -r requirements_project.txt

# Install CLIP
pip install git+https://github.com/openai/CLIP.git
```

> **Raspberry Pi 5:** Replace `opencv-python` with `opencv-python-headless` and set `rpi_mode=True` in the config.

### 2. Add the YOLO weights

Download your trained `best.pt` and place it in `yolov5/`.  
The model should be trained to detect: `person`, `gun`, `knife`, `hammer`, `baseball_bat`.

### 3. Enroll the owner

```bash
# Recommended: CLIP gallery (robust to pose/occlusion)
python enroll_owner_clip.py

# Legacy: FaceNet face-crop
python enroll_owner.py
```

Minimum images: **5 for FaceNet**, **3–5 views for CLIP** (front, back, side).

---

## Running

```bash
python threat_detection_test.py
```

Press **`q`** to quit.

On Windows you can also double-click `run_test.bat`.

---

## Threat Scoring

Each person gets a score every frame built from weighted signals:

| Signal | Weight |
|---|---|
| Owner recognized | −20 |
| Unknown person | +3 |
| Gun detected | +22 |
| Knife detected | +18 |
| Hammer detected | +12 |
| Baseball bat | +8 |
| Punch / swing / lunge | variable |
| Face concealed | +5 |
| Angry expression | variable |

Scores are smoothed over a rolling window of recent frames to avoid single-frame spikes triggering false alarms.

---

## Hardware Integration

`robot_interface.py` sends a JSON payload to the robot every N frames (configurable), and always immediately on a level change or `THREAT` event:

```json
{
  "timestamp": 1714567890.123,
  "system_level": "THREAT",
  "owner_location": {"track_id": 1, "bbox": [120, 80, 200, 400]},
  "top_threat": {"track_id": 3, "score": 27.4, "weapons": ["gun"]},
  "all_persons": [...]
}
```

**Output modes:** `serial` (UART to Arduino), `udp`, `file`, `console`

---

## REST Enrollment API

The enrollment server runs as a background thread inside the main process.

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/health` | GET | Health check |
| `/api/v1/status` | GET | Current system status |
| `/api/v1/enroll` | POST | Enroll a new person |
| `/api/v1/threats` | GET | Live threat feed |

---

## Test Scripts

```bash
python test_behavior_analyzer_live.py   # Pose-based action detection
python test_angry_score_live.py         # Facial expression scoring
python test_owner_recognition.py        # FaceNet owner matching
python test_clip_owner.py               # CLIP owner matching
python test_aggression.py               # Aggression pipeline
python test_mask_classifier.py          # Face concealment
```

---

## Performance

| Platform | YOLO | Pose (MediaPipe) | CLIP | FaceNet |
|---|---|---|---|---|
| PC (GPU) | ~10 ms | ~30 ms | ~50 ms | ~20 ms |
| PC (CPU) | ~100 ms | ~200 ms | ~200 ms | ~50 ms |
| Raspberry Pi 5 | ~300 ms* | ~400 ms* | N/A | ~200 ms* |

\* With frame-skip (every 3–4 frames) and ONNX runtime enabled.

---

## License

This project builds on [YOLOv5](https://github.com/ultralytics/yolov5) (GPL-3.0).  
All custom modules in this repository are released under the **MIT License**.
