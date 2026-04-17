import cv2

from behavior_analyzer import (
    BehaviorAnalyzer,
    NOSE,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
)
from camera_utils import make_camera


WINDOW_NAME = "BehaviorAnalyzer Live"
CAM_W = 1280
CAM_H = 720
CAM_FPS = 30
TRACK_ID = 1

KEY_LANDMARKS = [
    ("nose", NOSE),
    ("l_sh", LEFT_SHOULDER),
    ("r_sh", RIGHT_SHOULDER),
    ("l_el", LEFT_ELBOW),
    ("r_el", RIGHT_ELBOW),
    ("l_wr", LEFT_WRIST),
    ("r_wr", RIGHT_WRIST),
    ("l_hp", LEFT_HIP),
    ("r_hp", RIGHT_HIP),
    ("l_kn", LEFT_KNEE),
    ("r_kn", RIGHT_KNEE),
]

POSE_CONNECTIONS = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (RIGHT_HIP, RIGHT_KNEE),
]

DEBUG_SIGNALS = [
    "raised_arms",
    "single_arm_high",
    "arm_pose",
    "forward_arm",
    "wrist_fast",
    "swinging_arm",
    "torso_lean",
    "crouch",
    "head_lowered",
    "strike_windup",
    "punch_detected",
    "attack_motion",
    "lunge",
    "running",
]


def draw_lines(frame, lines, x, y, color):
    line_h = 28
    for i, text in enumerate(lines):
        yy = y + i * line_h
        if 15 <= yy < frame.shape[0] - 5:
            cv2.putText(
                frame,
                text,
                (x, yy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 0, 0),
                4,
            )
            cv2.putText(
                frame,
                text,
                (x, yy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                color,
                2,
            )


def to_global_points(landmarks, pose_bbox):
    if landmarks is None or pose_bbox is None:
        return {}

    px1, py1, px2, py2 = pose_bbox
    pw = max(1.0, float(px2 - px1))
    ph = max(1.0, float(py2 - py1))

    points = {}
    for label, lm_idx in KEY_LANDMARKS:
        if lm_idx >= len(landmarks):
            continue
        lm = landmarks[lm_idx]
        gx = int(px1 + lm.x * pw)
        gy = int(py1 + lm.y * ph)
        vis = float(getattr(lm, "visibility", 1.0))
        points[lm_idx] = (gx, gy, vis, label)
    return points


def draw_pose_overlay(frame, points):
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx in points and end_idx in points:
            x1, y1, _, _ = points[start_idx]
            x2, y2, _, _ = points[end_idx]
            cv2.line(frame, (x1, y1), (x2, y2), (255, 120, 0), 2)

    for gx, gy, vis, label in points.values():
        if 0 <= gx < frame.shape[1] and 0 <= gy < frame.shape[0]:
            cv2.circle(frame, (gx, gy), 5, (0, 255, 255), -1)
            cv2.putText(
                frame,
                label,
                (gx + 8, gy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )


def main():
    cam = make_camera(src=0, width=CAM_W, height=CAM_H, fps=CAM_FPS).start()
    analyzer = BehaviorAnalyzer(
        pose_every_n=1,
        velocity_window=10,
        min_crop_size=80,
    )

    print("Controls:")
    print("  esc = quit")

    frame_index = 0

    while True:
        frame = cam.read()
        if frame is None:
            continue

        fh, fw = frame.shape[:2]
        full_frame_box = [0, 0, fw - 1, fh - 1]

        score, behaviors = analyzer.analyze_person(
            frame,
            full_frame_box,
            track_id=TRACK_ID,
            frame_index=frame_index,
            run_pose=True,
        )

        display = frame.copy()
        pose_bbox = analyzer.get_pose_bbox(TRACK_ID)
        landmarks = analyzer.get_pose_landmarks(TRACK_ID)
        stable_signals = analyzer.get_pose_signals(TRACK_ID, stabilized=True)
        raw_signals = analyzer.get_pose_signals(TRACK_ID, stabilized=False)

        if pose_bbox is not None:
            px1, py1, px2, py2 = [int(v) for v in pose_bbox]
            cv2.rectangle(display, (px1, py1), (px2, py2), (255, 0, 0), 1)

        points = to_global_points(landmarks, pose_bbox)
        if points:
            draw_pose_overlay(display, points)

        key_lines = []
        if points:
            for gx, gy, vis, label in points.values():
                key_lines.append(f"{label}: ({gx},{gy}) v={vis:.2f}")
        else:
            key_lines.append("pose: no landmarks detected")

        signal_lines = [
            "stable: " + " ".join(f"{name}={stable_signals[name]}" for name in DEBUG_SIGNALS[:4]),
            "stable: " + " ".join(f"{name}={stable_signals[name]}" for name in DEBUG_SIGNALS[4:8]),
            "stable: " + " ".join(f"{name}={stable_signals[name]}" for name in DEBUG_SIGNALS[8:]),
            "raw: " + " ".join(f"{name}={raw_signals[name]}" for name in DEBUG_SIGNALS[:4]),
            "raw: " + " ".join(f"{name}={raw_signals[name]}" for name in DEBUG_SIGNALS[4:8]),
            "raw: " + " ".join(f"{name}={raw_signals[name]}" for name in DEBUG_SIGNALS[8:]),
        ]

        header = [
            "mode: full-frame pose only",
            f"behaviors: {', '.join(behaviors) if behaviors else 'None'}",
            f"score: {score}",
        ]

        draw_lines(display, header + signal_lines + key_lines, 20, 36, (255, 255, 255))

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        frame_index += 1

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
