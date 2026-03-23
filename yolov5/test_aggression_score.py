import cv2
import time

from aggression_analyzer import AggressionAnalyzer
from camera_utils import LatestFrameCamera


CAMERA_INDEX = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
CAMERA_FPS = 20

ANALYZE_EVERY = 4
DISPLAY_NAME = "Aggression Score Test"


def draw_multiline_text(frame, lines, x=10, y=25, line_h=22, color=(255, 255, 255)):
    yy = y
    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )
        yy += line_h


def main():
    analyzer = AggressionAnalyzer()

    cam = LatestFrameCamera(
        src=CAMERA_INDEX,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        fps=CAMERA_FPS
    ).start()

    print("Press ESC to quit")

    frame_idx = 0
    prev_time = None

    last_result = {
        "face_score": 0.0,
        "hand_score": 0.0,
        "pose_score": 0.0,
        "motion_score": 0.0,
        "aggression_score": 0.0,
        "state": "CALM",
        "debug": {}
    }

    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.005)
            continue

        frame_idx += 1

        now = time.time()
        if prev_time is None:
            dt = 1.0 / 30.0
        else:
            dt = max(1e-6, now - prev_time)
        prev_time = now

        if frame_idx % ANALYZE_EVERY == 0:
            timestamp_ms = frame_idx * 33
            try:
                analysis_frame = frame.copy()
                last_result = analyzer.analyze(analysis_frame, timestamp_ms)
            except Exception as e:
                print(f"Analyze error: {e}")

        aggression_score = float(last_result.get("aggression_score", 0.0))
        face_score = float(last_result.get("face_score", 0.0))
        hand_score = float(last_result.get("hand_score", 0.0))
        pose_score = float(last_result.get("pose_score", 0.0))
        motion_score = float(last_result.get("motion_score", 0.0))
        state = last_result.get("state", "CALM")
        debug = last_result.get("debug", {})

        if state == "AGGRESSIVE":
            color = (0, 0, 255)
        elif state == "WATCH":
            color = (0, 165, 255)
        else:
            color = (0, 255, 0)

        fps = 1.0 / dt if dt > 1e-6 else 0.0

        lines = [
            f"STATE: {state}",
            f"aggression_score={aggression_score:.3f}",
            "",
            "--- COMPONENTS ---",
            f"face_score={face_score:.3f}",
            f"hand_score={hand_score:.3f}",
            f"pose_score={pose_score:.3f}",
            f"motion_score={motion_score:.3f}",
            "",
            "--- RAW ---",
            f"face_raw={float(debug.get('face_raw', 0.0)):.3f}",
            f"hand_raw={float(debug.get('hand_raw', 0.0)):.3f}",
            f"pose_raw={float(debug.get('pose_raw', 0.0)):.3f}",
            f"motion_raw={float(debug.get('motion_raw', 0.0)):.3f}",
            "",
            "--- FACE ---",
            f"brow_down={float(debug.get('brow_down', 0.0)):.3f}",
            f"eye_squint={float(debug.get('eye_squint', 0.0)):.3f}",
            f"smile={float(debug.get('smile', 0.0)):.3f}",
            "",
            "--- HANDS ---",
            f"fist_count={int(debug.get('fist_count', 0))}",
            "",
            "--- POSE ---",
            f"forward_lean={float(debug.get('forward_lean', 0.0)):.3f}",
            "",
            f"display_fps={fps:.1f}",
            f"analyze_every={ANALYZE_EVERY}",
        ]

        cv2.putText(
            frame,
            state,
            (10, FRAME_HEIGHT - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            3
        )

        draw_multiline_text(frame, lines, x=10, y=25, line_h=20, color=(255, 255, 255))

        cv2.imshow(DISPLAY_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
