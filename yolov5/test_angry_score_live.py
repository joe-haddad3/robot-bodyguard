import time

import cv2

from aggression_analyzer import AggressionAnalyzer
from camera_utils import make_camera


WINDOW_NAME = "Angry Score Live"
CAM_W = 1280
CAM_H = 720
CAM_FPS = 30
FACE_MIN_SIZE = (60, 60)


def draw_lines(frame, lines, x=20, y=36, color=(255, 255, 255)):
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


def expand_face_box(face, frame_shape, pad_x_ratio=0.6, pad_y_ratio=0.5, body_down_ratio=2.0):
    x, y, w, h = face
    fh, fw = frame_shape[:2]
    pad_x = int(w * pad_x_ratio)
    pad_y = int(h * pad_y_ratio)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(fw - 1, x + w + pad_x)
    y2 = min(fh - 1, y + h + int(h * body_down_ratio))
    return [x1, y1, x2, y2]


def main():
    cam = make_camera(src=0, width=CAM_W, height=CAM_H, fps=CAM_FPS).start()
    analyzer = AggressionAnalyzer()
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    print("Controls:")
    print("  esc = quit")

    while True:
        frame = cam.read()
        if frame is None:
            continue

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=FACE_MIN_SIZE,
        )

        crop_box = None
        if len(faces) > 0:
            face = max(faces, key=lambda f: f[2] * f[3])
            crop_box = expand_face_box(face, frame.shape)
            fx, fy, fw, fh = face
            cv2.rectangle(display, (fx, fy), (fx + fw, fy + fh), (0, 255, 255), 1)

        if crop_box is None:
            fh, fw = frame.shape[:2]
            crop_box = [0, 0, fw - 1, fh - 1]

        x1, y1, x2, y2 = crop_box
        crop = frame[y1:y2, x1:x2]
        result = analyzer.analyze(crop, int(time.time() * 1000))
        debug = result.get("debug", {})

        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)

        lines = [
            "mode: face anger only",
            f"state: {result.get('state', 'CALM')}",
            f"raw angry_face_score: {debug.get('angry_face_score', 0.0):.3f}",
            f"smoothed face_score: {result.get('face_score', 0.0):.3f}",
            f"full aggression_score: {result.get('aggression_score', 0.0):.3f}",
            f"brow_down: {debug.get('brow_down', 0.0):.3f}",
            f"brow_up: {debug.get('brow_up', 0.0):.3f}",
            f"eye_squint: {debug.get('eye_squint', 0.0):.3f}",
            f"eye_wide: {debug.get('eye_wide', 0.0):.3f}",
            f"mouth_close: {debug.get('mouth_close', 0.0):.3f}",
            f"brow_squint_combo: {debug.get('brow_squint_combo', 0.0):.3f}",
            f"alert_combo: {debug.get('alert_combo', 0.0):.3f}",
            f"nose_sneer: {debug.get('nose_sneer', 0.0):.3f}",
            f"upper_lip_raise: {debug.get('upper_lip_raise', 0.0):.3f}",
            f"cheek_squint: {debug.get('cheek_squint', 0.0):.3f}",
            f"mouth_frown: {debug.get('mouth_frown', 0.0):.3f}",
            f"snarl_combo: {debug.get('snarl_combo', 0.0):.3f}",
            f"triggers: brow={debug.get('brow_trigger', False)} jaw={debug.get('jaw_trigger', False)} blueprint={debug.get('blueprint_trigger', False)} snarl={debug.get('snarl_trigger', False)} alert={debug.get('alert_trigger', False)}",
            f"fist_count: {debug.get('fist_count', 0)} hand_move: {debug.get('hand_movement_score', 0.0):.3f}",
        ]
        draw_lines(display, lines)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
