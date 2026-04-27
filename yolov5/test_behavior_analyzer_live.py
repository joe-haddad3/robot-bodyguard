import cv2

from behavior_analyzer import (
    BehaviorAnalyzer,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
)
from camera_utils import make_camera


WINDOW_NAME = "BehaviorAnalyzer Live"
CAM_W = 1280
CAM_H = 720
CAM_FPS = 30
TRACK_ID = 1

KEY_LANDMARKS = [
    ("l_el", LEFT_ELBOW),
    ("r_el", RIGHT_ELBOW),
    ("l_wr", LEFT_WRIST),
    ("r_wr", RIGHT_WRIST),
]

FACE_ZONE_MARGIN = 25.0
FACE_ZONE_HEIGHT_RATIO = 0.40


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


def make_owner_box(frame_shape):
    h, w = frame_shape[:2]
    box_w = int(w * 0.22)
    box_h = int(h * 0.55)
    x1 = int(w * 0.08)
    y1 = int(h * 0.20)
    return [x1, y1, x1 + box_w, y1 + box_h]


def make_face_zone(owner_box):
    ox1, oy1, ox2, oy2 = [float(v) for v in owner_box]
    face_bottom = oy1 + (oy2 - oy1) * FACE_ZONE_HEIGHT_RATIO
    return [
        int(ox1 - FACE_ZONE_MARGIN),
        int(oy1 - FACE_ZONE_MARGIN),
        int(ox2 + FACE_ZONE_MARGIN),
        int(face_bottom + FACE_ZONE_MARGIN),
    ]


def landmark_hits_face_zone(points, face_zone):
    fx1, fy1, fx2, fy2 = face_zone
    hits = []
    for lm_idx in (LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST):
        if lm_idx not in points:
            continue
        gx, gy, vis, label = points[lm_idx]
        if vis < 0.30:
            continue
        if fx1 <= gx <= fx2 and fy1 <= gy <= fy2:
            hits.append(label)
    return hits


def draw_pose_overlay(frame, points, hits):
    if LEFT_ELBOW in points and LEFT_WRIST in points:
        x1, y1, _, _ = points[LEFT_ELBOW]
        x2, y2, _, _ = points[LEFT_WRIST]
        cv2.line(frame, (x1, y1), (x2, y2), (255, 120, 0), 2)
    if RIGHT_ELBOW in points and RIGHT_WRIST in points:
        x1, y1, _, _ = points[RIGHT_ELBOW]
        x2, y2, _, _ = points[RIGHT_WRIST]
        cv2.line(frame, (x1, y1), (x2, y2), (255, 120, 0), 2)

    for gx, gy, vis, label in points.values():
        if 0 <= gx < frame.shape[1] and 0 <= gy < frame.shape[0]:
            color = (0, 0, 255) if label in hits else (0, 255, 255)
            cv2.circle(frame, (gx, gy), 7, color, -1)
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
        owner_box = make_owner_box(display.shape)
        face_zone = make_face_zone(owner_box)

        if pose_bbox is not None:
            px1, py1, px2, py2 = [int(v) for v in pose_bbox]
            cv2.rectangle(display, (px1, py1), (px2, py2), (255, 0, 0), 1)

        points = to_global_points(landmarks, pose_bbox)
        hits = landmark_hits_face_zone(points, face_zone)
        threat_detected = bool(hits)

        ox1, oy1, ox2, oy2 = owner_box
        fx1, fy1, fx2, fy2 = face_zone
        cv2.rectangle(display, (ox1, oy1), (ox2, oy2), (80, 180, 255), 2)
        cv2.rectangle(display, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
        cv2.putText(
            display,
            "OWNER BOX",
            (ox1, max(20, oy1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (80, 180, 255),
            2,
        )
        cv2.putText(
            display,
            "FACE ZONE",
            (fx1, max(20, fy1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        if points:
            draw_pose_overlay(display, points, set(hits))

        header = [
            "mode: owner-face proximity test",
            f"result: {'THREAT' if threat_detected else 'SAFE'}",
            f"trigger: {', '.join(hits) if hits else 'None'}",
        ]

        result_color = (0, 0, 255) if threat_detected else (0, 255, 0)
        draw_lines(display, header, 20, 36, result_color)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        frame_index += 1

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
