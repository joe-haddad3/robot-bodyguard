import cv2
import time

from owner_recognizer_facenet import FaceNetOwnerRecognizer


CAMERA_INDEX = 0
FACE_MIN_SIZE = (60, 60)
OWNER_DISTANCE_THRESHOLD = 0.30


def draw_label(frame, text, x, y, color=(0, 255, 0), scale=0.6, thickness=2):
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness
    )


def main():
    recognizer = FaceNetOwnerRecognizer(threshold=OWNER_DISTANCE_THRESHOLD)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return

    print("Press ESC to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read frame")
            time.sleep(0.05)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=FACE_MIN_SIZE
        )

        owner_count = 0

        if len(faces) == 0:
            draw_label(frame, "NO FACE DETECTED", 20, 30, (0, 255, 255))
        else:
            for i, (x, y, w, h) in enumerate(faces):
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(frame.shape[1], x + w)
                y2 = min(frame.shape[0], y + h)

                face_crop_bgr = frame[y1:y2, x1:x2]

                if face_crop_bgr.size == 0:
                    continue

                face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)

                result = recognizer.recognize(face_crop_rgb)

                distance = float(result.get("distance", 999.0))
                is_owner = distance < OWNER_DISTANCE_THRESHOLD
                label = "OWNER" if is_owner else "UNKNOWN"

                if is_owner:
                    owner_count += 1
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                text1 = f"Face {i}"
                text2 = f"{label}"
                text3 = f"dist={distance:.3f}"

                draw_label(frame, text1, x1, max(20, y1 - 30), color, scale=0.55, thickness=2)
                draw_label(frame, text2, x1, max(40, y1 - 10), color, scale=0.6, thickness=2)
                draw_label(frame, text3, x1, min(frame.shape[0] - 10, y2 + 20), color, scale=0.55, thickness=2)

        summary_text = f"Faces: {len(faces)} | Owners detected: {owner_count} | Owner if dist < {OWNER_DISTANCE_THRESHOLD}"
        draw_label(frame, summary_text, 10, 25, (255, 255, 255), scale=0.6, thickness=2)

        cv2.imshow("FaceNet Owner Recognition Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
