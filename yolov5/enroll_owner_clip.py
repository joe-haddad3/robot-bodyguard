"""
CLIP Owner Enrollment Script

Captures owner images from camera and stores CLIP embeddings to gallery.

Usage:
    python enroll_owner_clip.py --camera 0 --views 3

This will:
  1. Show camera feed
  2. Capture 3 different views (front, back, side)
  3. Generate CLIP embeddings
  4. Save to face_db/clip_owner_gallery.npy
"""

import cv2
import argparse
import sys
from clip_owner_recognizer import CLIPOwnerRecognizer

def main():
    parser = argparse.ArgumentParser(description="Enroll owner for CLIP-based recognition")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--views", type=int, default=3, help="Number of views to capture")
    parser.add_argument("--threshold", type=float, default=0.82, help="Similarity threshold")
    args = parser.parse_args()
    
    recognizer = CLIPOwnerRecognizer(threshold=args.threshold)
    
    if not recognizer.enabled:
        print("ERROR: CLIP is not installed. Install with:")
        print("  pip install git+https://github.com/openai/CLIP.git")
        sys.exit(1)
    
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {args.camera}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"CLIP Owner Enrollment")
    print(f"{'='*60}")
    print(f"Camera: {args.camera}")
    print(f"Views to capture: {args.views}")
    print(f"Threshold: {args.threshold}")
    print()
    
    captured_count = 0
    
    while captured_count < args.views:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read frame")
            break
        
        # Display instructions
        h, w = frame.shape[:2]

        # Guide rectangle: center 60 % of width, full height.
        # The embedding is computed from this region — same domain as runtime
        # recognition, which uses the person bounding box crop from YOLO.
        gx1 = w // 5
        gx2 = 4 * w // 5

        instruction = [
            f"Capture View {captured_count + 1}/{args.views}",
            "",
            "Stand inside the YELLOW box",
            "Press SPACE to capture",
            "Press R to retake",
            "Press ESC to cancel",
        ]

        # Add view-specific instructions
        if captured_count == 0:
            instruction.append("\u2192 Face towards camera (FRONT view)")
        elif captured_count == 1:
            instruction.append("\u2192 Turn around (BACK view)")
        else:
            instruction.append("\u2192 Turn to the side (SIDE view)")

        # Draw info box
        y_offset = 30
        cv2.rectangle(frame, (10, 10), (420, y_offset + len(instruction) * 25), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (420, y_offset + len(instruction) * 25), (0, 255, 255), 2)

        for i, text in enumerate(instruction):
            cv2.putText(
                frame, text,
                (20, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1
            )

        # Draw guide rectangle — shows the region that will be embedded
        cv2.rectangle(frame, (gx1, 0), (gx2, h - 1), (0, 255, 255), 2)
        cv2.putText(frame, "stand here", (gx1 + 5, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("CLIP Owner Enrollment", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("Enrollment cancelled")
            break
        elif key == ord(' '):  # SPACE - capture
            print(f"Capturing view {captured_count + 1}...")
            # Crop to the guide region — same domain as runtime person-box crops
            person_crop = frame[:, gx1:gx2]
            embedding = recognizer.embed_image(person_crop)

            if embedding is not None:
                view_labels = ["FRONT", "BACK", "SIDE"]
                view_label = view_labels[min(captured_count, len(view_labels) - 1)]
                recognizer.add_to_gallery(embedding, label=view_label)
                captured_count += 1
                print(f"✓ View {captured_count}/{args.views} captured")
            else:
                print("✗ Failed to generate embedding")
        
        elif key == ord('r'):  # R - retake
            print(f"Retaking view {captured_count + 1}...")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    print(f"Enrollment Complete!")
    print(f"Gallery size: {recognizer.get_gallery_size()} embeddings")
    print(f"Saved to: face_db/clip_owner_gallery.npy")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
