"""
Quick test of CLIP-based owner recognition.

Usage:
    python test_clip_owner.py
"""

import cv2
import sys
from clip_owner_recognizer import CLIPOwnerRecognizer

def main():
    print("\n" + "="*60)
    print("CLIP Owner Recognition Test")
    print("="*60 + "\n")
    
    # Initialize CLIP recognizer
    recognizer = CLIPOwnerRecognizer(threshold=0.82)
    
    if not recognizer.enabled:
        print("ERROR: CLIP not available. Install with:")
        print("  pip install git+https://github.com/openai/CLIP.git")
        return
    
    print(f"✓ CLIP model loaded")
    print(f"✓ Gallery size: {recognizer.get_gallery_size()} embeddings")
    
    if recognizer.get_gallery_size() == 0:
        print("\n⚠ Gallery is empty! Run enroll_owner_clip.py first to capture owner views.")
        return
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return
    
    print("\n" + "-"*60)
    print("Live Owner Recognition Test")
    print("-"*60)
    print("Press SPACE to test recognition on current frame")
    print("Press ESC to quit")
    print("-"*60 + "\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Display info
        h, w = frame.shape[:2]
        cv2.putText(frame, "Press SPACE to test | ESC to quit", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow("CLIP Owner Recognition Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # SPACE - test
            print(f"\nTesting frame {frame_count}...")
            result = recognizer.recognize(frame)
            
            print(f"  Is Owner: {result['is_owner']}")
            print(f"  Label: {result['label']}")
            print(f"  Similarity: {result['similarity']:.4f}")
            print(f"  Distance: {result['distance']:.4f}")
            
            # Draw result on frame
            color = (0, 255, 0) if result['is_owner'] else (0, 0, 255)
            text = f"{result['label']} ({result['similarity']:.2f})"
            cv2.rectangle(frame, (10, 100), (400, 150), color, -1, cv2.LINE_AA)
            cv2.putText(frame, text, (20, 135),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imshow("CLIP Owner Recognition Test", frame)
            cv2.waitKey(2000)  # Show result for 2 seconds
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
