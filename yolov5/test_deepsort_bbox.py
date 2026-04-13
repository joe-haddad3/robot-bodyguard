#!/usr/bin/env python3
"""
Quick test of DeepSORT bbox handling without camera.
"""
import sys
sys.path.insert(0, '.')

from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# Initialize DeepSORT
deepsort = DeepSort(max_age=30, n_init=3, nn_budget=100)

# Create fake detections (bbox and conf)
# Format: [[bbox, confidence], [bbox, confidence], ...]
# where bbox = [x1, y1, x2, y2]
fake_detections = [
    [[100, 150, 250, 400], 0.95],
    [[300, 200, 450, 450], 0.90],
    [[50, 100, 180, 350], 0.85],
]

fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # dummy frame

print("[TEST] Updating DeepSORT with fake detections...")
tracks = deepsort.update_tracks(fake_detections, frame=fake_frame)

print(f"[TEST] Got {len(tracks)} tracks")

for i, track in enumerate(tracks):
    if not track.is_confirmed():
        continue
    
    track_id = track.track_id
    print(f"\n[TEST] Track {i} (ID: {track_id}):")
    print(f"  - is_confirmed: {track.is_confirmed()}")
    
    # Try to get bbox in tlbr format
    try:
        bbox_tlbr = track.to_tlbr()
        print(f"  - to_tlbr(): {bbox_tlbr} (type: {type(bbox_tlbr)}, len: {len(bbox_tlbr)})")
    except Exception as e:
        print(f"  - to_tlbr() failed: {e}")
    
    # Try to get bbox in tlwh format
    try:
        bbox_tlwh = track.to_tlwh()
        print(f"  - to_tlwh(): {bbox_tlwh} (type: {type(bbox_tlwh)}, len: {len(bbox_tlwh)})")
    except Exception as e:
        print(f"  - to_tlwh() failed: {e}")
    
    # Test conversion from tlwh to tlbr
    if 'bbox_tlwh' in locals():
        tlwh = bbox_tlwh
        bbox_converted = [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]
        print(f"  - converted tlwh to tlbr: {bbox_converted}")
    
    # Test unpacking for OpenCV
    try:
        bbox = track.to_tlbr()
        x1, y1, x2, y2 = [int(v) for v in bbox]
        print(f"  - unpacked for OpenCV: ({x1}, {y1}) to ({x2}, {y2}) ✓")
    except Exception as e:
        print(f"  - unpacking failed: {e}")

print("\n[TEST] Complete!")
