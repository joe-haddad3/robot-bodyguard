#!/usr/bin/env python
"""Test DeepSORT bbox format."""

from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# Create DeepSort instance
deepsort = DeepSort(max_age=30, n_init=3, nn_budget=100)

# Simulate some detections
detections = [[100, 100, 200, 200, 0.9]]  # [x1, y1, x2, y2, conf]

tracks = deepsort.update_tracks(detections, frame=None)

for track in tracks:
    if track.is_confirmed():
        bbox = track.to_tlbr()
        print(f"bbox: {bbox} (type: {type(bbox)}, len: {len(bbox) if hasattr(bbox, '__len__') else 'N/A'})")
        print(f"list(bbox): {list(bbox)}")
        break