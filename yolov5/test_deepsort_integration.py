#!/usr/bin/env python
"""Test DeepSORT integration to verify update_tracks method and API."""

from deep_sort_realtime.deepsort_tracker import DeepSort
import inspect
import numpy as np

# Create DeepSort instance
deepsort = DeepSort(max_age=30, n_init=3, nn_budget=100)

# Print method signature
print("update_tracks method signature:")
print(inspect.signature(deepsort.update_tracks))
print()

# Print documentation
print("update_tracks docstring:")
print(deepsort.update_tracks.__doc__)
print()

# Try to understand the method
print("Method source (if available):")
try:
    print(inspect.getsource(deepsort.update_tracks))
except:
    print("Source not available")
