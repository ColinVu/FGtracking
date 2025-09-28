#!/usr/bin/env python3
"""
Test script to verify 3D computation is no longer skipped
"""

from football_tracker import FootballTracker
import numpy as np

def test_3d_computation():
    print("Testing 3D computation fix...")
    
    # Create a tracker instance
    tracker = FootballTracker("dummy_video.mp4", "dummy_output.mp4")
    
    # Simulate some annotations (without homography)
    tracker.annotations = [
        type('Annotation', (), {'frame_num': 0, 'center': (100, 200)})(),
        type('Annotation', (), {'frame_num': 10, 'center': (150, 180)})(),
        type('Annotation', (), {'frame_num': 20, 'center': (200, 160)})(),
        type('Annotation', (), {'frame_num': 30, 'center': (250, 180)})(),
        type('Annotation', (), {'frame_num': 40, 'center': (300, 200)})(),
    ]
    
    # Set up some basic properties
    tracker.fps = 30
    tracker.frame_width = 640
    tracker.frame_height = 480
    
    # Simulate final bboxes
    tracker.final_bboxes = {}
    for ann in tracker.annotations:
        x, y = ann.center
        tracker.final_bboxes[ann.frame_num] = (x-15, y-15, 30, 30)
    
    # Test without homography (should use alternative analysis)
    print("\n=== Testing WITHOUT homography ===")
    tracker.homography = None
    
    try:
        tracker.compute_3d_trajectory()
        if tracker.positions_3d:
            print(f"✓ SUCCESS: Generated {len(tracker.positions_3d)} 3D positions without calibration!")
            print(f"  Max height: {max(pos.y for pos in tracker.positions_3d):.1f} yards")
        else:
            print("✗ FAILED: No 3D positions generated")
    except Exception as e:
        print(f"✗ ERROR: {e}")
    
    # Test with dummy homography (should use enhanced fallback if optimization fails)
    print("\n=== Testing WITH dummy homography ===")
    tracker.homography = np.eye(3)  # Identity matrix as dummy
    tracker.positions_3d = []  # Reset
    
    try:
        tracker.compute_3d_trajectory()
        if tracker.positions_3d:
            print(f"✓ SUCCESS: Generated {len(tracker.positions_3d)} 3D positions with calibration!")
            print(f"  Max height: {max(pos.y for pos in tracker.positions_3d):.1f} yards")
        else:
            print("✗ FAILED: No 3D positions generated")
    except Exception as e:
        print(f"✗ ERROR: {e}")

if __name__ == "__main__":
    test_3d_computation()
