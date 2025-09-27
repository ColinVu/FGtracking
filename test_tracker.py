#!/usr/bin/env python3
"""
Test script for the football tracker system
"""

import cv2
import numpy as np
import os
from football_tracker import FootballTracker, KalmanFilter


def create_test_video(filename="test_video.mp4", duration=5):
    """
    Create a simple test video with a moving circle (simulating a football)
    
    Args:
        filename: Output video filename
        duration: Video duration in seconds
    """
    print(f"Creating test video: {filename}")
    
    # Video properties
    width, height = 640, 480
    fps = 30
    total_frames = duration * fps
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    # Create frames with moving circle
    for frame_num in range(total_frames):
        # Create black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some background noise/texture
        noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        # Calculate circle position (parabolic trajectory)
        t = frame_num / fps
        x = int(100 + t * 400)  # Move from left to right
        y = int(200 - (t - 1) * (t - 1) * 100)  # Parabolic motion
        
        # Ensure circle stays within bounds
        x = max(20, min(x, width - 20))
        y = max(20, min(y, height - 20))
        
        # Draw circle (football simulation)
        cv2.circle(frame, (x, y), 15, (20, 50, 100), -1)  # Brownish color
        cv2.circle(frame, (x, y), 15, (40, 80, 120), 2)   # Border
        
        # Add some text
        cv2.putText(frame, f"Test Frame {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Test video created: {filename}")


def test_kalman_filter():
    """Test the Kalman filter functionality"""
    print("Testing Kalman Filter...")
    
    kf = KalmanFilter(dt=1.0)
    
    # Test initialization
    initial_pos = (100, 200)
    kf.initialize(initial_pos)
    
    # Test prediction and update
    for i in range(10):
        # Simulate noisy measurements
        measurement = (100 + i * 10 + np.random.normal(0, 2), 
                      200 + i * 5 + np.random.normal(0, 2))
        
        # Update filter
        corrected = kf.update(measurement)
        prediction = kf.predict()
        
        print(f"Step {i}: Measurement={measurement}, Corrected={corrected}, Prediction={prediction}")
    
    print("Kalman Filter test completed.")


def test_tracker_components():
    """Test individual tracker components"""
    print("Testing Tracker Components...")
    
    # Create test video
    test_video = "test_video.mp4"
    create_test_video(test_video, duration=3)
    
    # Test tracker initialization
    tracker = FootballTracker(test_video, "test_output.mp4", bbox_size=30, skip_frames=5)
    
    print(f"Tracker initialized with:")
    print(f"  Input: {tracker.input_video_path}")
    print(f"  Output: {tracker.output_video_path}")
    print(f"  Bbox size: {tracker.bbox_size}")
    print(f"  Skip frames: {tracker.skip_frames}")
    
    # Test HSV range
    print(f"HSV range: {tracker.lower_hsv} to {tracker.upper_hsv}")
    
    # Clean up test files
    if os.path.exists(test_video):
        os.remove(test_video)
    
    print("Component test completed.")


def main():
    """Run all tests"""
    print("Football Tracker Test Suite")
    print("=" * 40)
    
    try:
        # Test Kalman filter
        test_kalman_filter()
        print()
        
        # Test tracker components
        test_tracker_components()
        print()
        
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
