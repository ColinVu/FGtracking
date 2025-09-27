#!/usr/bin/env python3
"""
Example usage of the Football Tracker system

This script demonstrates how to use the FootballTracker class programmatically
without command line arguments.
"""

from football_tracker import FootballTracker
import os


def main():
    """Example usage of the football tracker"""
    
    # Configuration
    input_video = "field_goal.mp4"
    output_video = "output.mp4"
    bbox_size = 30  # Size of bounding box for annotation
    skip_frames = 5  # Annotate every 5th frame
    
    print("Football Tracker Example")
    print("=" * 30)
    
    # Check if input video exists
    if not os.path.exists(input_video):
        print(f"Error: Input video '{input_video}' not found.")
        print("Please place your video file in the current directory.")
        return
    
    # Create tracker instance
    tracker = FootballTracker(
        input_video_path=input_video,
        output_video_path=output_video,
        bbox_size=bbox_size,
        skip_frames=skip_frames
    )
    
    print(f"Input video: {input_video}")
    print(f"Output video: {output_video}")
    print(f"Bounding box size: {bbox_size}")
    print(f"Frame skip interval: {skip_frames} (Note: Now shows all frames)")
    print()
    print("Workflow:")
    print("Phase 0 - Field Calibration:")
    print("- Click on yard line intersections")
    print("- Enter downfield distance (yards from goal line)")
    print("- Left-right position calculated automatically")
    print("- Need at least 4 points for 3D tracking")
    print()
    print("Phase 1 - Navigation controls during annotation:")
    print("- A/D: Navigate frames (1 frame left/right)")
    print("- W/X: Navigate frames (10 frames forward/back)")
    print("- HOME: Go to first frame, END: Go to last frame")
    print("- SPACE: Pause/resume navigation")
    print("- 'r': Remove annotation on current frame")
    print("- 's': Skip current frame, 'q': Finish annotation")
    print()
    print("Phase 2 - Automated processing with 3D trajectory fitting")
    print("Phase 3 - Video generation with 3D coordinates display")
    print()
    
    # Run the complete tracking pipeline
    try:
        tracker.run()
        print(f"\nSuccess! Check the output video: {output_video}")
    except Exception as e:
        print(f"Error during tracking: {str(e)}")


if __name__ == "__main__":
    main()
