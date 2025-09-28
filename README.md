# Semi-Automatic Football Tracking System for Field Goals

A Python-based computer vision system for tracking a football in video files using OpenCV. The system combines user annotation with automated tracking algorithms to provide accurate and smooth football trajectory tracking.

## Features

- **Interactive Annotation**: User-guided annotation on every frame
- **Automated Refinement**: Color-based detection to refine user annotations
- **CSRT Tracking**: Advanced tracking between annotated frames
- **Kalman Filtering**: Smooth trajectory prediction and filtering
- **Trajectory Visualization**: Real-time trajectory drawing on output video
- **Flexible Configuration**: Customizable bounding box size and frame skipping

## Installation

1. Install Python 3.7 or higher
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python football_tracker.py
```

This will use the default input video `field_goal.mp4` and create `output.mp4`.

### Advanced Usage

```bash
python football_tracker.py --input your_video.mp4 --output result.mp4 --bbox-size 25 --skip-frames 3
```

### Command Line Arguments

- `--input, -i`: Input video file path (default: field_goal.mp4)
- `--output, -o`: Output video file path (default: output.mp4)
- `--bbox-size`: Bounding box size for annotation (default: 30)
- `--skip-frames`: Number of frames to skip between annotations (default: 5)

## How It Works

### Phase 0: Field Calibration (NEW!)
- Click on yard line intersections to establish field coordinates
- Enter downfield position for each point (Z=yards from goal line)
- Left-right position (X) calculated automatically from image position
- System computes camera calibration and ground plane homography
- Enables 3D position tracking in real-world coordinates

### Phase 1: Interactive Annotation
- All frames are accessible with full navigation control
- Click on the football to create a bounding box
- Navigation controls:
  - A/D: Navigate frames (1 frame left/right)
  - W/X: Navigate frames (10 frames forward/back)
  - HOME: Go to first frame, END: Go to last frame
  - SPACE: Pause/resume navigation
  - 'r': Remove annotation on current frame
  - 's': Skip current frame, 'q': Finish annotation

### Phase 2: Automated Tracking & 3D Analysis
- User annotations are refined using HSV color-based detection
- Linear interpolation between annotations (never after the last annotation)
- CSRT tracker fills gaps where interpolation isn't possible
- Kalman filter smooths the trajectory
- **Dynamic camera motion tracking** using field features (NEW!)
- **3D ballistic trajectory fitting** using physics-based model
- **Ground-level constraint**: First position always starts at height = 0 (NEW!)
- Real-world position calculation in yards (X, Y, Z coordinates)

### Phase 3: Visualization with 3D Data
- Output video is generated with:
  - Green bounding boxes around the tracked football
  - Red center points
  - Blue trajectory line showing the flight path
  - **Real-time 3D coordinates display**:
    - Height and downfield position in yards
    - Lateral position as relative movement
  - **Live trajectory graph** (Forward vs Height) in top-right corner (NEW!)
  - **Crossing point analysis**: Shows forward position when ball reaches 3.33yd height (NEW!)
  - Frame counter overlay

## Technical Details

### Color Detection
The system uses HSV color space to detect football color:
- Lower HSV: (10, 100, 20) - Brown/orange range
- Upper HSV: (25, 255, 255)
- Morphological operations clean up the detection mask

### Kalman Filter
- 4-state filter tracking position and velocity
- Predicts smooth motion between measurements
- Blends tracker and Kalman predictions for stability

### CSRT Tracker
- Advanced correlation-based tracking
- Handles occlusion and appearance changes
- Falls back to interpolation on failure

## Requirements

- Python 3.7+
- OpenCV 4.8.0+
- NumPy 1.21.0+
- SciPy 1.9.0+ (for 3D trajectory optimization)
- Matplotlib 3.5.0+ (for trajectory graphing)

## Example Workflow

1. Place your video file as `field_goal.mp4` in the project directory
2. Run the script: `python football_tracker.py`
3. **Phase 0**: Click on yard line intersections and enter downfield distances
4. **Phase 1**: Navigate through frames and annotate the football position
5. **Phase 2**: Wait for automated tracking and 3D trajectory computation
6. **Phase 3**: View the result in `output.mp4` with 3D coordinates displayed

## Tips for Best Results

- Ensure good lighting in your video
- Annotate the football when it's clearly visible
- Skip frames where the ball is occluded or unclear
- Use consistent annotation positions (center of the ball)
- Consider adjusting HSV ranges for different lighting conditions

## Troubleshooting

- **"Input video not found"**: Ensure the video file exists and path is correct
- **Poor tracking**: Try adjusting the HSV color ranges in the code
- **Opencv import error**: Install opencv-python: `pip install opencv-python`
- **Memory issues**: Reduce video resolution or use shorter clips

## License

This project is provided as-is for educational and research purposes.
