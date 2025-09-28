#!/usr/bin/env python3
"""
Semi-Automatic Football Tracking System

This script enables semi-automatic tracking of a football in a video file using OpenCV.
The process involves three main phases:
1. Interactive user annotation (every 5th frame)
2. Automated refinement and CSRT tracking with Kalman filtering
3. Video generation with trajectory visualization

Author: AI Assistant
"""

import cv2
import numpy as np
import os
import sys
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import argparse
from scipy.optimize import least_squares
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg


@dataclass
class Annotation:
    """Data structure to store user annotations"""
    frame_num: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[int, int]  # (cx, cy)
    point_type: str = "regular"  # "regular", "beginning", "end"


@dataclass
class FieldPoint:
    """Data structure for field calibration points"""
    image_point: Tuple[int, int]  # (u, v) in pixels
    field_point: Tuple[float, float]  # (x, z) in yards on field - x will be auto-calculated


@dataclass
class Position3D:
    """Data structure for 3D position"""
    x: float  # left-right (yards)
    y: float  # height (yards) 
    z: float  # forward-back (yards)
    frame_num: int


class KalmanFilter:
    """Kalman Filter for smoothing football trajectory"""
    
    def __init__(self, dt: float = 1.0):
        """
        Initialize Kalman filter for 2D position and velocity tracking
        
        Args:
            dt: Time step between measurements
        """
        self.kf = cv2.KalmanFilter(4, 2)  # 4 state variables, 2 measurements
        
        # State transition matrix (position and velocity)
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (only position is measured)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # Measurement noise covariance
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        
        # Error covariance
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        
        self.initialized = False
    
    def predict(self) -> Tuple[float, float]:
        """Predict next position"""
        prediction = self.kf.predict()
        return float(prediction[0]), float(prediction[1])
    
    def update(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        """Update filter with new measurement"""
        measurement = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
        corrected = self.kf.correct(measurement)
        return float(corrected[0]), float(corrected[1])
    
    def initialize(self, initial_pos: Tuple[float, float]):
        """Initialize the filter with first measurement"""
        self.kf.statePre = np.array([initial_pos[0], initial_pos[1], 0, 0], dtype=np.float32)
        self.kf.statePost = np.array([initial_pos[0], initial_pos[1], 0, 0], dtype=np.float32)
        self.initialized = True


class FootballTracker:
    """Main class for semi-automatic football tracking"""
    
    def __init__(self, input_video_path: str, output_video_path: str = "output.mp4", 
                 bbox_size: int = 30, skip_frames: int = 5):
        """
        Initialize the football tracker
        
        Args:
            input_video_path: Path to input video file
            output_video_path: Path for output video file
            bbox_size: Size of bounding box for annotation
            skip_frames: Number of frames to skip between annotations
        """
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.bbox_size = bbox_size
        self.skip_frames = skip_frames
        
        # HSV range for football color (brown/orange)
        self.lower_hsv = np.array([10, 100, 20])
        self.upper_hsv = np.array([25, 255, 255])
        
        # Data structures
        self.annotations: List[Annotation] = []
        self.final_bboxes: List[Tuple[int, int, int, int]] = []
        self.trajectory_points: List[Tuple[int, int]] = []
        self.field_points: List[FieldPoint] = []
        self.positions_3d: List[Position3D] = []
        
        # Special point tracking
        self.beginning_point: Optional[Annotation] = None
        self.end_point: Optional[Annotation] = None
        self.calculated_distance: Optional[float] = None
        self.starting_position: Optional[float] = None
        
        # Camera calibration (initial/reference)
        self.homography = None
        self.camera_matrix = None
        self.rotation_matrix = None
        self.translation_vector = None
        
        # Dynamic camera tracking
        self.frame_homographies: List[np.ndarray] = []  # Homography for each frame
        self.frame_camera_matrices: List[np.ndarray] = []  # Camera matrix for each frame
        self.reference_features = None  # Field features from calibration frame
        
        # Trajectory analysis
        self.crossing_point_y = None  # Y position when Z=3.33 on downward swing
        self.trajectory_data = []  # Store (Y, Z) pairs for graphing
        
        # Tracking components
        self.kalman_filter = KalmanFilter()
        self.csrt_tracker = None
        
        # Video properties
        self.cap = None
        self.fps = 30
        self.frame_width = 0
        self.frame_height = 0
        self.total_frames = 0
        
        # Mouse callback variables
        self.mouse_pos = None
        self.annotation_ready = False
        self.pending_point_type = "regular"  # Track what type of point to create next
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for annotation selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_pos = (x, y)
            self.annotation_ready = True
    
    def draw_annotation_box(self, frame: np.ndarray, center: Tuple[int, int], point_type: str = "regular") -> np.ndarray:
        """Draw annotation bounding box on frame with color coding based on point type"""
        x, y = center
        half_size = self.bbox_size // 2
        
        # Ensure box stays within frame bounds
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(frame.shape[1], x + half_size)
        y2 = min(frame.shape[0], y + half_size)
        
        # Choose color based on point type
        if point_type == "beginning":
            color = (0, 255, 0)  # Green for beginning point
            label = "BEGIN"
        elif point_type == "end":
            color = (0, 0, 255)  # Red for end point
            label = "END"
        else:
            color = (255, 0, 0)  # Blue for regular points
            label = "REG"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, center, 3, color, -1)
        
        # Add label
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def phase1_annotation(self):
        """Phase 1: Interactive user annotation with full frame navigation"""
        print("Phase 1: Interactive Annotation")
        print("Instructions:")
        print("- Click on the football to annotate it")
        print("- Press 'g' to mark next click as BEGINNING point (green) - starting position")
        print("- Press 'h' to mark next click as END point (red) - ending position for distance calc")
        print("- Press 's' to skip current frame")
        print("- Press 'r' to remove annotation from current frame")
        print("- Press 'q' to finish annotation and proceed to Phase 2")
        print("- Use keyboard keys to navigate frames:")
        print("  A/D: Move one frame (left/right)")
        print("  W/X: Move 10 frames (forward/back)")
        print("  HOME: Go to first frame")
        print("  END: Go to last frame")
        print("- Press SPACE to pause/resume navigation")
        print("- Both points will be analyzed using CV field line detection for yard positions")
        print("- END point will be used as manual calibration at height=0 for distance calculation")
        
        if not os.path.exists(self.input_video_path):
            raise FileNotFoundError(f"Input video not found: {self.input_video_path}")
        
        self.cap = cv2.VideoCapture(self.input_video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {self.frame_width}x{self.frame_height}, {self.fps} FPS, {self.total_frames} frames")
        
        # Create window and set mouse callback
        cv2.namedWindow('Football Annotation', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Football Annotation', self.mouse_callback)
        
        current_frame = 0
        self.mouse_pos = None
        self.annotation_ready = False
        
        # Load all frames into memory for faster navigation
        print("Loading video frames...")
        frames = []
        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame.copy())
            frame_count += 1
        
        self.cap.release()
        print(f"Loaded {len(frames)} frames")
        
        # Main annotation loop
        while True:
            if current_frame >= len(frames):
                current_frame = len(frames) - 1
            elif current_frame < 0:
                current_frame = 0
            
            frame = frames[current_frame]
            
            # Check if this frame already has an annotation
            has_annotation = False
            existing_annotation = None
            for ann in self.annotations:
                if ann.frame_num == current_frame:
                    has_annotation = True
                    existing_annotation = ann
                    break
            
            # Display frame with instructions and current status
            display_frame = frame.copy()
            
            # Show frame information
            cv2.putText(display_frame, f"Frame {current_frame} / {len(frames)-1}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show annotation status
            if has_annotation:
                center = existing_annotation.center
                self.draw_annotation_box(display_frame, center, existing_annotation.point_type)
                cv2.putText(display_frame, f"ANNOTATED ({existing_annotation.point_type.upper()}) at {center}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # Show current point type mode
                mode_text = f"Click to annotate ({self.pending_point_type.upper()}) - Press 'g' for BEGIN, 'h' for END"
                cv2.putText(display_frame, mode_text, 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Show navigation instructions
            cv2.putText(display_frame, "A/D: 1 frame | W/X: 10 frames | SPACE: Pause | 's': Skip | 'q': Finish", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show point type status
            begin_status = "SET" if self.beginning_point else "NOT SET"
            end_status = "SET" if self.end_point else "NOT SET"
            cv2.putText(display_frame, f"Total: {len(self.annotations)} | Begin: {begin_status} | End: {end_status}", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Football Annotation', display_frame)
            
            # Handle keyboard input with small delay for better control
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                print(f"Annotation complete. Total annotations: {len(self.annotations)}")
                cv2.destroyAllWindows()
                return
            
            elif key == ord('s'):
                print(f"Skipped frame {current_frame}")
                current_frame += 1
            
            elif key == ord('w'):  # 'w' - move forward 10 frames
                current_frame += 10
            
            elif key == ord('x'):  # 'x' - move back 10 frames
                current_frame -= 10
            
            elif key == ord('a'):  # 'a' - previous frame
                current_frame -= 1
            
            elif key == ord('d'):  # 'd' - next frame
                current_frame += 1
            
            elif key == ord(' '):  # SPACE - pause/resume
                print("Paused. Press SPACE again to resume.")
                while True:
                    pause_key = cv2.waitKey(0) & 0xFF
                    if pause_key == ord(' '):
                        print("Resumed.")
                        break
                    elif pause_key == ord('q'):
                        print(f"Annotation complete. Total annotations: {len(self.annotations)}")
                        cv2.destroyAllWindows()
                        return
            
            elif key == 80:  # HOME key - go to first frame
                current_frame = 0
            
            elif key == 87:  # END key - go to last frame
                current_frame = len(frames) - 1
            
            elif key == ord('r'):  # Remove annotation on current frame
                if has_annotation:
                    # If removing a special point, clear the reference
                    if existing_annotation.point_type == "beginning":
                        self.beginning_point = None
                    elif existing_annotation.point_type == "end":
                        self.end_point = None
                    
                    self.annotations = [ann for ann in self.annotations if ann.frame_num != current_frame]
                    print(f"Removed annotation from frame {current_frame}")
            
            elif key == ord('g'):  # Set mode to create beginning point
                self.pending_point_type = "beginning"
                print("Mode: Creating BEGINNING point - click to place")
            
            elif key == ord('h'):  # Set mode to create end point
                self.pending_point_type = "end"
                print("Mode: Creating END point - click to place")
            
            elif self.annotation_ready and self.mouse_pos and not has_annotation:
                # Create new annotation
                center = self.mouse_pos
                bbox = (center[0] - self.bbox_size//2, center[1] - self.bbox_size//2, 
                        self.bbox_size, self.bbox_size)
                
                annotation = Annotation(current_frame, bbox, center, self.pending_point_type)
                
                # Handle special point types
                if self.pending_point_type == "beginning":
                    if self.beginning_point is not None:
                        # Remove previous beginning point
                        self.annotations = [ann for ann in self.annotations if ann.point_type != "beginning"]
                    self.beginning_point = annotation
                    print(f"Set BEGINNING point at frame {current_frame}, position {center}")
                elif self.pending_point_type == "end":
                    if self.end_point is not None:
                        # Remove previous end point
                        self.annotations = [ann for ann in self.annotations if ann.point_type != "end"]
                    self.end_point = annotation
                    print(f"Set END point at frame {current_frame}, position {center}")
                else:
                    print(f"Annotated frame {current_frame} at position {center}")
                
                self.annotations.append(annotation)
                self.annotation_ready = False
                
                # Reset to regular mode after placing special points
                if self.pending_point_type in ["beginning", "end"]:
                    self.pending_point_type = "regular"
                
                # Continue to next iteration to show the annotation immediately
        
        cv2.destroyAllWindows()
    
    def phase0_field_calibration(self):
        """Phase 0: Field calibration using yard line markers"""
        print("Phase 0: Field Calibration")
        print("Instructions:")
        print("- Click on yard line intersections that you can identify")
        print("- For each click, you'll be asked to enter the downfield coordinate")
        print("- Downfield coordinate: Z = yards from goal line (0=goal line, 100=far goal line)")
        print("- Left-right position (X) will be calculated automatically")
        print("- IMPORTANT: Spread points across different yard lines and both sides of field")
        print("- Click at least 4 points for good calibration")
        print("- Press 'q' when done with calibration")
        
        if not os.path.exists(self.input_video_path):
            raise FileNotFoundError(f"Input video not found: {self.input_video_path}")
        
        # Use the first frame for calibration
        cap = cv2.VideoCapture(self.input_video_path)
        
        # Get video properties
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError("Could not read first frame for calibration")
        
        print(f"Video loaded: {self.frame_width}x{self.frame_height}, {self.fps} FPS, {self.total_frames} frames")
        
        cv2.namedWindow('Field Calibration', cv2.WINDOW_NORMAL)
        
        calibration_points = []
        current_click = None
        
        def calibration_mouse_callback(event, x, y, flags, param):
            nonlocal current_click
            if event == cv2.EVENT_LBUTTONDOWN:
                current_click = (x, y)
        
        cv2.setMouseCallback('Field Calibration', calibration_mouse_callback)
        
        while True:
            display_frame = frame.copy()
            
            # Draw existing calibration points
            for i, fp in enumerate(calibration_points):
                cv2.circle(display_frame, fp.image_point, 5, (0, 255, 0), -1)
                cv2.putText(display_frame, f"Z={fp.field_point[1]:.0f}yd", 
                           (fp.image_point[0] + 10, fp.image_point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show instructions
            cv2.putText(display_frame, f"Calibration points: {len(calibration_points)}/4+", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, "Click on yard line intersections, press 'q' when done", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display_frame, "Spread points across different yard lines and sides", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            if current_click:
                cv2.circle(display_frame, current_click, 3, (0, 0, 255), -1)
            
            cv2.imshow('Field Calibration', display_frame)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                if len(calibration_points) >= 4:
                    print(f"Calibration complete with {len(calibration_points)} points")
                    break
                else:
                    print(f"Need at least 4 points, have {len(calibration_points)}")
            
            elif current_click:
                # Get field coordinates from user
                print(f"\nClicked at pixel ({current_click[0]}, {current_click[1]})")
                try:
                    z_field = float(input("Enter downfield coordinate (yards from goal line, 0-100): "))
                    
                    # Calculate relative X coordinate based on image position
                    # Use pixel position relative to image center, scaled to reasonable field width
                    # Assume field width visible is about 40-60 yards
                    field_width_estimate = 50.0  # yards
                    
                    if self.frame_width > 0:
                        x_field = (current_click[0] - self.frame_width / 2) * field_width_estimate / self.frame_width
                    else:
                        print("Error: Frame width is 0. Using default X coordinate.")
                        x_field = 0.0
                    
                    field_point = FieldPoint(current_click, (x_field, z_field))
                    calibration_points.append(field_point)
                    self.field_points.append(field_point)
                    
                    print(f"Added calibration point: pixel {current_click} -> field (X={x_field:.1f}, Z={z_field})")
                    
                except ValueError:
                    print("Invalid input, skipping point")
                
                current_click = None
        
        cv2.destroyAllWindows()
        
        # Compute homography
        if len(calibration_points) >= 4:
            self.compute_camera_calibration()
        else:
            print("Insufficient calibration points, 3D tracking will be disabled")
    
    def compute_camera_calibration(self):
        """Compute homography and camera parameters from field calibration points"""
        if len(self.field_points) < 4:
            return
        
        try:
            # Extract image and field points
            image_points = np.array([fp.image_point for fp in self.field_points], dtype=np.float32)
            field_points = np.array([fp.field_point for fp in self.field_points], dtype=np.float32)
            
            # Convert field coordinates to meters (1 yard = 0.9144 meters)
            field_points_m = field_points * 0.9144
            
            # Compute homography (image to ground plane)
            self.homography, _ = cv2.findHomography(image_points, field_points_m, cv2.RANSAC)
            
            if self.homography is None:
                print("Failed to compute homography - insufficient or degenerate points")
                return
            
            # Use a more robust camera parameter estimation
            # Estimate focal length from field of view assumptions
            # Typical broadcast camera has ~50-60 degree horizontal FOV
            estimated_fov_degrees = 55.0
            estimated_fov_radians = np.radians(estimated_fov_degrees)
            fx = self.frame_width / (2.0 * np.tan(estimated_fov_radians / 2.0))
            fy = fx  # Assume square pixels
            
            cx = self.frame_width / 2.0
            cy = self.frame_height / 2.0
            
            self.camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Simplified camera pose estimation using homography decomposition
            # Use OpenCV's built-in homography decomposition which is more robust
            retval, rotations, translations, normals = cv2.decomposeHomographyMat(
                self.homography, self.camera_matrix)
            
            if retval > 0:
                # Choose the solution with the most reasonable normal vector (pointing up)
                best_idx = 0
                best_normal_y = -1
                
                for i in range(retval):
                    normal = normals[i].flatten()
                    if normal[1] > best_normal_y:  # Y component should be positive (up)
                        best_normal_y = normal[1]
                        best_idx = i
                
                self.rotation_matrix = rotations[best_idx]
                self.translation_vector = translations[best_idx].flatten()
                
                print("Camera calibration computed successfully")
                print(f"Focal length: {fx:.1f} pixels")
                print(f"Camera position: {self.translation_vector}")
                
            else:
                print("Failed to decompose homography - using simplified approach")
                # Fallback to identity rotation and zero translation
                self.rotation_matrix = np.eye(3, dtype=np.float32)
                self.translation_vector = np.array([0, 0, 0], dtype=np.float32)
                
        except np.linalg.LinAlgError as e:
            print(f"Camera calibration failed due to singular matrix: {e}")
            print("This usually means the calibration points are not well distributed.")
            print("Try selecting points that are more spread out across the field.")
            # Disable 3D tracking by setting homography to None
            self.homography = None
            
        except Exception as e:
            print(f"Camera calibration failed: {e}")
            self.homography = None
    
    def image_to_3d_ballistic(self, image_points_with_time: List[Tuple[Tuple[int, int], float]]) -> List[Position3D]:
        """
        Convert image points to 3D using ballistic trajectory fitting with dynamic camera tracking
        
        Args:
            image_points_with_time: List of ((u, v), time_seconds) tuples
            
        Returns:
            List of 3D positions
        """
        if len(image_points_with_time) < 3:
            return []
        
        # If no proper calibration, use alternative 3D analysis methods
        if self.homography is None:
            print("No camera calibration available - using alternative 3D trajectory analysis")
            return self.alternative_3d_analysis(image_points_with_time)
        
        # Use frame-specific homographies if available
        use_dynamic_camera = len(self.frame_homographies) > 0
        
        if use_dynamic_camera:
            print(f"Using dynamic camera tracking with {len(self.frame_homographies)} frame-specific homographies")
        else:
            print("Using static camera model")
        
        def ballistic_model(params, times):
            """Ballistic trajectory model: P(t) = P0 + V0*t + 0.5*g*t^2"""
            x0, y0, z0, vx0, vy0, vz0 = params
            g = 9.81  # gravity in m/s^2
            
            positions = []
            for t in times:
                x = x0 + vx0 * t
                y = y0 + vy0 * t - 0.5 * g * t * t
                z = z0 + vz0 * t
                positions.append([x, y, z])
            
            return np.array(positions)
        
        def residual_function(params):
            """Residual function for optimization with dynamic camera parameters"""
            try:
                times = [t for _, t in image_points_with_time]
                predicted_3d = ballistic_model(params, times)
                
                residuals = []
                for i, ((u, v), t) in enumerate(image_points_with_time):
                    # Project 3D point to image
                    point_3d = predicted_3d[i]
                    
                    # Get frame-specific camera parameters if available
                    if use_dynamic_camera:
                        frame_num = int(t * self.fps)
                        if 0 <= frame_num < len(self.frame_homographies):
                            # Use frame-specific homography for projection
                            # For simplicity, use homography-based projection
                            point_ground = np.array([point_3d[0], point_3d[2], 1], dtype=np.float32)  # X, Z, 1
                            frame_homography = self.frame_homographies[frame_num]
                            
                            try:
                                projected_2d = np.linalg.inv(frame_homography) @ point_ground
                                if abs(projected_2d[2]) > 1e-6:
                                    projected_2d = projected_2d / projected_2d[2]
                                    u_proj, v_proj = projected_2d[0], projected_2d[1]
                                    
                                    # Adjust for height with proper perspective scaling
                                    # Height effect depends on distance from camera
                                    distance_to_camera = max(abs(point_3d[2]) + 10, 5)  # Minimum 5m distance
                                    # Perspective scaling: closer objects show more height change per meter
                                    height_pixels_per_meter = max(self.frame_height, self.frame_width) / (distance_to_camera * 0.5)
                                    height_offset = point_3d[1] * height_pixels_per_meter
                                    v_proj -= height_offset  # Subtract because image Y increases downward
                                    
                                    residuals.extend([u - u_proj, v - v_proj])
                                else:
                                    residuals.extend([1000, 1000])
                            except:
                                residuals.extend([1000, 1000])
                        else:
                            residuals.extend([1000, 1000])
                    else:
                        # Use static camera model
                        # Transform to camera coordinates
                        point_cam = self.rotation_matrix @ point_3d + self.translation_vector
                        
                        # Project to image
                        if point_cam[2] > 0:  # In front of camera
                            projected = self.camera_matrix @ point_cam
                            u_proj = projected[0] / projected[2]
                            v_proj = projected[1] / projected[2]
                            
                            residuals.extend([u - u_proj, v - v_proj])
                        else:
                            residuals.extend([1000, 1000])  # Large error for points behind camera
                
                return np.array(residuals)
            except Exception as e:
                print(f"Error in residual function: {e}")
                # Return large residuals to indicate failure
                return np.array([1000.0] * (len(image_points_with_time) * 2))
        
        # Initial guess: use homography to get ground plane positions, assume reasonable height
        initial_ground_points = []
        times = []
        
        for (u, v), t in image_points_with_time:
            try:
                # Project to ground plane using appropriate homography
                point_img = np.array([u, v, 1], dtype=np.float32)
                
                # Use frame-specific homography if available
                if use_dynamic_camera:
                    frame_num = int(t * self.fps)
                    if 0 <= frame_num < len(self.frame_homographies):
                        current_homography = self.frame_homographies[frame_num]
                    else:
                        current_homography = self.homography
                else:
                    current_homography = self.homography
                
                point_ground = current_homography @ point_img
                
                # Check for valid homography result
                if abs(point_ground[2]) > 1e-6:
                    point_ground = point_ground / point_ground[2]  # Normalize
                    # Height depends on position in trajectory - first point is always at ground level (height = 0)
                    if len(initial_ground_points) == 0:
                        # First point is always at ground level
                        initial_height = 0.0  # meters (ground level)
                    else:
                        # Estimate height based on trajectory progression - parabolic arc
                        trajectory_factor = len(initial_ground_points) / max(len(image_points_with_time), 1)
                        # Parabolic trajectory: starts at 0, peaks in middle, ends at 0
                        initial_height = 15.0 * trajectory_factor * (1 - trajectory_factor)  # Up to 15m (~16 yards) peak
                    initial_ground_points.append([point_ground[0], initial_height, point_ground[1]])
                else:
                    # Fallback if homography gives bad result
                    if len(initial_ground_points) == 0:
                        initial_ground_points.append([0, 0.0, 10])  # First point at ground level
                    else:
                        # Use parabolic height estimation for fallback too
                        trajectory_factor = len(initial_ground_points) / max(len(image_points_with_time), 1)
                        fallback_height = 15.0 * trajectory_factor * (1 - trajectory_factor)
                        initial_ground_points.append([0, fallback_height, 10])
                    
                times.append(t)
            except Exception as e:
                print(f"Warning: Failed to project point {(u, v)}: {e}")
                # Use fallback position
                if len(initial_ground_points) == 0:
                    initial_ground_points.append([0, 0.0, 10])  # First point at ground level
                else:
                    # Use parabolic height estimation for exception fallback too
                    trajectory_factor = len(initial_ground_points) / max(len(image_points_with_time), 1)
                    fallback_height = 15.0 * trajectory_factor * (1 - trajectory_factor)
                    initial_ground_points.append([0, fallback_height, 10])
                times.append(t)
        
        if not initial_ground_points:
            print("No valid ground points found")
            return []
        
        initial_ground_points = np.array(initial_ground_points)
        
        # Estimate initial velocity from finite differences with better error handling
        if len(initial_ground_points) > 1:
            dt = times[1] - times[0] if len(times) > 1 and times[1] != times[0] else 1.0
            if dt > 0:
                velocity_estimate = (initial_ground_points[1] - initial_ground_points[0]) / dt
                # Clamp velocity to reasonable ranges (much more generous for football)
                velocity_estimate[0] = np.clip(velocity_estimate[0], -20, 20)  # lateral velocity
                velocity_estimate[1] = np.clip(velocity_estimate[1], -5, 50)   # vertical velocity (allow up to 50 m/s)
                velocity_estimate[2] = np.clip(velocity_estimate[2], -10, 30)  # forward velocity
                initial_velocity = velocity_estimate
            else:
                initial_velocity = np.array([0, 20, 15])  # Default with higher vertical velocity
        else:
            initial_velocity = np.array([0, 20, 15])  # Default with higher vertical velocity
        
        # Initial parameters: [x0, y0, z0, vx0, vy0, vz0]
        # Force the initial height (y0) to be 0
        initial_params = np.concatenate([initial_ground_points[0], initial_velocity])
        initial_params[1] = 0.0  # Force initial height to be 0
        
        # Make bounds more flexible and ensure initial guess is within bounds
        # Clamp initial parameters to be within bounds, but force y0 = 0
        lower_bounds = [-100, 0, -50, -50, -10, -50]  # More flexible bounds
        upper_bounds = [100, 0, 200, 50, 80, 100]     # Allow much higher trajectories (80m = ~87 yards max height)
        
        # Ensure initial guess is within bounds
        initial_params_clamped = np.clip(initial_params, lower_bounds, upper_bounds)
        initial_params_clamped[1] = 0.0  # Ensure initial height is exactly 0
        
        print(f"Initial trajectory parameters: {initial_params_clamped}")
        print(f"Position: ({initial_params_clamped[0]:.1f}, {initial_params_clamped[1]:.1f}, {initial_params_clamped[2]:.1f}) m")
        print(f"Velocity: ({initial_params_clamped[3]:.1f}, {initial_params_clamped[4]:.1f}, {initial_params_clamped[5]:.1f}) m/s")
        
        # Debug: Show initial height estimates and expected pixel effects
        print("Debug - Initial height estimates:")
        for i, point in enumerate(initial_ground_points):
            # Estimate pixel effect for this height
            distance_est = max(abs(point[2]) + 10, 5)
            height_pixels_per_meter = max(self.frame_height, self.frame_width) / (distance_est * 0.5)
            pixel_effect = point[1] * height_pixels_per_meter
            print(f"  Point {i}: Height = {point[1]:.2f}m ({point[1] * 1.094:.1f} yards) -> {pixel_effect:.0f} pixel effect")
        
        # Optimize
        print("Starting optimization...")
        print(f"Initial params before optimization: {initial_params_clamped}")
        try:
            result = least_squares(residual_function, initial_params_clamped, 
                                 bounds=(lower_bounds, upper_bounds))
            print(f"Optimization result: success={result.success}, cost={result.cost:.6f}")
            print(f"Final params after optimization: {result.x}")
            
            if result.success:
                print(f"Optimization converged! Cost: {result.cost:.6f}")
                
                # Generate 3D positions for all time points
                optimized_params = result.x
                print(f"Optimized parameters: {optimized_params}")
                print(f"Final position: ({optimized_params[0]:.1f}, {optimized_params[1]:.1f}, {optimized_params[2]:.1f}) m")
                print(f"Final velocity: ({optimized_params[3]:.1f}, {optimized_params[4]:.1f}, {optimized_params[5]:.1f}) m/s")
                
                times_all = [t for _, t in image_points_with_time]
                positions_3d_m = ballistic_model(optimized_params, times_all)
                
                # Check actual trajectory heights
                max_height_m = max(pos[1] for pos in positions_3d_m)
                print(f"Calculated max height: {max_height_m:.2f}m ({max_height_m * 1.094:.1f} yards)")
                
                # Show trajectory progression
                print("Trajectory heights:")
                for i, pos in enumerate(positions_3d_m[:min(5, len(positions_3d_m))]):
                    print(f"  Point {i}: {pos[1]:.2f}m ({pos[1] * 1.094:.1f} yards)")
                
                # Convert back to yards and create Position3D objects
                positions_3d = []
                for i, pos_m in enumerate(positions_3d_m):
                    pos_yards = pos_m / 0.9144  # Convert meters to yards
                    frame_num = int(image_points_with_time[i][1] * self.fps)  # Convert time to frame
                    
                    # Ensure first position is always at ground level (height = 0)
                    if i == 0:
                        pos_yards[1] = 0.0  # Force first height to be exactly 0
                    
                    positions_3d.append(Position3D(pos_yards[0], pos_yards[1], pos_yards[2], frame_num))
                
                # Validate that first position is at ground level
                if positions_3d and positions_3d[0].y != 0.0:
                    print(f"Warning: Adjusting first position height from {positions_3d[0].y:.3f} to 0.0 yards")
                    positions_3d[0] = Position3D(positions_3d[0].x, 0.0, positions_3d[0].z, positions_3d[0].frame_num)
                
                return positions_3d
            
        except Exception as e:
            print(f"3D trajectory optimization failed: {e}")
            print("*** USING ENHANCED FALLBACK TRAJECTORY MODEL ***")
            print("Main optimization failed - providing detailed fallback analysis:")
            
            # Enhanced fallback: detailed analysis even with failed optimization
            try:
                print("\n=== ENHANCED FALLBACK ANALYSIS ===")
                
                # First, try to extract what we can from the homography
                if self.homography is not None:
                    print("Using available homography for ground plane projection...")
                    
                    # Extract ground positions
                    ground_positions = []
                    for u, v in [point for point, _ in image_points_with_time]:
                        point_img = np.array([u, v, 1], dtype=np.float32)
                        point_ground = self.homography @ point_img
                        if abs(point_ground[2]) > 1e-6:
                            point_ground = point_ground / point_ground[2]
                            ground_positions.append([point_ground[0], point_ground[1]])
                    
                    if ground_positions:
                        ground_positions = np.array(ground_positions)
                        ground_distance = np.linalg.norm(ground_positions[-1] - ground_positions[0])
                        print(f"   Ground plane distance: {ground_distance:.1f} meters ({ground_distance * 1.094:.1f} yards)")
                        
                        # Analyze ground motion pattern
                        if len(ground_positions) > 2:
                            velocities = []
                            for i in range(1, len(ground_positions)):
                                dt = image_points_with_time[i][1] - image_points_with_time[i-1][1]
                                if dt > 0:
                                    vel = np.linalg.norm(ground_positions[i] - ground_positions[i-1]) / dt
                                    velocities.append(vel)
                            
                            if velocities:
                                avg_velocity = np.mean(velocities)
                                print(f"   Average ground velocity: {avg_velocity:.1f} m/s ({avg_velocity * 1.094:.1f} yards/s)")
                
                # Analyze image trajectory for height estimation
                print("\nAnalyzing image trajectory for height estimation...")
                v_coords = [v for u, v in [point for point, _ in image_points_with_time]]
                times = [t for _, t in image_points_with_time]
                
                # Find peak and estimate physics
                min_v_idx = np.argmin(v_coords)
                peak_time = times[min_v_idx] - times[0]
                total_time = times[-1] - times[0]
                
                print(f"   Time to peak: {peak_time:.2f}s (of {total_time:.2f}s total)")
                
                # Multiple height estimation methods
                methods_results = []
                
                # Method 1: Physics-based from time to peak
                if peak_time > 0:
                    vy0_physics = 9.81 * peak_time
                    max_height_physics = (vy0_physics ** 2) / (2 * 9.81)
                    methods_results.append(("Physics (time to peak)", max_height_physics))
                    print(f"   Physics method: {max_height_physics:.1f}m ({max_height_physics * 1.094:.1f} yards)")
                
                # Method 2: Pixel displacement analysis
                v_range = max(v_coords) - min(v_coords)
                if v_range > 0:
                    # Estimate based on typical camera angles (30-60 degrees from horizontal)
                    for angle in [30, 45, 60]:
                        height_from_pixels = v_range * np.tan(np.radians(angle)) * 0.1  # rough conversion
                        methods_results.append((f"Pixel analysis ({angle}° camera)", height_from_pixels))
                        print(f"   Pixel method ({angle}° camera): {height_from_pixels:.1f}m ({height_from_pixels * 1.094:.1f} yards)")
                
                # Method 3: Trajectory curvature analysis
                if len(v_coords) >= 5:
                    # Fit parabola to v-coordinates
                    t_norm = np.linspace(0, 1, len(v_coords))
                    try:
                        # Fit v = a*t^2 + b*t + c
                        coeffs = np.polyfit(t_norm, v_coords, 2)
                        # Peak occurs at t = -b/(2a)
                        if coeffs[0] != 0:
                            t_peak_fit = -coeffs[1] / (2 * coeffs[0])
                            if 0 <= t_peak_fit <= 1:
                                curvature_height = abs(coeffs[0]) * 0.05  # rough conversion
                                methods_results.append(("Curvature analysis", curvature_height))
                                print(f"   Curvature method: {curvature_height:.1f}m ({curvature_height * 1.094:.1f} yards)")
                    except:
                        pass
                
                # Choose best estimate
                if methods_results:
                    # Prefer physics-based method, but validate against others
                    physics_results = [r for name, r in methods_results if "Physics" in name]
                    if physics_results:
                        best_height = physics_results[0]
                        method_name = "Physics-based"
                    else:
                        best_height = np.median([r for _, r in methods_results])
                        method_name = "Median of methods"
                    
                    print(f"\n   BEST ESTIMATE: {best_height:.1f}m ({best_height * 1.094:.1f} yards) [{method_name}]")
                else:
                    best_height = 15.0  # Default reasonable height
                    print(f"\n   Using default height: {best_height:.1f}m ({best_height * 1.094:.1f} yards)")
                
                # Generate trajectory with enhanced height model
                fallback_positions = []
                for i, ((u, v), t) in enumerate(image_points_with_time):
                    if self.homography is not None:
                        # Use homography for X,Z
                        point_img = np.array([u, v, 1], dtype=np.float32)
                        point_ground = self.homography @ point_img
                        point_ground = point_ground / point_ground[2]
                        x_pos, z_pos = point_ground[0], point_ground[1]
                    else:
                        # Estimate from pixel positions
                        x_pos = (u - 320) * 0.1  # rough conversion
                        z_pos = i * 5  # assume 5m forward per frame
                    
                    # Enhanced height model using best estimate
                    t_normalized = (t - times[0]) / (times[-1] - times[0]) if times[-1] > times[0] else 0
                    height_factor = 4 * t_normalized * (1 - t_normalized)  # parabolic, peak at t=0.5
                    height_m = best_height * height_factor
                    
                    pos_yards = np.array([x_pos, height_m, z_pos]) / 0.9144
                    frame_num = int(t * self.fps) if hasattr(self, 'fps') else i
                    fallback_positions.append(Position3D(pos_yards[0], pos_yards[1], pos_yards[2], frame_num))
                
                # Ensure first position is at ground level
                if fallback_positions:
                    if fallback_positions[0].y != 0.0:
                        print(f"Adjusting fallback first position height from {fallback_positions[0].y:.3f} to 0.0 yards")
                        fallback_positions[0] = Position3D(fallback_positions[0].x, 0.0, fallback_positions[0].z, fallback_positions[0].frame_num)
                
                print(f"Fallback method generated {len(fallback_positions)} positions")
                return fallback_positions
                
            except Exception as fallback_error:
                print(f"Fallback method also failed: {fallback_error}")
        
        return []
    
    def detect_field_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect field features (yard lines, hash marks) for camera tracking
        
        Args:
            frame: Input frame
            
        Returns:
            Array of detected feature points
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast to make field lines more visible
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Detect edges using Canny
        edges = cv2.Canny(enhanced, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return np.array([])
        
        # Filter for horizontal lines (yard lines) and vertical lines (hash marks)
        field_features = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line angle
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Keep horizontal lines (yard lines) - within 15 degrees of horizontal
            if (abs(angle) < 15 or abs(angle - 180) < 15 or abs(angle + 180) < 15) and length > 100:
                # Add endpoints as features
                field_features.extend([(x1, y1), (x2, y2)])
            
            # Keep vertical lines (hash marks) - within 15 degrees of vertical
            elif (abs(angle - 90) < 15 or abs(angle + 90) < 15) and length > 30:
                # Add endpoints as features
                field_features.extend([(x1, y1), (x2, y2)])
        
        # Use corner detection for additional features
        corners = cv2.goodFeaturesToTrack(enhanced, maxCorners=100, 
                                         qualityLevel=0.01, minDistance=10)
        
        if corners is not None:
            corners = corners.reshape(-1, 2)
            field_features.extend([(int(x), int(y)) for x, y in corners])
        
        # Remove duplicates and convert to numpy array
        if field_features:
            unique_features = list(set(field_features))
            return np.array(unique_features, dtype=np.float32)
        else:
            return np.array([])
    
    def estimate_frame_homography(self, frame: np.ndarray, reference_features: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate homography for current frame relative to reference frame
        
        Args:
            frame: Current frame
            reference_features: Features from reference frame
            
        Returns:
            Homography matrix or None if estimation fails
        """
        if reference_features.size == 0:
            return None
        
        # Detect features in current frame
        current_features = self.detect_field_features(frame)
        
        if current_features.size == 0 or len(current_features) < 4:
            return None
        
        try:
            # Match features between reference and current frame using FLANN matcher
            # Convert to keypoints for matching
            ref_kp = [cv2.KeyPoint(x, y, 1) for x, y in reference_features]
            cur_kp = [cv2.KeyPoint(x, y, 1) for x, y in current_features]
            
            # Use ORB detector for descriptors
            orb = cv2.ORB_create()
            
            # Compute descriptors
            ref_gray = cv2.cvtColor(self.reference_frame if hasattr(self, 'reference_frame') else frame, cv2.COLOR_BGR2GRAY)
            cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            _, ref_desc = orb.compute(ref_gray, ref_kp)
            _, cur_desc = orb.compute(cur_gray, cur_kp)
            
            if ref_desc is None or cur_desc is None:
                return None
            
            # Match descriptors
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(ref_desc, cur_desc)
            
            if len(matches) < 4:
                return None
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Extract matched points
            ref_points = np.array([reference_features[m.queryIdx] for m in matches[:20]], dtype=np.float32)
            cur_points = np.array([current_features[m.trainIdx] for m in matches[:20]], dtype=np.float32)
            
            # Compute homography
            homography, mask = cv2.findHomography(cur_points, ref_points, 
                                                 cv2.RANSAC, 5.0)
            
            return homography
            
        except Exception as e:
            print(f"Warning: Failed to estimate homography for frame: {e}")
            return None
    
    def track_camera_motion(self):
        """
        Track camera motion throughout the video using field features
        This runs after ball tracking but before 3D position calculation
        """
        if self.homography is None:
            print("No initial calibration available for camera tracking")
            return
        
        print("Phase 2.5: Tracking Camera Motion")
        print("Analyzing field features to detect camera movement...")
        
        # Store reference frame features from calibration
        cap = cv2.VideoCapture(self.input_video_path)
        ret, reference_frame = cap.read()
        
        if ret:
            self.reference_frame = reference_frame
            self.reference_features = self.detect_field_features(reference_frame)
            print(f"Detected {len(self.reference_features)} reference features")
        else:
            print("Failed to read reference frame")
            cap.release()
            return
        
        # Initialize frame-specific camera parameters
        self.frame_homographies = []
        self.frame_camera_matrices = []
        
        frame_count = 0
        successful_tracks = 0
        
        # Process each frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        
        while frame_count < self.total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count == 0:
                # Use original calibration for first frame
                self.frame_homographies.append(self.homography.copy())
                self.frame_camera_matrices.append(self.camera_matrix.copy())
            else:
                # Estimate homography for current frame
                frame_homography = self.estimate_frame_homography(frame, self.reference_features)
                
                if frame_homography is not None:
                    # Combine with reference homography to get field-to-image mapping
                    combined_homography = frame_homography @ self.homography
                    self.frame_homographies.append(combined_homography)
                    
                    # Update camera matrix based on homography change
                    # For simplicity, keep same camera matrix but could be refined
                    self.frame_camera_matrices.append(self.camera_matrix.copy())
                    successful_tracks += 1
                else:
                    # Fallback to previous frame's parameters
                    if self.frame_homographies:
                        self.frame_homographies.append(self.frame_homographies[-1].copy())
                        self.frame_camera_matrices.append(self.frame_camera_matrices[-1].copy())
                    else:
                        self.frame_homographies.append(self.homography.copy())
                        self.frame_camera_matrices.append(self.camera_matrix.copy())
            
            frame_count += 1
            
            # Progress indicator
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{self.total_frames} frames for camera tracking")
        
        cap.release()
        
        success_rate = successful_tracks / max(1, frame_count - 1) * 100
        print(f"Camera tracking complete: {successful_tracks}/{frame_count-1} frames tracked successfully ({success_rate:.1f}%)")
        
        if success_rate < 30:
            print("Warning: Low camera tracking success rate. 3D positions may be less accurate.")
            print("Consider using a video with more stable field features or less camera movement.")
    
    def alternative_3d_analysis(self, image_points_with_time: List[Tuple[Tuple[int, int], float]]) -> List[Position3D]:
        """
        Alternative 3D trajectory analysis without camera calibration
        Uses image-based analysis and physics constraints to estimate 3D trajectory
        
        Args:
            image_points_with_time: List of ((u, v), time_seconds) tuples
            
        Returns:
            List of 3D positions with estimated coordinates
        """
        print("=== ALTERNATIVE 3D TRAJECTORY ANALYSIS ===")
        print("Extracting maximum information from image trajectory...")
        
        # Extract image coordinates and times
        image_points = [point for point, _ in image_points_with_time]
        times = [t for _, t in image_points_with_time]
        
        # 1. ANALYZE IMAGE TRAJECTORY CHARACTERISTICS
        print("\n1. IMAGE TRAJECTORY ANALYSIS:")
        
        # Horizontal motion analysis
        u_coords = [u for u, v in image_points]
        v_coords = [v for u, v in image_points]
        
        u_range = max(u_coords) - min(u_coords)
        v_range = max(v_coords) - min(v_coords)
        
        print(f"   Horizontal motion range: {u_range:.0f} pixels")
        print(f"   Vertical motion range: {v_range:.0f} pixels")
        
        # Find trajectory peak (lowest v-coordinate = highest in image)
        min_v_idx = np.argmin(v_coords)
        peak_time = times[min_v_idx]
        peak_v = v_coords[min_v_idx]
        
        print(f"   Trajectory peak at: t={peak_time:.2f}s, v={peak_v:.0f} pixels")
        print(f"   Peak occurs at {min_v_idx+1}/{len(image_points)} of trajectory")
        
        # 2. ESTIMATE RELATIVE SCALE AND MOTION
        print("\n2. RELATIVE SCALE ESTIMATION:")
        
        # Estimate pixel-to-meter conversion using typical football field dimensions
        # Assume the trajectory spans roughly 20-60 yards horizontally
        estimated_horizontal_distance = 40  # yards, reasonable assumption
        pixels_per_yard = u_range / estimated_horizontal_distance if u_range > 0 else 20
        
        print(f"   Estimated scale: {pixels_per_yard:.1f} pixels/yard")
        print(f"   Horizontal distance: ~{estimated_horizontal_distance} yards")
        
        # 3. PHYSICS-BASED HEIGHT ESTIMATION
        print("\n3. PHYSICS-BASED TRAJECTORY FITTING:")
        
        # Use ballistic physics to estimate trajectory parameters
        total_time = times[-1] - times[0]
        time_to_peak = peak_time - times[0]
        
        # Estimate initial vertical velocity from time to peak
        # At peak: vy = vy0 - g*t_peak = 0, so vy0 = g*t_peak
        g = 9.81  # m/s^2
        estimated_vy0 = g * time_to_peak
        estimated_max_height = (estimated_vy0 ** 2) / (2 * g)
        
        print(f"   Time to peak: {time_to_peak:.2f}s")
        print(f"   Estimated initial vertical velocity: {estimated_vy0:.1f} m/s")
        print(f"   Estimated maximum height: {estimated_max_height:.1f}m ({estimated_max_height * 1.094:.1f} yards)")
        
        # 4. ESTIMATE HORIZONTAL VELOCITIES
        print("\n4. HORIZONTAL MOTION ANALYSIS:")
        
        # Convert pixel motion to estimated real-world motion
        u_start, u_end = u_coords[0], u_coords[-1]
        horizontal_pixels = u_end - u_start
        horizontal_yards = horizontal_pixels / pixels_per_yard
        horizontal_velocity = horizontal_yards / total_time if total_time > 0 else 0
        
        print(f"   Horizontal pixel motion: {horizontal_pixels:.0f} pixels")
        print(f"   Estimated horizontal distance: {horizontal_yards:.1f} yards")
        print(f"   Estimated horizontal velocity: {horizontal_velocity:.1f} yards/s")
        
        # 5. CREATE 3D TRAJECTORY ESTIMATE
        print("\n5. GENERATING 3D TRAJECTORY:")
        
        positions_3d = []
        
        for i, ((u, v), t) in enumerate(image_points_with_time):
            # Time relative to start
            t_rel = t - times[0]
            
            # X (lateral): Convert pixel position to estimated yards
            x_center = (min(u_coords) + max(u_coords)) / 2
            x_yards = (u - x_center) / pixels_per_yard
            
            # Y (height): Use ballistic trajectory
            y_meters = estimated_vy0 * t_rel - 0.5 * g * t_rel * t_rel
            y_yards = max(0, y_meters * 1.094)  # Convert to yards, don't go below ground
            
            # Z (forward): Assume linear forward motion
            z_yards = horizontal_velocity * t_rel
            
            frame_num = int(t * self.fps) if hasattr(self, 'fps') else i
            positions_3d.append(Position3D(x_yards, y_yards, z_yards, frame_num))
        
        # 6. TRAJECTORY QUALITY ASSESSMENT
        print("\n6. TRAJECTORY QUALITY ASSESSMENT:")
        
        heights = [pos.y for pos in positions_3d]
        max_height_calc = max(heights)
        
        # Check if trajectory makes physical sense
        starts_at_ground = heights[0] < 1.0  # Within 1 yard of ground
        ends_at_ground = heights[-1] < 1.0
        has_reasonable_peak = 5 < max_height_calc < 100  # Between 5-100 yards
        
        print(f"   Calculated max height: {max_height_calc:.1f} yards")
        print(f"   Starts near ground: {starts_at_ground}")
        print(f"   Ends near ground: {ends_at_ground}")
        print(f"   Reasonable peak height: {has_reasonable_peak}")
        
        quality_score = sum([starts_at_ground, ends_at_ground, has_reasonable_peak])
        print(f"   Trajectory quality score: {quality_score}/3")
        
        if quality_score >= 2:
            print("   ✓ Trajectory appears physically reasonable")
        else:
            print("   ⚠ Trajectory may have issues - results are rough estimates")
        
        # 7. ADDITIONAL INSIGHTS
        print("\n7. ADDITIONAL TRAJECTORY INSIGHTS:")
        
        # Analyze trajectory shape
        v_motion = np.array(v_coords)
        v_smooth = np.convolve(v_motion, np.ones(3)/3, mode='same')  # Simple smoothing
        
        # Find if trajectory is symmetric
        first_half = v_smooth[:len(v_smooth)//2]
        second_half = v_smooth[len(v_smooth)//2:]
        
        if len(first_half) > 0 and len(second_half) > 0:
            asymmetry = abs(np.mean(np.diff(first_half)) + np.mean(np.diff(second_half)))
            print(f"   Trajectory asymmetry measure: {asymmetry:.1f}")
            
            if asymmetry < 5:
                print("   ✓ Trajectory appears symmetric (good ballistic fit)")
            else:
                print("   ⚠ Trajectory appears asymmetric (may indicate tracking issues)")
        
        # Estimate launch angle
        if len(positions_3d) > 1:
            initial_horizontal = abs(positions_3d[1].z - positions_3d[0].z)
            initial_vertical = abs(positions_3d[1].y - positions_3d[0].y)
            
            if initial_horizontal > 0:
                launch_angle = np.degrees(np.arctan(initial_vertical / initial_horizontal))
                print(f"   Estimated launch angle: {launch_angle:.1f} degrees")
                
                if 20 < launch_angle < 60:
                    print("   ✓ Launch angle in reasonable range for football")
                else:
                    print("   ⚠ Launch angle seems unusual for football trajectory")
        
        print("\n=== ALTERNATIVE ANALYSIS COMPLETE ===")
        print(f"Generated {len(positions_3d)} 3D position estimates")
        print("Note: These are estimates based on image analysis and physics.")
        print("Accuracy depends on camera angle and trajectory assumptions.")
        
        return positions_3d
    
    def analyze_trajectory(self):
        """Analyze trajectory data to find crossing points and prepare for visualization"""
        if not self.positions_3d:
            return
        
        print("Analyzing trajectory for crossing points...")
        
        # Extract Y (forward/backward) and Z (height) values
        self.trajectory_data = [(pos.z, pos.y) for pos in self.positions_3d]  # (forward, height)
        
        # Find crossing point where Z (height) = 3.33 yards on downward swing
        self.find_height_crossing_point()
    
    def find_height_crossing_point(self):
        """Find Y position when Z=3.33 yards on the downward swing"""
        target_height = 3.33  # yards
        
        if len(self.positions_3d) < 3:
            print("Insufficient trajectory data for crossing point analysis")
            return
        
        # Find the peak (maximum height)
        heights = [pos.y for pos in self.positions_3d]
        max_height_idx = np.argmax(heights)
        max_height = heights[max_height_idx]
        
        print(f"Peak height: {max_height:.1f} yards at frame {self.positions_3d[max_height_idx].frame_num}")
        
        if max_height < target_height:
            print(f"Peak height ({max_height:.1f}yd) is below target height ({target_height}yd)")
            return
        
        # Look for crossing point on downward swing (after peak)
        for i in range(max_height_idx + 1, len(self.positions_3d) - 1):
            current_height = self.positions_3d[i].y
            next_height = self.positions_3d[i + 1].y
            
            # Check if we cross the target height between these two points
            if current_height >= target_height >= next_height:
                # Linear interpolation to find exact crossing point
                current_y = self.positions_3d[i].z  # forward/backward position
                next_y = self.positions_3d[i + 1].z
                
                # Interpolation factor
                t = (target_height - next_height) / (current_height - next_height)
                crossing_y = next_y + t * (current_y - next_y)
                
                self.crossing_point_y = crossing_y
                crossing_frame = self.positions_3d[i].frame_num + t * (self.positions_3d[i + 1].frame_num - self.positions_3d[i].frame_num)
                
                print(f"Crossing point found: Y={crossing_y:.2f} yards when Z={target_height} yards")
                print(f"Occurs at approximately frame {crossing_frame:.1f}")
                return
        
        print(f"No crossing point found at {target_height} yards on downward swing")
    
    def detect_yard_lines(self, frame: np.ndarray) -> List[Tuple[float, List[Tuple[int, int]]]]:
        """
        Detect yard lines in the frame and estimate their positions
        
        Args:
            frame: Input frame
            
        Returns:
            List of (estimated_yard_position, line_points) tuples
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast to make field lines more visible
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Detect edges using Canny
        edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, 
                               minLineLength=100, maxLineGap=20)
        
        if lines is None:
            return []
        
        # Filter for horizontal lines (yard lines)
        yard_lines = []
        frame_height, frame_width = frame.shape[:2]
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line angle
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Keep horizontal lines (yard lines) - within 10 degrees of horizontal
            if (abs(angle) < 10 or abs(angle - 180) < 10 or abs(angle + 180) < 10) and length > frame_width * 0.3:
                # Calculate average y position of the line
                avg_y = (y1 + y2) / 2
                
                # Estimate yard position based on perspective
                # Lines closer to bottom of frame are closer to camera (lower yard numbers)
                # This is a rough estimation - in reality would need proper calibration
                relative_position = (frame_height - avg_y) / frame_height
                
                # Assume we can see roughly from goal line (0) to 30-40 yard line
                estimated_yard = relative_position * 40  # Rough estimate
                
                yard_lines.append((estimated_yard, [(x1, y1), (x2, y2)]))
        
        # Sort by estimated yard position
        yard_lines.sort(key=lambda x: x[0])
        
        return yard_lines
    
    def calculate_position_from_point(self, point: Annotation, point_name: str) -> Optional[float]:
        """
        Calculate the downfield position of a given point using CV-based field line detection
        
        Args:
            point: The annotation point to analyze
            point_name: Name for logging (e.g., "end point", "beginning point")
            
        Returns:
            Estimated downfield distance in yards, or None if calculation fails
        """
        if point is None:
            print(f"No {point_name} set for distance calculation")
            return None
        
        print(f"\n=== CALCULATING {point_name.upper()} POSITION ===")
        print(f"Using {point_name} at frame {point.frame_num}, position {point.center}")
        if point_name == "end point":
            print("End point is treated as manual calibration at height=0")
        
        # Get the frame containing the point
        cap = cv2.VideoCapture(self.input_video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, point.frame_num)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Failed to read frame for distance calculation")
            return None
        
        # Detect yard lines in the frame
        yard_lines = self.detect_yard_lines(frame)
        
        if not yard_lines:
            print("No yard lines detected in frame")
            return None
        
        print(f"Detected {len(yard_lines)} potential yard lines")
        
        # Find the yard line closest to the point
        point_x, point_y = point.center
        closest_line = None
        min_distance = float('inf')
        
        for yard_pos, line_points in yard_lines:
            # Calculate distance from end point to this line
            x1, y1 = line_points[0]
            x2, y2 = line_points[1]
            
            # Distance from point to line
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            distance = abs(A * point_x + B * point_y + C) / np.sqrt(A*A + B*B)
            
            if distance < min_distance:
                min_distance = distance
                closest_line = (yard_pos, line_points)
        
        if closest_line is None:
            print("Could not find closest yard line")
            return None
        
        closest_yard, closest_points = closest_line
        print(f"Closest yard line estimated at {closest_yard:.1f} yards, distance: {min_distance:.1f} pixels")
        
        # Use perspective analysis to refine the estimate
        frame_height = frame.shape[0]
        
        # The point's y-coordinate gives us perspective information
        # Points higher in the frame (lower y values) are further downfield
        perspective_factor = (frame_height - point_y) / frame_height
        
        # Estimate based on typical football field perspective
        # Assume we can see from goal line to about 40-50 yard line
        max_visible_distance = 45  # yards
        estimated_distance = perspective_factor * max_visible_distance
        
        # Refine using the closest yard line as reference
        if min_distance < 50:  # If end point is close to a detected line
            # Use the detected line position as primary estimate
            refined_distance = closest_yard
        else:
            # Use perspective-based estimate
            refined_distance = estimated_distance
        
        # Apply calibration adjustment based on point type
        if point_name == "end point":
            # For end point (landing), typically past the goal line (negative territory)
            # Since this is a manual calibration point at height=0, we can be more confident
            if refined_distance < 0:
                refined_distance = abs(refined_distance)  # Convert to positive distance
            final_distance = -refined_distance if refined_distance > 0 else refined_distance
            print(f"Calculated downfield distance: {final_distance:.1f} yards")
            print(f"(Negative values indicate distance past the goal line)")
        else:
            # For beginning point (kick position), typically positive yard line
            # Adjust based on typical field goal kick positions
            if refined_distance < 0:
                refined_distance = abs(refined_distance)  # Convert to positive
            
            # Starting positions are typically between 15-40 yard line
            if refined_distance > 50:
                refined_distance = 50  # Cap at reasonable maximum
            elif refined_distance < 10:
                refined_distance = 15  # Minimum reasonable kick distance
            
            final_distance = refined_distance
            print(f"Calculated starting position: {final_distance:.1f} yard line")
        
        return final_distance
    
    def calculate_downfield_distance(self) -> Optional[float]:
        """
        Calculate the downfield distance of the end point using CV-based field line detection
        Uses the end point as a manual calibration point at height=0
        
        Returns:
            Estimated downfield distance in yards, or None if calculation fails
        """
        result = self.calculate_position_from_point(self.end_point, "end point")
        if result is not None:
            self.calculated_distance = result
        return result
    
    def calculate_starting_position(self) -> Optional[float]:
        """
        Calculate the starting position of the beginning point using CV-based field line detection
        
        Returns:
            Estimated starting yard line position, or None if calculation fails
        """
        result = self.calculate_position_from_point(self.beginning_point, "beginning point")
        if result is not None:
            self.starting_position = result
        return result
    
    def create_trajectory_graph(self, current_frame: int, graph_width: int = 400, graph_height: int = 300) -> np.ndarray:
        """
        Create a real-time trajectory graph showing Y vs Z
        
        Args:
            current_frame: Current frame number for highlighting current position
            graph_width: Width of the graph in pixels
            graph_height: Height of the graph in pixels
            
        Returns:
            Graph image as numpy array
        """
        try:
            # Create matplotlib figure
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(graph_width/100, graph_height/100), dpi=100)
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            
            if not self.trajectory_data:
                # Empty graph
                ax.text(0.5, 0.5, 'No Trajectory Data', transform=ax.transAxes, 
                       ha='center', va='center', color='white', fontsize=12)
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 20)
            else:
                # Extract data up to current frame
                current_data = []
                for pos in self.positions_3d:
                    if pos.frame_num <= current_frame:
                        current_data.append((pos.z, pos.y))  # (forward, height)
                
                if current_data:
                    y_vals, z_vals = zip(*current_data)
                    
                    # Plot trajectory
                    ax.plot(y_vals, z_vals, 'cyan', linewidth=2, alpha=0.8, label='Trajectory')
                    
                    # Highlight current position
                    if current_data:
                        ax.plot(y_vals[-1], z_vals[-1], 'red', marker='o', markersize=8, label='Current')
                    
                    # Add target height line
                    y_range = ax.get_xlim()
                    ax.axhline(y=3.33, color='yellow', linestyle='--', alpha=0.7, label='Target Height (3.33yd)')
                    
                    # Add crossing point if found
                    if self.crossing_point_y is not None:
                        ax.plot(self.crossing_point_y, 3.33, 'lime', marker='*', markersize=12, 
                               label=f'Crossing: {self.crossing_point_y:.1f}yd')
                    
                    # Set labels and limits
                    ax.set_xlabel('Forward Distance (yards)', color='white')
                    ax.set_ylabel('Height (yards)', color='white')
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='upper right', fontsize=8)
                    
                    # Set reasonable axis limits
                    y_min, y_max = min(y_vals), max(y_vals)
                    z_min, z_max = min(z_vals), max(z_vals)
                    ax.set_xlim(y_min - 5, y_max + 5)
                    ax.set_ylim(max(0, z_min - 2), z_max + 2)
            
            ax.set_title('Ball Trajectory (Forward vs Height)', color='white', fontsize=10)
            ax.tick_params(colors='white')
            
            # Convert to image
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            
            # Use the newer buffer_rgba method
            try:
                # Try newer method first
                buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
                buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
                # Convert RGBA to RGB
                buf = buf[:, :, :3]
            except AttributeError:
                try:
                    # Fallback to older method
                    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
                    buf = buf.reshape(canvas.get_width_height()[::-1] + (3,))
                except AttributeError:
                    # Final fallback
                    buf = np.array(canvas.renderer.buffer_rgba())
                    buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
                    buf = buf[:, :, :3]
            
            plt.close(fig)
        
            # Convert RGB to BGR for OpenCV
            return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            print(f"Warning: Failed to create trajectory graph: {e}")
            # Return a black image as fallback
            return np.zeros((graph_height, graph_width, 3), dtype=np.uint8)
    
    def refine_annotation(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Refine user annotation using color-based detection
        
        Args:
            frame: Input frame
            bbox: User-provided bounding box (x, y, w, h)
            
        Returns:
            Refined bounding box (x, y, w, h)
        """
        x, y, w, h = bbox
        
        # Ensure bbox is within frame bounds
        x = max(0, min(x, frame.shape[1] - w))
        y = max(0, min(y, frame.shape[0] - h))
        
        # Extract ROI
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return bbox
        
        # Convert to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Create mask for football color
        mask = cv2.inRange(hsv_roi, self.lower_hsv, self.upper_hsv)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find best contour based on area and aspect ratio
            best_contour = None
            best_score = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 10:  # Skip very small contours
                    continue
                
                # Get bounding rectangle
                rect = cv2.boundingRect(contour)
                rect_w, rect_h = rect[2], rect[3]
                
                # Score based on area and aspect ratio (football should be roughly round)
                aspect_ratio = min(rect_w, rect_h) / max(rect_w, rect_h) if max(rect_w, rect_h) > 0 else 0
                score = area * aspect_ratio
                
                if score > best_score:
                    best_score = score
                    best_contour = contour
            
            if best_contour is not None:
                # Get refined bounding box
                rect = cv2.boundingRect(best_contour)
                refined_bbox = (x + rect[0], y + rect[1], rect[2], rect[3])
                return refined_bbox
        
        # If no good contour found, return original bbox
        return bbox
    
    def interpolate_position(self, frame_idx: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Linear interpolation between annotations for missing frames
        Only interpolates between two existing annotations, never after the last annotation
        
        Args:
            frame_idx: Current frame index
            
        Returns:
            Interpolated bounding box or None if not possible
        """
        # Sort annotations by frame number
        sorted_annotations = sorted(self.annotations, key=lambda x: x.frame_num)
        
        # Find surrounding annotations
        prev_annotation = None
        next_annotation = None
        
        for annotation in sorted_annotations:
            if annotation.frame_num < frame_idx:
                prev_annotation = annotation
            elif annotation.frame_num > frame_idx and next_annotation is None:
                next_annotation = annotation
                break
        
        # Only interpolate if we have both previous and next annotations
        # This ensures we never guess frames after the last annotation
        if prev_annotation is None or next_annotation is None:
            return None
        
        # Linear interpolation
        t = (frame_idx - prev_annotation.frame_num) / (next_annotation.frame_num - prev_annotation.frame_num)
        
        prev_center = prev_annotation.center
        next_center = next_annotation.center
        
        # Interpolate center position
        interp_center = (
            int(prev_center[0] + t * (next_center[0] - prev_center[0])),
            int(prev_center[1] + t * (next_center[1] - prev_center[1]))
        )
        
        # Use average size
        prev_size = prev_annotation.bbox[2]
        next_size = next_annotation.bbox[2]
        interp_size = int((prev_size + next_size) / 2)
        
        return (interp_center[0] - interp_size//2, interp_center[1] - interp_size//2, 
                interp_size, interp_size)
    
    def phase2_tracking(self):
        """Phase 2: Automated refinement and tracking"""
        print("Phase 2: Automated Tracking and Refinement")
        
        self.cap = cv2.VideoCapture(self.input_video_path)
        frame_count = 0
        last_refined_bbox = None
        
        # Initialize final bboxes list
        self.final_bboxes = [None] * self.total_frames
        
        while frame_count < self.total_frames:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Check if this frame has user annotation
            annotation = None
            for ann in self.annotations:
                if ann.frame_num == frame_count:
                    annotation = ann
                    break
            
            if annotation is not None:
                # Refine the user annotation
                refined_bbox = self.refine_annotation(frame, annotation.bbox)
                self.final_bboxes[frame_count] = refined_bbox
                last_refined_bbox = refined_bbox
                
                # Update Kalman filter
                center = (refined_bbox[0] + refined_bbox[2]//2, refined_bbox[1] + refined_bbox[3]//2)
                if not self.kalman_filter.initialized:
                    self.kalman_filter.initialize(center)
                else:
                    self.kalman_filter.update(center)
                
                print(f"Frame {frame_count}: Refined annotation at {center}")
            
            else:
                # No annotation - use tracking or interpolation
                if last_refined_bbox is not None:
                    # Try interpolation first (only between annotations)
                    interpolated_bbox = self.interpolate_position(frame_count)
                    
                    if interpolated_bbox is not None:
                        # Use interpolated position
                        self.final_bboxes[frame_count] = interpolated_bbox
                        center = (interpolated_bbox[0] + interpolated_bbox[2]//2, 
                                interpolated_bbox[1] + interpolated_bbox[3]//2)
                        self.kalman_filter.update(center)
                        print(f"Frame {frame_count}: Using interpolated position at {center}")
                    
                    else:
                        # No interpolation possible - try CSRT tracking
                        if self.csrt_tracker is None:
                            self.csrt_tracker = cv2.TrackerCSRT_create()
                            self.csrt_tracker.init(frame, last_refined_bbox)
                        
                        success, tracked_bbox = self.csrt_tracker.update(frame)
                        
                        if success:
                            # Validate tracked bbox (check if it's reasonable)
                            bbox_center = (tracked_bbox[0] + tracked_bbox[2]//2, tracked_bbox[1] + tracked_bbox[3]//2)
                            
                            # Use Kalman filter to smooth the prediction
                            kalman_center = self.kalman_filter.predict()
                            
                            # Blend tracker and Kalman predictions
                            blend_factor = 0.7  # More weight to Kalman filter
                            final_center = (
                                int(blend_factor * kalman_center[0] + (1 - blend_factor) * bbox_center[0]),
                                int(blend_factor * kalman_center[1] + (1 - blend_factor) * bbox_center[1])
                            )
                            
                            # Update Kalman filter
                            self.kalman_filter.update(final_center)
                            
                            # Create final bbox with original size
                            bbox_size = last_refined_bbox[2]
                            final_bbox = (final_center[0] - bbox_size//2, final_center[1] - bbox_size//2, 
                                        bbox_size, bbox_size)
                            
                            self.final_bboxes[frame_count] = final_bbox
                            print(f"Frame {frame_count}: Using CSRT tracking at {final_center}")
                        else:
                            # Tracking failed - check if we're after the last annotation
                            last_annotation_frame = max([ann.frame_num for ann in self.annotations]) if self.annotations else -1
                            
                            if frame_count > last_annotation_frame:
                                # After last annotation - don't guess, set to None
                                self.final_bboxes[frame_count] = None
                                print(f"Frame {frame_count}: After last annotation, no tracking")
                            else:
                                # Use last known position
                                self.final_bboxes[frame_count] = last_refined_bbox
                                print(f"Frame {frame_count}: Using last known position")
                
                else:
                    # No previous bbox available
                    self.final_bboxes[frame_count] = None
            
            frame_count += 1
        
        self.cap.release()
        print(f"Tracking complete. Processed {frame_count} frames.")
        
        # Track camera motion using field features
        if self.homography is not None:
            self.track_camera_motion()
        
        # Compute 3D positions using ballistic trajectory fitting
        print("Computing 3D trajectory...")
        self.compute_3d_trajectory()
        
        # Analyze trajectory for crossing points
        if self.positions_3d:
            self.analyze_trajectory()
        
        # Calculate positions using CV-based field line detection
        print("\n=== POSITION ANALYSIS USING CV FIELD LINE DETECTION ===")
        
        # Calculate starting position
        if self.beginning_point is not None:
            starting_position = self.calculate_starting_position()
            if starting_position is not None:
                print(f"\nStarting position: {starting_position:.1f} yard line")
            else:
                print("Starting position calculation failed")
        else:
            print("\nNo beginning point marked - skipping starting position calculation")
            print("Use 'g' key during annotation to mark a beginning point")
        
        # Calculate ending position using end point as manual calibration
        if self.end_point is not None:
            calculated_distance = self.calculate_downfield_distance()
            if calculated_distance is not None:
                print(f"\nFinal calculated distance: {calculated_distance:.1f} yards from goal line")
                if calculated_distance < 0:
                    print(f"Ball landed {abs(calculated_distance):.1f} yards PAST the goal line")
                else:
                    print(f"Ball landed {calculated_distance:.1f} yards SHORT of the goal line")
            else:
                print("Distance calculation failed")
        else:
            print("\nNo end point marked - skipping distance calculation")
            print("Use 'h' key during annotation to mark an end point for distance calculation")
        
        # Calculate total distance if both points are available
        if self.starting_position is not None and self.calculated_distance is not None:
            if self.calculated_distance < 0:  # Ball went past goal line
                total_distance = self.starting_position + abs(self.calculated_distance)
                print(f"\n*** TOTAL KICK DISTANCE: {total_distance:.1f} yards ***")
                print(f"From {self.starting_position:.1f} yard line to {abs(self.calculated_distance):.1f} yards past goal line")
            else:  # Ball fell short
                total_distance = self.starting_position - self.calculated_distance
                print(f"\n*** TOTAL KICK DISTANCE: {total_distance:.1f} yards ***")
                print(f"From {self.starting_position:.1f} yard line to {self.calculated_distance:.1f} yard line")
    
    def compute_3d_trajectory(self):
        """Compute 3D positions for all frames with valid bounding boxes"""
        # Always attempt 3D computation - use alternative methods if no calibration
        
        # Collect image points with time for trajectory fitting
        image_points_with_time = []
        
        for frame_idx, bbox in enumerate(self.final_bboxes):
            if bbox is not None:
                center = (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2)
                time_seconds = frame_idx / self.fps
                image_points_with_time.append((center, time_seconds))
        
        if len(image_points_with_time) < 3:
            print("Insufficient points for 3D trajectory fitting")
            return
        
        # Fit ballistic trajectory
        positions_3d = self.image_to_3d_ballistic(image_points_with_time)
        
        if positions_3d:
            self.positions_3d = positions_3d
            print(f"Successfully computed 3D trajectory with {len(positions_3d)} points")
            
            # Print some sample positions for verification
            for i in range(min(5, len(positions_3d))):
                pos = positions_3d[i]
                print(f"Frame {pos.frame_num}: ({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f}) yards")
        else:
            print("3D trajectory fitting failed")
    
    def phase3_visualization(self):
        """Phase 3: Video generation with trajectory visualization"""
        print("Phase 3: Generating Output Video")
        
        # Calculate trajectory points
        self.trajectory_points = []
        for bbox in self.final_bboxes:
            if bbox is not None:
                center = (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2)
                self.trajectory_points.append(center)
            else:
                self.trajectory_points.append(None)
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, 
                            (self.frame_width, self.frame_height))
        
        self.cap = cv2.VideoCapture(self.input_video_path)
        frame_count = 0
        
        while frame_count < self.total_frames:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Draw bounding box if available
            if self.final_bboxes[frame_count] is not None:
                bbox = self.final_bboxes[frame_count]
                
                # Determine color based on point type
                point_type = "regular"
                if self.beginning_point and self.beginning_point.frame_num == frame_count:
                    point_type = "beginning"
                elif self.end_point and self.end_point.frame_num == frame_count:
                    point_type = "end"
                
                # Choose color based on point type
                if point_type == "beginning":
                    box_color = (0, 255, 0)  # Green
                elif point_type == "end":
                    box_color = (0, 0, 255)  # Red
                else:
                    box_color = (0, 255, 0)  # Default green
                
                cv2.rectangle(frame, (bbox[0], bbox[1]), 
                            (bbox[0] + bbox[2], bbox[1] + bbox[3]), box_color, 2)
                
                # Draw center point
                center = (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2)
                cv2.circle(frame, center, 3, (0, 0, 255), -1)
                
                # Add label for special points
                if point_type == "beginning":
                    cv2.putText(frame, "BEGIN", (bbox[0], bbox[1] - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                elif point_type == "end":
                    cv2.putText(frame, "END (Cal.)", (bbox[0], bbox[1] - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Draw 3D position if available
                if self.positions_3d:
                    # Find the closest 3D position for this frame
                    closest_pos = None
                    min_frame_diff = float('inf')
                    
                    for pos_3d in self.positions_3d:
                        frame_diff = abs(pos_3d.frame_num - frame_count)
                        if frame_diff < min_frame_diff:
                            min_frame_diff = frame_diff
                            closest_pos = pos_3d
                    
                    if closest_pos and min_frame_diff <= 2:  # Within 2 frames
                        # Display 3D coordinates next to the ball
                        text_x = center[0] + 20
                        text_y = center[1] - 10
                        
                        # Position text - focus on height and downfield position
                        pos_text = f"Height:{closest_pos.y:.1f}yd  Downfield:{closest_pos.z:.1f}yd"
                        cv2.putText(frame, pos_text, (text_x, text_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                        
                        # Add lateral position as relative info
                        cv2.putText(frame, f"Lateral:{closest_pos.x:+.1f}yd", (text_x, text_y + 15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
            
            # Draw trajectory
            valid_points = [p for p in self.trajectory_points[:frame_count+1] if p is not None]
            if len(valid_points) > 1:
                # Draw trajectory line
                for i in range(1, len(valid_points)):
                    cv2.line(frame, valid_points[i-1], valid_points[i], (255, 0, 0), 2)
                
                # Draw trajectory points
                for point in valid_points:
                    cv2.circle(frame, point, 2, (255, 0, 0), -1)
            
            # Add frame information
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add trajectory graph if 3D data is available
            if self.positions_3d:
                graph = self.create_trajectory_graph(frame_count, 400, 300)
                
                # Position graph in top-right corner
                graph_h, graph_w = graph.shape[:2]
                frame_h, frame_w = frame.shape[:2]
                
                # Ensure graph fits in frame
                if graph_w <= frame_w and graph_h <= frame_h:
                    x_offset = frame_w - graph_w - 10
                    y_offset = 10
                    
                    # Overlay graph on frame
                    frame[y_offset:y_offset+graph_h, x_offset:x_offset+graph_w] = graph
                
                # Add crossing point information
                info_y_pos = frame_h - 80
                
                if self.crossing_point_y is not None:
                    crossing_text = f"Crossing at 3.33yd height: {self.crossing_point_y:.2f}yd forward"
                    cv2.putText(frame, crossing_text, (10, info_y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # Add background rectangle for better visibility
                    text_size = cv2.getTextSize(crossing_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (5, info_y_pos - 20), (text_size[0] + 15, info_y_pos + 10), (0, 0, 0), -1)
                    cv2.putText(frame, crossing_text, (10, info_y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    info_y_pos += 30
                
                # Add starting position information
                if self.starting_position is not None:
                    start_text = f"Start: {self.starting_position:.1f} yard line"
                    text_color = (255, 255, 0)  # Cyan for starting position
                    
                    # Add background rectangle
                    text_size = cv2.getTextSize(start_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (5, info_y_pos - 20), (text_size[0] + 15, info_y_pos + 5), (0, 0, 0), -1)
                    cv2.putText(frame, start_text, (10, info_y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
                    info_y_pos += 30
                
                # Add calculated distance information
                if self.calculated_distance is not None:
                    if self.calculated_distance < 0:
                        distance_text = f"End: {abs(self.calculated_distance):.1f}yd PAST goal line"
                        text_color = (0, 255, 255)  # Yellow for past goal line
                    else:
                        distance_text = f"End: {self.calculated_distance:.1f}yd SHORT of goal line"
                        text_color = (0, 165, 255)  # Orange for short
                    
                    # Add background rectangle
                    text_size = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (5, info_y_pos - 20), (text_size[0] + 15, info_y_pos + 5), (0, 0, 0), -1)
                    cv2.putText(frame, distance_text, (10, info_y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
                    info_y_pos += 25
                    
                    # Add total distance if both points available
                    if self.starting_position is not None:
                        if self.calculated_distance < 0:
                            total_dist = self.starting_position + abs(self.calculated_distance)
                        else:
                            total_dist = self.starting_position - self.calculated_distance
                        
                        total_text = f"Total: {total_dist:.1f} yards"
                        cv2.putText(frame, total_text, (10, info_y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                        info_y_pos += 20
                    
                    # Add calibration note
                    cal_text = "(CV field line detection)"
                    cv2.putText(frame, cal_text, (10, info_y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            
            # Write frame to output video
            out.write(frame)
            frame_count += 1
        
        # Clean up
        self.cap.release()
        out.release()
        
        print(f"Output video saved as: {self.output_video_path}")
    
    def run(self):
        """Run the complete tracking pipeline"""
        try:
            print("Starting Semi-Automatic Football Tracking with 3D Analysis")
            print("=" * 50)
            
            # Phase 0: Field Calibration
            self.phase0_field_calibration()
            
            # Phase 1: User Annotation
            self.phase1_annotation()
            
            if not self.annotations:
                print("No annotations provided. Exiting.")
                return
            
            # Phase 2: Automated Tracking
            self.phase2_tracking()
            
            # Phase 3: Visualization
            self.phase3_visualization()
            
            print("=" * 50)
            print("Tracking pipeline completed successfully!")
            print(f"Total annotations: {len(self.annotations)}")
            
            # Summary of special points
            if self.beginning_point:
                print(f"Beginning point: Frame {self.beginning_point.frame_num} at {self.beginning_point.center}")
            if self.end_point:
                print(f"End point: Frame {self.end_point.frame_num} at {self.end_point.center}")
            
            # Position analysis results
            print(f"\n*** FIELD POSITION ANALYSIS RESULTS ***")
            
            if self.starting_position is not None:
                print(f"Starting position: {self.starting_position:.1f} yard line")
            else:
                print("Starting position: Not calculated (no beginning point marked)")
            
            if self.calculated_distance is not None:
                if self.calculated_distance < 0:
                    print(f"Ending position: {abs(self.calculated_distance):.1f} yards PAST the goal line")
                    print("Result: FIELD GOAL SUCCESSFUL! ✓")
                else:
                    print(f"Ending position: {self.calculated_distance:.1f} yards SHORT of the goal line")
                    print("Result: Field goal attempt was short ✗")
            else:
                print("Ending position: Not calculated (no end point marked)")
            
            # Total distance calculation
            if self.starting_position is not None and self.calculated_distance is not None:
                if self.calculated_distance < 0:
                    total_distance = self.starting_position + abs(self.calculated_distance)
                    print(f"\n🏈 TOTAL KICK DISTANCE: {total_distance:.1f} YARDS 🏈")
                    print(f"   From {self.starting_position:.1f}yd line → {abs(self.calculated_distance):.1f}yd past goal")
                else:
                    total_distance = self.starting_position - self.calculated_distance
                    print(f"\n🏈 TOTAL KICK DISTANCE: {total_distance:.1f} YARDS 🏈")
                    print(f"   From {self.starting_position:.1f}yd line → {self.calculated_distance:.1f}yd line")
            
            print("\n(Analysis based on CV field line detection with manual calibration)")
            
            print(f"\nOutput video: {self.output_video_path}")
            
        except Exception as e:
            print(f"Error during tracking: {str(e)}")
            raise
        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Semi-Automatic Football Tracking')
    parser.add_argument('--input', '-i', default='field_goal.mp4', 
                       help='Input video file path (default: field_goal.mp4)')
    parser.add_argument('--output', '-o', default='output.mp4', 
                       help='Output video file path (default: output.mp4)')
    parser.add_argument('--bbox-size', type=int, default=30, 
                       help='Bounding box size for annotation (default: 30)')
    parser.add_argument('--skip-frames', type=int, default=5, 
                       help='Number of frames to skip between annotations (default: 5)')
    
    args = parser.parse_args()
    
    # Check if OpenCV is available
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
    except ImportError:
        print("Error: opencv-python is required. Install it with: pip install opencv-python")
        sys.exit(1)
    
    # Create and run tracker
    tracker = FootballTracker(
        input_video_path=args.input,
        output_video_path=args.output,
        bbox_size=args.bbox_size,
        skip_frames=args.skip_frames
    )
    
    tracker.run()


if __name__ == "__main__":
    main()
