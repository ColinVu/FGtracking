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


@dataclass
class Annotation:
    """Data structure to store user annotations"""
    frame_num: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[int, int]  # (cx, cy)


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
        
        # Camera calibration
        self.homography = None
        self.camera_matrix = None
        self.rotation_matrix = None
        self.translation_vector = None
        
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
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for annotation selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_pos = (x, y)
            self.annotation_ready = True
    
    def draw_annotation_box(self, frame: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
        """Draw annotation bounding box on frame"""
        x, y = center
        half_size = self.bbox_size // 2
        
        # Ensure box stays within frame bounds
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(frame.shape[1], x + half_size)
        y2 = min(frame.shape[0], y + half_size)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, center, 3, (0, 255, 0), -1)
        
        return frame
    
    def phase1_annotation(self):
        """Phase 1: Interactive user annotation with full frame navigation"""
        print("Phase 1: Interactive Annotation")
        print("Instructions:")
        print("- Click on the football to annotate it")
        print("- Press 's' to skip current frame")
        print("- Press 'q' to finish annotation and proceed to Phase 2")
        print("- Use keyboard keys to navigate frames:")
        print("  A/D: Move one frame (left/right)")
        print("  W/X: Move 10 frames (forward/back)")
        print("  HOME: Go to first frame")
        print("  END: Go to last frame")
        print("- Press SPACE to pause/resume navigation")
        
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
                self.draw_annotation_box(display_frame, center)
                cv2.putText(display_frame, f"ANNOTATED at {center}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "Click on the football to annotate", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Show navigation instructions
            cv2.putText(display_frame, "A/D: 1 frame | W/X: 10 frames | SPACE: Pause | 's': Skip | 'q': Finish", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Total annotations: {len(self.annotations)}", 
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
                    self.annotations = [ann for ann in self.annotations if ann.frame_num != current_frame]
                    print(f"Removed annotation from frame {current_frame}")
            
            elif self.annotation_ready and self.mouse_pos and not has_annotation:
                # Create new annotation
                center = self.mouse_pos
                bbox = (center[0] - self.bbox_size//2, center[1] - self.bbox_size//2, 
                        self.bbox_size, self.bbox_size)
                
                annotation = Annotation(current_frame, bbox, center)
                self.annotations.append(annotation)
                
                print(f"Annotated frame {current_frame} at position {center}")
                self.annotation_ready = False
                
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
        Convert image points to 3D using ballistic trajectory fitting
        
        Args:
            image_points_with_time: List of ((u, v), time_seconds) tuples
            
        Returns:
            List of 3D positions
        """
        if self.homography is None or len(image_points_with_time) < 3:
            return []
        
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
            """Residual function for optimization"""
            times = [t for _, t in image_points_with_time]
            predicted_3d = ballistic_model(params, times)
            
            residuals = []
            for i, ((u, v), _) in enumerate(image_points_with_time):
                # Project 3D point to image
                point_3d = predicted_3d[i]
                
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
        
        # Initial guess: use homography to get ground plane positions, assume reasonable height
        initial_ground_points = []
        times = []
        
        for (u, v), t in image_points_with_time:
            try:
                # Project to ground plane using homography
                point_img = np.array([u, v, 1], dtype=np.float32)
                point_ground = self.homography @ point_img
                
                # Check for valid homography result
                if abs(point_ground[2]) > 1e-6:
                    point_ground = point_ground / point_ground[2]  # Normalize
                    # Reasonable initial height for a football (1-5 meters)
                    initial_height = 3.0  # meters (~3.3 yards)
                    initial_ground_points.append([point_ground[0], initial_height, point_ground[1]])
                else:
                    # Fallback if homography gives bad result
                    initial_ground_points.append([0, 3.0, 10])
                    
                times.append(t)
            except Exception as e:
                print(f"Warning: Failed to project point {(u, v)}: {e}")
                # Use fallback position
                initial_ground_points.append([0, 3.0, 10])
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
                # Clamp velocity to reasonable ranges
                velocity_estimate[0] = np.clip(velocity_estimate[0], -20, 20)  # lateral velocity
                velocity_estimate[1] = np.clip(velocity_estimate[1], -5, 15)   # vertical velocity  
                velocity_estimate[2] = np.clip(velocity_estimate[2], -10, 30)  # forward velocity
                initial_velocity = velocity_estimate
            else:
                initial_velocity = np.array([0, 5, 15])  # Default reasonable velocity
        else:
            initial_velocity = np.array([0, 5, 15])  # Default reasonable velocity
        
        # Initial parameters: [x0, y0, z0, vx0, vy0, vz0]
        initial_params = np.concatenate([initial_ground_points[0], initial_velocity])
        
        # Make bounds more flexible and ensure initial guess is within bounds
        # Clamp initial parameters to be within bounds
        lower_bounds = [-100, 0, -50, -50, -10, -50]  # More flexible bounds
        upper_bounds = [100, 50, 200, 50, 30, 100]
        
        # Ensure initial guess is within bounds
        initial_params_clamped = np.clip(initial_params, lower_bounds, upper_bounds)
        
        print(f"Initial trajectory parameters: {initial_params_clamped}")
        print(f"Position: ({initial_params_clamped[0]:.1f}, {initial_params_clamped[1]:.1f}, {initial_params_clamped[2]:.1f}) m")
        print(f"Velocity: ({initial_params_clamped[3]:.1f}, {initial_params_clamped[4]:.1f}, {initial_params_clamped[5]:.1f}) m/s")
        
        # Optimize
        try:
            result = least_squares(residual_function, initial_params_clamped, 
                                 bounds=(lower_bounds, upper_bounds))
            
            if result.success:
                # Generate 3D positions for all time points
                optimized_params = result.x
                times_all = [t for _, t in image_points_with_time]
                positions_3d_m = ballistic_model(optimized_params, times_all)
                
                # Convert back to yards and create Position3D objects
                positions_3d = []
                for i, pos_m in enumerate(positions_3d_m):
                    pos_yards = pos_m / 0.9144  # Convert meters to yards
                    frame_num = int(image_points_with_time[i][1] * self.fps)  # Convert time to frame
                    positions_3d.append(Position3D(pos_yards[0], pos_yards[1], pos_yards[2], frame_num))
                
                return positions_3d
            
        except Exception as e:
            print(f"3D trajectory optimization failed: {e}")
            print("Attempting fallback with simpler trajectory model...")
            
            # Fallback: use simple linear interpolation in 3D space
            try:
                fallback_positions = []
                for i, ((u, v), t) in enumerate(image_points_with_time):
                    # Use homography for X,Z and assume parabolic height
                    point_img = np.array([u, v, 1], dtype=np.float32)
                    point_ground = self.homography @ point_img
                    point_ground = point_ground / point_ground[2]
                    
                    # Simple parabolic height model
                    t_normalized = t / max([time for _, time in image_points_with_time])
                    height_m = 3.0 + 5.0 * t_normalized * (1 - t_normalized)  # Parabolic arc
                    
                    pos_yards = np.array([point_ground[0], height_m, point_ground[1]]) / 0.9144
                    frame_num = int(t * self.fps)
                    fallback_positions.append(Position3D(pos_yards[0], pos_yards[1], pos_yards[2], frame_num))
                
                print(f"Fallback method generated {len(fallback_positions)} positions")
                return fallback_positions
                
            except Exception as fallback_error:
                print(f"Fallback method also failed: {fallback_error}")
        
        return []
    
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
        
        # Compute 3D positions using ballistic trajectory fitting
        if self.homography is not None:
            print("Computing 3D trajectory...")
            self.compute_3d_trajectory()
        else:
            print("3D trajectory computation skipped (no valid calibration)")
    
    def compute_3d_trajectory(self):
        """Compute 3D positions for all frames with valid bounding boxes"""
        if self.homography is None:
            return
        
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
                cv2.rectangle(frame, (bbox[0], bbox[1]), 
                            (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
                
                # Draw center point
                center = (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2)
                cv2.circle(frame, center, 3, (0, 0, 255), -1)
                
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
            print(f"Output video: {self.output_video_path}")
            
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
