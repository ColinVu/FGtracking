#!/usr/bin/env python3
"""
Debug script to test the height calculation issue
"""

import numpy as np

# Test the fallback height calculation
def test_fallback_height():
    print("Testing fallback height calculation...")
    
    # Simulate 5 time points from 0 to 1
    time_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    print("OLD fallback (8.0 * t * (1-t)):")
    for t in time_points:
        height_m = 8.0 * t * (1 - t)
        height_yards = height_m * 1.094
        print(f"  t={t:.2f}: {height_m:.2f}m = {height_yards:.1f} yards")
    
    max_height_old = 8.0 * 0.25  # Maximum at t=0.5
    print(f"  Maximum: {max_height_old:.2f}m = {max_height_old * 1.094:.1f} yards")
    
    print("\nNEW fallback (25.0 * t * (1-t)):")
    for t in time_points:
        height_m = 25.0 * t * (1 - t)
        height_yards = height_m * 1.094
        print(f"  t={t:.2f}: {height_m:.2f}m = {height_yards:.1f} yards")
    
    max_height_new = 25.0 * 0.25  # Maximum at t=0.5
    print(f"  Maximum: {max_height_new:.2f}m = {max_height_new * 1.094:.1f} yards")

# Test ballistic trajectory with different initial velocities
def test_ballistic_trajectory():
    print("\nTesting ballistic trajectory heights...")
    
    g = 9.81  # gravity
    time_points = np.linspace(0, 2, 11)  # 0 to 2 seconds
    
    velocities = [10, 15, 20, 25, 30]  # m/s
    
    for vy0 in velocities:
        max_height = (vy0 ** 2) / (2 * g)
        max_height_yards = max_height * 1.094
        print(f"Initial velocity {vy0} m/s -> Max height: {max_height:.1f}m = {max_height_yards:.1f} yards")
        
        # Show trajectory
        heights = []
        for t in time_points:
            y = vy0 * t - 0.5 * g * t * t
            if y >= 0:  # Only show above ground
                heights.append(y)
            else:
                break
        
        if heights:
            print(f"  Trajectory: {[f'{h:.1f}' for h in heights[:5]]}... (meters)")

if __name__ == "__main__":
    test_fallback_height()
    test_ballistic_trajectory()
