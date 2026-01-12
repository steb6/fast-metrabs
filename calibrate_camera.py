#!/usr/bin/env python
"""
Camera calibration script using OpenCV and YARP.
This will calculate the intrinsic parameters (fx, fy, cx, cy) for your webcam.

Instructions:
1. Make sure yarpserver is running and webcam stream is available at /stream/webcam:o
2. Display checkerboard pattern on phone or print it
3. Run this script
4. Show the checkerboard to the camera from different angles and distances
5. Press SPACE to capture each image (capture at least 10-15 good images)
6. Press 'q' to finish and calculate calibration
7. Update the values in configs.py with the calculated parameters
"""

import cv2
import numpy as np
import os
import yarp

# Initialize YARP
yarp.Network.init()

# Checkerboard dimensions (internal corners)
# For 5x8 checkerboard, internal corners are (4, 7)
CHECKERBOARD = (4, 7)  # Internal corners for 5x8 checkerboard
SQUARE_SIZE = 0.011  # Size of a square in meters (11mm = 0.011m)

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Create directory for calibration images
os.makedirs('calibration_images', exist_ok=True)

# Initialize YARP port to receive images
print("Initializing YARP port...")
input_port = yarp.BufferedPortImageRgb()
input_port.open("/calibration/image:i")

print("Connecting to /stream/webcam:o...")
yarp.Network.connect("/stream/webcam:o", "/calibration/image:i")

# Get image dimensions from first frame
print("Waiting for first frame...")
yarp_image = input_port.read(True)  # Blocking read
if yarp_image is None:
    print("Error: Could not read from /stream/webcam:o")
    print("Make sure the webcam stream is running!")
    exit()

image_width = yarp_image.width()
image_height = yarp_image.height()
print(f"Image size: {image_width}x{image_height}")

# Setup buffer for reading images
input_buffer_array = np.ones((image_height, image_width, 3), dtype=np.uint8)
input_buffer_image = yarp.ImageRgb()
input_buffer_image.resize(image_width, image_height)
input_buffer_image.setExternal(input_buffer_array.data, image_width, image_height)

print("\nCalibration Instructions:")
print("- Show the checkerboard pattern to the camera")
print("- Move it around, tilt it, change distance")
print("- Press SPACE when you see green corners to capture")
print("- Capture at least 10-15 good images")
print("- Press 'q' when done to calculate calibration\n")

image_count = 0

while True:
    # Read image from YARP port
    yarp_image = input_port.read(False)
    if yarp_image is None:
        continue
    
    # Copy to buffer
    input_buffer_image.copy(yarp_image)
    frame = np.copy(input_buffer_array)
    
    # Convert from RGB to BGR for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find the checkerboard corners
    ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
                                                      cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                      cv2.CALIB_CB_FAST_CHECK + 
                                                      cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    # Display frame with corners if found
    frame_display = frame.copy()
    if ret_corners:
        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Draw corners
        cv2.drawChessboardCorners(frame_display, CHECKERBOARD, corners_refined, ret_corners)
        cv2.putText(frame_display, "Checkerboard detected! Press SPACE to capture", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame_display, "No checkerboard detected", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.putText(frame_display, f"Captured: {image_count} images", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame_display, "Press 'q' to finish", 
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Camera Calibration', frame_display)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' ') and ret_corners:  # Space bar
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        
        # Save image
        img_name = f'calibration_images/calib_{image_count:02d}.jpg'
        cv2.imwrite(img_name, frame)
        image_count += 1
        print(f"Captured image {image_count}")
        
    elif key == ord('q'):  # Quit
        break

# Cleanup
input_port.close()
yarp.Network.fini()
cv2.destroyAllWindows()

if len(objpoints) < 10:
    print(f"\nError: Not enough images captured ({len(objpoints)})")
    print("You need at least 10 images for good calibration")
    exit()

print(f"\nCalibrating camera with {len(objpoints)} images...")

# Calibrate camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

if ret:
    print("\n" + "="*60)
    print("CALIBRATION SUCCESSFUL!")
    print("="*60)
    
    # Extract parameters
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    print("\nIntrinsic Parameters:")
    print(f"fx (focal length x): {fx:.6f}")
    print(f"fy (focal length y): {fy:.6f}")
    print(f"cx (principal point x): {cx:.6f}")
    print(f"cy (principal point y): {cy:.6f}")
    
    print("\nDistortion Coefficients:")
    print(f"k1, k2, p1, p2, k3: {dist_coeffs.ravel()}")
    
    print("\n" + "="*60)
    print("UPDATE configs.py WITH THESE VALUES:")
    print("="*60)
    print(f"\nfx: float = {fx:.6f}")
    print(f"fy: float = {fy:.6f}")
    print(f"ppx: float = {cx:.6f}")
    print(f"ppy: float = {cy:.6f}")
    print("\n" + "="*60)
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                         camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    mean_error /= len(objpoints)
    print(f"\nMean reprojection error: {mean_error:.4f} pixels")
    print("(Lower is better, < 0.5 pixels is excellent, < 1.0 is good)\n")
    
    # Save calibration
    np.savez('camera_calibration.npz', 
             camera_matrix=camera_matrix, 
             dist_coeffs=dist_coeffs,
             fx=fx, fy=fy, cx=cx, cy=cy)
    print("Calibration saved to: camera_calibration.npz")
    
else:
    print("\nCalibration failed!")
