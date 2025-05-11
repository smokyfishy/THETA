import cv2
import pyrealsense2 as rs
import numpy as np

# Define camera IDs (change if needed)
FRONT_CAM_ID = 2  # Replace with your front camera index
LEFT_CAM_ID = 3   # Replace with your left camera index

# Initialize RealSense pipeline for RGB stream only
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB Stream Only

# Start RealSense pipeline
pipeline.start(config)

# Open front and left webcams
front_cam = cv2.VideoCapture(FRONT_CAM_ID)
left_cam = cv2.VideoCapture(LEFT_CAM_ID)

if not front_cam.isOpened():
    print("Error: Could not open front camera.")
if not left_cam.isOpened():
    print("Error: Could not open left camera.")

# Main loop to display feeds
while True:
    # Get RealSense RGB frame
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    
    if not color_frame:
        print("Error: Could not get RGB frame from RealSense camera.")
        continue

    # Convert RealSense frame to NumPy array
    color_image = np.asanyarray(color_frame.get_data()) # Fix: Convert to NumPy array

    # Read from front and left cameras
    ret_front, front_img = front_cam.read()
    ret_left, left_img = left_cam.read()

    if not ret_front or not ret_left:
        print("Error: Could not get frames from webcams.")
        continue

    # Display all camera feeds
    cv2.imshow("Right Camera", front_img)
    cv2.imshow("Left Camera", left_img)
    cv2.imshow("RealSense RGB Camera", color_image)  # Fix: Now correctly displays RealSense RGB

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
pipeline.stop()
front_cam.release()
left_cam.release()
cv2.destroyAllWindows()
