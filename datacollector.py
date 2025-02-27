import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

# Set image capture parameters
IMAGE_COUNT = 1000  # Total images to capture
DURATION = 10      # Duration in seconds
FRAME_INTERVAL = DURATION / IMAGE_COUNT  # Time per frame

# Get user input for gesture name
gesture_name = input("Enter the name of the gesture: ").strip()

# Create directory to store images
save_dir = os.path.join("gesture_data", gesture_name)
os.makedirs(save_dir, exist_ok=True)

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure RealSense stream (RGB only)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Countdown before capturing
print("Position yourself! Starting in:")
for i in range(5, 0, -1):
    print(f"{i}...")
    time.sleep(1)

print(f"Collecting data for '{gesture_name}'... Move your hand into the frame.")

# Capture images
start_time = time.time()
image_counter = 0

while image_counter < IMAGE_COUNT:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        continue  # Skip if no frame captured

    # Convert frame to numpy array
    color_image = np.asanyarray(color_frame.get_data())

    # Display countdown before first capture
    if image_counter == 0:
        for i in range(3, 0, -1):
            frame_with_text = color_image.copy()
            cv2.putText(frame_with_text, f"Starting in {i}", (200, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.imshow("Capturing Gesture", frame_with_text)
            cv2.waitKey(1000)

    # Save image
    image_path = os.path.join(save_dir, f"{gesture_name}_{image_counter:03d}.jpg")
    cv2.imwrite(image_path, color_image)

    # Show preview
    cv2.imshow("Capturing Gesture", color_image)
    cv2.waitKey(1)  # Small delay to allow image display

    # Wait for the next frame interval
    elapsed_time = time.time() - start_time
    expected_time = (image_counter + 1) * FRAME_INTERVAL
    while time.time() - start_time < expected_time:
        pass  # Wait to maintain frame rate

    image_counter += 1

# Stop pipeline and close windows
pipeline.stop()
cv2.destroyAllWindows()
print(f"Data collection complete! {IMAGE_COUNT} images saved in '{save_dir}'.")
