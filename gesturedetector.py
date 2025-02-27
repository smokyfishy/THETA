import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np

# Import MediaPipe Tasks properly
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load the MediaPipe Gesture Recognition Model
model_path = "alex.task"  # Ensure the model file is in the same directory
options = vision.GestureRecognizerOptions(base_options=python.BaseOptions(model_asset_path=model_path))
gesture_recognizer = vision.GestureRecognizer.create_from_options(options)

# Initialize RealSense Camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# OpenCV Window for Display
cv2.namedWindow("MediaPipe Gesture Detector", cv2.WINDOW_AUTOSIZE)

try:
    while True:
        # Get Frames from RealSense Camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert Image to OpenCV Format
        frame = np.asanyarray(color_frame.get_data())
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert Frame to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Run Gesture Recognition
        result = gesture_recognizer.recognize(mp_image)

        # Process and Display Gesture Results
        if result.gestures:
            gesture_name = result.gestures[0][0].category_name  # Get the highest confidence gesture
            confidence = result.gestures[0][0].score  # Get gesture confidence score

            # Display Gesture Name and Confidence Score
            cv2.putText(frame, f"Gesture: {gesture_name} ({confidence:.2f})", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the Frame
        cv2.imshow("MediaPipe Gesture Detector", frame)

        # Press 'q' to Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
