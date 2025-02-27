import cv2
import pyrealsense2 as rs
import numpy as np
import tensorflow as tf

# Load the trained gesture recognition model
MODEL_PATH = "gesture_recognition_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define the same image size as used during training
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Define gesture labels (should match the classes used in training)
gesture_labels = ["none", "paper", "rock", "scissors"]  # Modify if needed

# Initialize RealSense Camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# OpenCV Window
cv2.namedWindow("RealSense Gesture Recognition", cv2.WINDOW_AUTOSIZE)

try:
    while True:
        # Get frames from RealSense Camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert Image to OpenCV Format
        frame = np.asanyarray(color_frame.get_data())

        # Preprocess the frame for the model
        resized_frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))  # Resize to match model input
        preprocessed_frame = resized_frame.astype("float32") / 255.0  # Normalize pixel values
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)  # Add batch dimension

        # Predict gesture
        predictions = model.predict(preprocessed_frame)
        confidence_scores = predictions[0]  # Extract confidence scores
        predicted_label_index = np.argmax(confidence_scores)  # Get the highest confidence index
        predicted_label = gesture_labels[predicted_label_index]  # Get corresponding gesture name
        confidence = confidence_scores[predicted_label_index]  # Confidence score of the prediction

        # Display Gesture and Confidence Score on the frame
        cv2.putText(frame, f"Gesture: {predicted_label} ({confidence:.2f})", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow("RealSense Gesture Recognition", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the RealSense pipeline and close OpenCV windows
    pipeline.stop()
    cv2.destroyAllWindows()
