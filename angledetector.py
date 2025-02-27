import asyncio
import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
from collections import deque

# Constants
JOINT_DEADBAND = 2  # Degrees threshold to filter noise
SMOOTHING_ALPHA = 0.3  # Exponential moving average smoothing factor
WIDTH, HEIGHT = 640, 480  # Camera resolution

# Define joint pairs for angle calculation (MCP, PIP, DIP for all fingers)
FINGER_JOINTS = [
    ("thumb_cmc", "thumb_mcp", "thumb_pip"),
    ("thumb_mcp", "thumb_pip", "thumb_ip"),
    ("index_wrist", "index_mcp", "index_pip"),
    ("index_mcp", "index_pip", "index_dip"),
    ("index_pip", "index_dip", "index_tip"),
    ("middle_wrist", "middle_mcp", "middle_pip"),
    ("middle_mcp", "middle_pip", "middle_dip"),
    ("middle_pip", "middle_dip", "middle_tip"),
    ("ring_wrist", "ring_mcp", "ring_pip"),
    ("ring_mcp", "ring_pip", "ring_dip"),
    ("ring_pip", "ring_dip", "ring_tip"),
    ("pinky_wrist", "pinky_mcp", "pinky_pip"),
    ("pinky_mcp", "pinky_pip", "pinky_dip"),
    ("pinky_pip", "pinky_dip", "pinky_tip"),
]

# Map joint names to MediaPipe indices
JOINT_INDICES = {
    "thumb_cmc": 1, "thumb_mcp": 2, "thumb_pip": 3, "thumb_ip": 4,
    "index_wrist": 0, "index_mcp": 5, "index_pip": 6, "index_dip": 7, "index_tip": 8,
    "middle_wrist": 0, "middle_mcp": 9, "middle_pip": 10, "middle_dip": 11, "middle_tip": 12,
    "ring_wrist": 0, "ring_mcp": 13, "ring_pip": 14, "ring_dip": 15, "ring_tip": 16,
    "pinky_wrist": 0, "pinky_mcp": 17, "pinky_pip": 18, "pinky_dip": 19, "pinky_tip": 20,
}

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
pipeline.start(config)

# Align depth to color stream
align = rs.align(rs.stream.color)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Dictionary to store smoothed joint angles
angle_history = {}

# Function to calculate joint angles with improved accuracy
def calculate_angle(a, b, c):
    """Calculates the angle (in degrees) between three 3D points (a, b, c)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = (a - b) / np.linalg.norm(a - b)
    bc = (c - b) / np.linalg.norm(c - b)
    cosine_angle = np.dot(ba, bc)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

# Function to get depth from RealSense
def get_depth_at_pixel(depth_frame, x, y):
    """Returns the depth value at a specific pixel coordinate (x, y)."""
    x = np.clip(x, 0, WIDTH - 1)
    y = np.clip(y, 0, HEIGHT - 1)
    depth = depth_frame.get_distance(x, y)
    return depth if depth > 0 else None  # Ensure valid depth

# Exponential Moving Average (EMA) for angle smoothing
def smooth_angle(joint, new_angle):
    """Smooths the joint angle using an exponential moving average."""
    if joint not in angle_history:
        angle_history[joint] = new_angle  # Initialize with first value
    else:
        angle_history[joint] = SMOOTHING_ALPHA * new_angle + (1 - SMOOTHING_ALPHA) * angle_history[joint]
    return angle_history[joint]

# Hand tracking function
async def hand_tracking():
    print("RealSense Hand Tracking Running... Press 'Q' to exit.")
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # Convert color frame to NumPy array
            color_image = np.asanyarray(color_frame.get_data())
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Process hand landmarks with MediaPipe
            results = hands.process(rgb_image)
            joint_angles = {}

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Extract joint positions with RealSense depth
                    joint_positions = {}
                    for name, idx in JOINT_INDICES.items():
                        x = int(hand_landmarks.landmark[idx].x * WIDTH)
                        y = int(hand_landmarks.landmark[idx].y * HEIGHT)
                        z = get_depth_at_pixel(depth_frame, x, y)  # Use real depth data
                        joint_positions[name] = (x, y, z if z is not None else 0)  # Fallback to 0 if depth missing

                    # Compute joint angles for MCP, PIP, and DIP
                    for finger in FINGER_JOINTS:
                        j1, j2, j3 = joint_positions[finger[0]], joint_positions[finger[1]], joint_positions[finger[2]]

                        raw_angle = calculate_angle(j1, j2, j3)  # Raw angle
                        smoothed_angle = smooth_angle(finger[1], raw_angle)  # Apply smoothing

                        # Store smoothed angle
                        joint_angles[f"{finger[1]}"] = smoothed_angle

                    # Overlay joint angles on image
                    for joint, angle in joint_angles.items():
                        if joint in joint_positions:
                            x, y, _ = joint_positions[joint]
                            cv2.putText(color_image, f"{angle:.1f}°", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Print joint angles in the console
                    print("\nReal-time Smoothed Joint Angles:")
                    for joint, angle in joint_angles.items():
                        print(f"{joint}: {angle:.2f}°")

            # Display processed video
            cv2.imshow("RealSense Hand Tracking", color_image)

            # Press 'Q' to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            await asyncio.sleep(0.01)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("RealSense Hand Tracking Closed.")

# Main async function
async def main():
    await asyncio.create_task(hand_tracking())

asyncio.run(main())
