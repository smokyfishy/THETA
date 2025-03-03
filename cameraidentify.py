import cv2

# ==============================
# ✅ Open Webcam (Default: ID 0)
# ==============================
camera_id = 3 # Change to 1, 2, etc., if you have multiple webcams
cap = cv2.VideoCapture(camera_id)

if not cap.isOpened():
    print(f"❌ Error: Could not open webcam with ID {camera_id}")
    exit()

print("✅ Webcam is open. Press 'q' to exit.")

# ==============================
# ✅ Capture Frames in a Loop
# ==============================
while True:
    ret, frame = cap.read()  # Capture frame-by-frame

    if not ret:
        print("⚠️ Warning: Failed to capture frame")
        break

    cv2.imshow("Webcam Feed", frame)  # Display the frame

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# ✅ Cleanup and Release Resources
# ==============================
print("🔄 Closing webcam...")
cap.release()
cv2.destroyAllWindows()
