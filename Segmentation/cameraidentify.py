import cv2


camera_id = 3 
cap = cv2.VideoCapture(camera_id)

if not cap.isOpened():
    print(f"Error: Could not open webcam with ID {camera_id}")
    exit()

print("Webcam is open. Press 'q' to exit.")


while True:
    ret, frame = cap.read()  

    if not ret:
        print("Warning: Failed to capture frame")
        break

    cv2.imshow("Webcam Feed", frame)  

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


print("Closing webcam...")
cap.release()
cv2.destroyAllWindows()
