import cv2

# Initialize video captures for front and back cameras
cap_front = cv2.VideoCapture(0)  # front camera
cap_back = cv2.VideoCapture(1)   # back camera
active_capture = cap_front

# Check if cameras are opened successfully
if not cap_front.isOpened() or not cap_back.isOpened():
    print("Error: Unable to access one or both cameras.")
    cap_front.release()
    cap_back.release()
    cv2.destroyAllWindows()
    exit()

while True:
    ret, frame = active_capture.read()

    if not ret:
        print("Error: Unable to read frame from the active camera.")
        break

    key = cv2.waitKey(1) & 0xFF  # Use 0xFF mask for compatibility
    if key == ord("b"):  # Switch to back camera
        active_capture = cap_back
    elif key == ord("f"):  # Switch to front camera
        active_capture = cap_front
    elif key == ord('q'):  # Quit
        break

    # Display the frame
    cv2.imshow("Camera Feed", frame)

# Release resources and close windows
cap_front.release()
cap_back.release()
cv2.destroyAllWindows()
