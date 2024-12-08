import cv2
import numpy as np
import time, os
import pytesseract

def sort_corners(corners):
    """
    Sorts the corners of the document into the correct order by finding sum and diff
    Smallest sum is TL corner and largest sum is BR
    Smallest difference is TR and largest difference is BL
    Order: TL, TR, BR, BL
    :param corners:
    :return:
    """
    rectangle = np.zeros((4, 2), dtype="float32")
    sum_corners = corners.sum(axis=1)
    diff_corners = np.diff(corners, axis=1)

    rectangle[0] = corners[np.argmin(sum_corners)] # Top left corner
    rectangle[1] = corners[np.argmin(diff_corners)] # Top right corner
    rectangle[2] = corners[np.argmax(sum_corners)] # Bottom right corner
    rectangle[3] = corners[np.argmax(diff_corners)] # Bottom left corner
    return rectangle

def warp_document(image, corners):
    """
    Transforms view to top down of the document
    :param image:
    :param corners:
    :return:
    """
    ordered_corners = sort_corners(corners)

    # Find width and height of document
    width_top = np.linalg.norm(ordered_corners[1] - ordered_corners[0])    # TR - TL
    width_bottom = np.linalg.norm(ordered_corners[2] - ordered_corners[3]) # BR - BL
    max_width = max(int(width_top), int(width_bottom))
    height_left = np.linalg.norm(ordered_corners[3] - ordered_corners[0])  # BL - TL
    height_right = np.linalg.norm(ordered_corners[2] - ordered_corners[1]) # BR - TR
    max_height = max(int(height_left), int(height_right))

    # Map document to new rectangle
    destination_points = np.array([
        [0, 0],                           # TL
        [max_width - 1, 0],               # TR
        [max_width - 1, max_height - 1],  # BR
        [0, max_height - 1]               # BL
    ], dtype="float32")

    # Transform the image
    transformation_matrix = cv2.getPerspectiveTransform(ordered_corners, destination_points)
    warped_image = cv2.warpPerspective(image, transformation_matrix, (max_width, max_height))
    return warped_image

def preprocess_frame(frame):
    """
    Image preprocessing
    :param frame:
    :return:
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    edges = cv2.adaptiveThreshold(
        blurred_frame,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        29,  # block size - lower for sharper final doc but needs const lighting
        2  # edge constant - lower for more edge higher for less
    )
    return edges

def detect_document(edges):
    """
    Finds quadrilaterals and highlights the largest one
    :param edges:
    :return: approximated_contour:
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours to find largest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        contour_length = cv2.arcLength(contour, True)
        approximated_contour = cv2.approxPolyDP(contour, 0.02 * contour_length, True)
        if len(approximated_contour) == 4:
            return approximated_contour
    return None

def convert_txt(image_path):
    """
    Processes the saved image to extract text and saves it to a .txt file.

    Args:
        image_path (str): Path to the saved image file.
    """
    # Extract text from the image using Tesseract
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding for better text contrast
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Remove noise
    cleaned = cv2.medianBlur(threshold, 3)

    config = "--psm 6 -l eng"
    extracted_text = pytesseract.image_to_string(cleaned, config=config)

    # Generate a corresponding .txt file name
    txt_file_path = os.path.splitext(image_path)[0] + ".txt"

    # Save the extracted text to the .txt file
    with open(txt_file_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(extracted_text)

    print(f"Extracted text has been saved to: {txt_file_path}")

def resize_with_aspect_ratio(frame, width=None, height=None):
    """
    Resize an image while maintaining its aspect ratio.

    Args:
        frame (np.ndarray): The input image.
        width (int): Desired width (optional).
        height (int): Desired height (optional).

    Returns:
        np.ndarray: The resized image with the aspect ratio preserved.
    """
    h, w = frame.shape[:2]

    if width is not None:
        # Calculate the new height while maintaining aspect ratio
        new_height = int((width / w) * h)
        return cv2.resize(frame, (width, new_height))
    elif height is not None:
        # Calculate the new width while maintaining aspect ratio
        new_width = int((height / h) * w)
        return cv2.resize(frame, (new_width, height))
    else:
        # If no size is provided, return the original frame
        return frame

def main():
    """
    Main logic for scanner
    :return:
    """
    cap_front = cv2.VideoCapture(0)  # Front camera
    cap_back = cv2.VideoCapture(1)   # Back camera
    active_capture = cap_front

    if not cap_front.isOpened():
        print("Error: iPhone not connected")
        cap_front.release()
        active_capture = cap_back

    elif not cap_back.isOpened():
        print("Error: idk how the webcam would disconnect")
        cap_back.release()
        active_capture = cap_front

    while True:
        ret, frame = active_capture.read()
        if not ret:
            print("Error: Unable to read frame from the active camera.")
            break
        if active_capture == cap_front:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            # Maintain aspect ratio while resizing
            frame = resize_with_aspect_ratio(frame, width=800)
        else:
            # Resize directly for the back camera
            frame = cv2.resize(frame, (800, 600))

        markup = frame.copy()
        resized_frame = cv2.resize(frame, (800, 600))
        edges = preprocess_frame(resized_frame)

        document_contour = detect_document(edges)
        if document_contour is not None:
            scale_x = frame.shape[1] / resized_frame.shape[1]
            scale_y = frame.shape[0] / resized_frame.shape[0]
            scaled_points = document_contour.reshape(4, 2) * [scale_x, scale_y]

            cv2.drawContours(markup, [scaled_points.astype(int)], -1, (0, 255, 0), 2)
            cv2.putText(markup, "Press SPACE to capture", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space bar key
                warped_document = warp_document(frame, scaled_points)
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                file_name = f"scanned_document_{timestamp}.jpg"
                cv2.imwrite(file_name, warped_document)

                convert_txt(file_name)

                print(f"Document saved as {file_name}")
                cv2.imshow("Scanned Document", cv2.resize(warped_document, (400, 300)))

        # Switch cameras
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):  # Switch to front camera
            active_capture = cap_front
            print("Switched to front camera.")
        elif key == ord('2'):  # Switch to back camera
            active_capture = cap_back
            print("Switched to back camera.")
        elif key == ord('q'):  # Quit
            break

        cv2.imshow("Document Scanner", markup)
        cv2.imshow("Edges", edges)

    # Release resources and close windows
    cap_front.release()
    cap_back.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
