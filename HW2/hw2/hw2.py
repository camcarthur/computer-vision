import cv2
import math
import numpy as np
from cv2 import GaussianBlur


def sign_lines(img: np.ndarray) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of lines.

    https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
    :param img: Image as numpy array
    :return: Numpy array of lines.
    """

    edges = cv2.Canny(img, threshold1=50, threshold2=150, apertureSize=3)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,            # Distance resolution of the accumulator in pixels
        theta=np.pi / 180,  # Angle resolution of the accumulator in radians
        threshold=120,    # Minimum number of intersections to detect a line
        minLineLength=50,  # Minimum length of a line
        maxLineGap=10      # Maximum allowed gap between line segments
    )

    # If no lines are detected, return an empty array
    if lines is None:
        return np.array([])
    return np.array(lines[:, 0])

def sign_circle(img: np.ndarray) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of circles.
    :param img: Image as numpy array
    :return: Numpy array of circles.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    filled_circle = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Step 2: Edge detection
    edges = cv2.Canny(filled_circle, 50, 150)
    cv2.imshow("Edges for Circle Detection", edges)
    cv2.waitKey(0)

    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,  # Inverse ratio of resolution
        minDist=30,  # Minimum distance between circle centers
        param1=50,  # Upper threshold for the Canny edge detector
        param2=35,  # Threshold for center detection
        minRadius=50,  # Minimum radius of circles
        maxRadius=200  # Maximum radius of circles
    )

    if circles is None:
        return np.array([])
    return np.uint16(np.around(circles[0]))

def sign_axis(lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    This function takes in a numpy array of lines and returns a tuple of np.ndarray and np.ndarray.

    This function should identify the lines that make up a sign and split the x and y coordinates.
    :param lines: Numpy array of lines.
    :return: Tuple of np.ndarray and np.ndarray with each np.ndarray consisting of the x coordinates and y coordinates
             respectively.
    """
    xaxis = np.empty(0, dtype=np.int32)
    yaxis = np.empty(0, dtype=np.int32)
    return xaxis, yaxis

def identify_traffic_light(img: np.ndarray) -> tuple:
    """
    This function identifies the location of the traffic light in the image and the lit light color.
    Returns (x, y, color) if a traffic light is detected, or (-1, -1, 'unknown') if not.
    """
    import cv2
    import numpy as np

    # img_resized = cv2.resize(img, (600, 400))
    img_resized = img.copy()
    gray_image = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hsv_image = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    # mask for dark colors to isolate TL
    dark_lower = np.array([0, 0, 0])
    dark_upper = np.array([180, 255, 80])
    dark_mask = cv2.inRange(hsv_image, dark_lower, dark_upper)

    # reduce noise in mask
    kernel = np.ones((3, 3), np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

    # display mask
    # cv2.imshow('mask', dark_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Edge detection on the masked image
    masked_gray = cv2.bitwise_and(gray_image, gray_image, mask=dark_mask)
    edges = cv2.Canny(masked_gray, threshold1=50, threshold2=150)

    # display edges
    # cv2.imshow('edges', edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # find the contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # is rectangle?
    def is_rectangle(contour):
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return len(approx) == 4 and cv2.isContourConvex(approx)

    # try to find rectangles
    def filter_rectangles(contours):
        rectangles = []
        for contour in contours:
            if is_rectangle(contour):
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                aspect_ratio = h / float(w)
                print(
                    f"Contour Area: {area}, Aspect Ratio: {aspect_ratio:.2f}, Coordinates: ({x}, {y}), Size: ({w}, {h})")
                # Adjust thresholds as needed
                if area > 200 and 0.4 < aspect_ratio < 6.0:
                    rectangles.append((contour, (x, y, w, h)))
        return rectangles

    rectangles = filter_rectangles(contours)
    print(f"Number of rectangles found: {len(rectangles)}")

    # display rectangles
    # img_rectangles = img_resized.copy()
    # for contour, (x, y, w, h) in rectangles:
    #     cv2.rectangle(img_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imshow('Detected Rectangles', img_rectangles)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if not rectangles:
        print("No traffic light found")
        return -1, -1, 'unknown'

    # find best match
    best_match = None
    max_brightness = 0
    lit_color = 'unknown'
    x_center, y_center = -1, -1

    for contour, (x, y, w, h) in rectangles:
        # split tl into three sections to compare
        light_height = h // 3
        regions = {
            'red': (x, y, w, light_height),
            'yellow': (x, y + light_height, w, light_height),
            'green': (x, y + 2 * light_height, w, h - 2 * light_height)
        }

        brightness = {}
        positions = {}
        for color, (rx, ry, rw, rh) in regions.items():
            roi = hsv_image[ry:ry + rh, rx:rx + rw]

            # create color masks
            if color == 'red':
                mask1 = cv2.inRange(roi, np.array([0, 100, 100]), np.array([10, 255, 255]))
                mask2 = cv2.inRange(roi, np.array([160, 100, 100]), np.array([179, 255, 255]))
                mask = cv2.bitwise_or(mask1, mask2)
            elif color == 'yellow':
                mask = cv2.inRange(roi, np.array([15, 100, 100]), np.array([35, 255, 255]))
            elif color == 'green':
                mask = cv2.inRange(roi, np.array([45, 100, 100]), np.array([75, 255, 255]))

            # reduce noise in mask
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # display color masks
            # cv2.imshow(f'{color.capitalize()} Mask', mask)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # Calculate the percentage of the mask that is non-zero
            non_zero_pixels = cv2.countNonZero(mask)
            total_pixels = mask.shape[0] * mask.shape[1]
            if total_pixels == 0:
                continue
            color_ratio = non_zero_pixels / total_pixels

            brightness[color] = color_ratio
            # find center position
            positions[color] = (rx + rw // 2, ry + rh // 2)

        if brightness:
            # find highest color ratio
            max_color = max(brightness, key=brightness.get)
            if brightness[max_color] > max_brightness and brightness[max_color] > 0.1:
                max_brightness = brightness[max_color]
                lit_color = max_color
                x_center, y_center = positions[lit_color]
                best_match = (x_center, y_center, lit_color)

    if best_match:
        x_center, y_center, lit_color = best_match
        print(f"The {lit_color} light is lit at position ({x_center}, {y_center}).")
        # translate coordinates to original image
        x = int(x_center * img.shape[1] / img_resized.shape[1])
        y = int(y_center * img.shape[0] / img_resized.shape[0])
        return x, y, lit_color
    else:
        print("No lit traffic light detected.")
        return -1, -1, 'unknown'

def identify_stop_sign(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'stop')
    """
    img_resized = cv2.resize(img, (600, 400))
    img_blur = cv2.GaussianBlur(img_resized, (5, 5), 0)
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

    # define red color range
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([179, 255, 255])

    # create red mask
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    best_match = None
    max_area = 0

    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # is octagon?
        if len(approx) == 8 and cv2.isContourConvex(approx):
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                best_match = approx

    if best_match is not None:
        # find center of sign
        M = cv2.moments(best_match)
        if M['m00'] == 0:
            M['m00'] = 1
        x_center = int(M['m10'] / M['m00'])
        y_center = int(M['m01'] / M['m00'])

        # translate coordinates to original image
        x = int(x_center * img.shape[1] / img_resized.shape[1])
        y = int(y_center * img.shape[0] / img_resized.shape[0])

        print(f"Stop sign detected at ({x}, {y}).")
        return x, y, 'stop'

    else:
        print("No stop sign detected.")
        return -1, -1, 'unknown'

def identify_yield(img: np.ndarray) -> tuple:
    """
    Identifies the yield sign in the image.
    Returns (x, y, 'yield') if detected, or (-1, -1, 'unknown') if not.
    """
    import cv2
    import numpy as np

    img_resized = cv2.resize(img, (600, 400))
    img_blur = cv2.GaussianBlur(img_resized, (5, 5), 0)
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

    # define red color range
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([179, 255, 255])

    # cv2.imshow("test", hsv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # create red mask
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # reduce noise in mask
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # canny edge detection
    edges = cv2.Canny(red_mask, 50, 150)

    # find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    best_match = None
    max_area = 0

    for cnt in contours:
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # is triangle?
        if len(approx) == 3 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > 500:
                pts = approx.reshape(3, 2)
                # find center
                M = cv2.moments(approx)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                pts = pts[pts[:, 1].argsort()]
                # make sure it's pointing the right way
                if pts[2][1] > pts[0][1] and pts[2][1] > pts[1][1]:
                    if area > max_area:
                        max_area = area
                        best_match = approx

    if best_match is not None:
        # find center
        M = cv2.moments(best_match)
        if M['m00'] == 0:
            M['m00'] = 1
        x_center = int(M['m10'] / M['m00'])
        y_center = int(M['m01'] / M['m00'])

        # translate coordinates to original image
        x = int(x_center * img.shape[1] / img_resized.shape[1])
        y = int(y_center * img.shape[0] / img_resized.shape[0])

        print(f"Yield sign detected at ({x}, {y}).")
        return x, y, 'yield'

    else:
        print("No yield sign detected.")
        return -1, -1, 'unknown'

def identify_construction(img: np.ndarray) -> tuple:
    """
    Identifies the construction sign in the image.
    Returns (x, y, 'construction') if detected, or (-1, -1, 'unknown') if not.
    """
    import cv2
    import numpy as np

    img_resized = cv2.resize(img, (600, 400))
    img_blur = cv2.GaussianBlur(img_resized, (5, 5), 0)
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

    # define orange color range
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([20, 255, 255])

    # create orange mask
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # reduce noise in mask
    kernel = np.ones((5, 5), np.uint8)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)

    # find contours
    contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_match = None
    max_area = 0

    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # is 4 sided?
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > 500:
                x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(approx)
                aspect_ratio = float(w_rect) / h_rect

                # find angle
                (vx, vy, x_center, y_center) = cv2.fitLine(approx, cv2.DIST_L2, 0, 0.01, 0.01)
                angle = math.degrees(math.atan2(vy, vx))

                # is diamond?
                if 40 < abs(angle) < 50 or 130 < abs(angle) < 140:
                    if area > max_area:
                        max_area = area
                        best_match = approx

    if best_match is not None:
        # find center
        M = cv2.moments(best_match)
        if M['m00'] == 0:
            M['m00'] = 1
        x_center = int(M['m10'] / M['m00'])
        y_center = int(M['m01'] / M['m00'])

        # translate coordinates to original image
        x = int(x_center * img.shape[1] / img_resized.shape[1])
        y = int(y_center * img.shape[0] / img_resized.shape[0])

        print(f"Construction sign detected at ({x}, {y}).")
        return x, y, 'construction'
    else:
        print("No construction sign detected.")
        return -1, -1, 'unknown'

def identify_warning(img: np.ndarray) -> tuple:
    """
    Identifies the warning sign in the image.
    Returns (x, y, 'warning') if detected, or (-1, -1, 'unknown') if not.
    """
    import cv2
    import numpy as np

    img_resized = cv2.resize(img, (600, 400))
    img_blur = cv2.GaussianBlur(img_resized, (5, 5), 0)
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

    # define yellow color range
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    # create yellow mask
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # reduce noise in mask
    kernel = np.ones((5, 5), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

    # find contours
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_match = None
    max_area = 0

    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 4 sided?
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > 500:
                x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(approx)
                aspect_ratio = float(w_rect) / h_rect

                # find angle
                (vx, vy, x_center, y_center) = cv2.fitLine(approx, cv2.DIST_L2, 0, 0.01, 0.01)
                angle = math.degrees(math.atan2(vy, vx))

                # is diamond?
                if 40 < abs(angle) < 50 or 130 < abs(angle) < 140:
                    if area > max_area:
                        max_area = area
                        best_match = approx

    if best_match is not None:
        # find center
        M = cv2.moments(best_match)
        if M['m00'] == 0:
            M['m00'] = 1
        x_center = int(M['m10'] / M['m00'])
        y_center = int(M['m01'] / M['m00'])

        # translate coordinates to original image
        x = int(x_center * img.shape[1] / img_resized.shape[1])
        y = int(y_center * img.shape[0] / img_resized.shape[0])

        print(f"Warning sign detected at ({x}, {y}).")
        return x, y, 'warning'
    else:
        print("No warning sign detected.")
        return -1, -1, 'unknown'

def identify_rr_crossing(img: np.ndarray) -> tuple:
    """
    Identifies the railroad crossing sign in the image.
    Returns (x, y, 'rr_crossing') if detected, or (-1, -1, 'unknown') if not.
    """
    #img_resized = cv2.resize(img, (600, 400))
    img_blur = cv2.medianBlur(img, 5)
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

    # Define yellow color range
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    # Create yellow mask
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Reduce noise in mask
    kernel = np.ones((5, 5), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

    # Debug yellow mask
    # cv2.imshow("Yellow Mask", yellow_mask)
    # cv2.waitKey(0)

    # Masked grayscale image
    gray_image = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray_image, gray_image, mask=yellow_mask)


    cv2.imshow("Masked Grayscale Image", masked_gray)
    cv2.waitKey(0)
    # Step 1: Detect circles
    circles = sign_circle(masked_gray)
    print("Detected Circles:", circles)
    if len(circles) == 0:
        return -1, -1, 'unknown'

    # Draw detected circles
    for x, y, radius in circles:
        cv2.circle(img, (x, y), radius, (255, 0, 0), 2)  # Circle outline
        cv2.circle(img, (x, y), 2, (0, 255, 0), 3)  # Circle center
    cv2.imshow("Circles", img)
    cv2.waitKey(0)

    blur = GaussianBlur(masked_gray, (5,5), 0)
    # Step 2: Detect lines
    lines = sign_lines(blur)
    print("Detected Lines:", lines)
    if len(lines) == 0:
        return -1, -1, 'unknown'

    # Draw detected lines
    for x1, y1, x2, y2 in lines:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Line
    cv2.imshow("Lines", img)
    cv2.waitKey(0)

    # Step 3: Validate the "X" shape
    for circle in circles:
        x, y, radius = circle
        lines_in_circle = []

        # Find lines within the circle
        for x1, y1, x2, y2 in lines:
            # Check if both endpoints of the line are within the circle
            if ((x1 - x) ** 2 + (y1 - y) ** 2 <= radius ** 2 and
                    (x2 - x) ** 2 + (y2 - y) ** 2 <= radius ** 2):
                lines_in_circle.append((x1, y1, x2, y2))

        # Validate if lines form an "X" by finding intersections
        for i, line1 in enumerate(lines_in_circle):
            for line2 in lines_in_circle[i + 1:]:
                x1, y1, x2, y2 = line1
                x3, y3, x4, y4 = line2

                # Calculate the intersection point
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if denom == 0:
                    continue  # Parallel lines

                px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
                py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

                # Check if the intersection point lies within the circle
                if (px - x) ** 2 + (py - y) ** 2 <= radius ** 2:
                    print(f"Intersection detected at ({px}, {py})")
                    cv2.circle(img_resized, (int(px), int(py)), 5, (255, 255, 0), -1)  # Draw intersection
                    cv2.imshow("Validated X Shape", img_resized)
                    cv2.waitKey(0)
                    return x, y, 'rr_crossing'

    return -1, -1, 'unknown'


def identify_services(img: np.ndarray) -> tuple:
    """
    Identifies the services sign in the image.
    Returns (x, y, 'services') if detected, or (-1, -1, 'unknown') if not.
    """
    import cv2
    import numpy as np

    img_resized = cv2.resize(img, (600, 400))
    img_blur = cv2.GaussianBlur(img_resized, (5, 5), 0)
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

    # define blue color range
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([130, 255, 255])

    # create blue mask
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # reduce noise in mask
    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

    # find contours
    contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_match = None
    max_area = 0

    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 4 sided?
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > 500:
                # find aspect ratio
                x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(approx)
                aspect_ratio = float(w_rect) / h_rect
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0:
                    continue
                solidity = float(area) / hull_area

                # check if square or rectangle
                if 0.8 < aspect_ratio < 1.2 and solidity > 0.9:
                    if cv2.contourArea(approx) / cv2.contourArea(cnt) > 0.9:
                        if area > max_area:
                            max_area = area
                            best_match = approx

    if best_match is not None:
        # find the center
        M = cv2.moments(best_match)
        if M['m00'] == 0:
            M['m00'] = 1
        x_center = int(M['m10'] / M['m00'])
        y_center = int(M['m01'] / M['m00'])

        # translate coordinates to original image
        x = int(x_center * img.shape[1] / img_resized.shape[1])
        y = int(y_center * img.shape[0] / img_resized.shape[0])

        print(f"Services sign detected at ({x}, {y}).")
        return x, y, 'services'

    else:
        print("No services sign detected.")
        return -1, -1, 'unknown'



def identify_signs(img: np.ndarray) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of all signs locations and name.
    Call the other identify functions to determine where that sign is if it exists.
    :param img: Image as numpy array
    :return: Numpy array of all signs locations and name.
             [[x, y, 'stop'],
              [x, y, 'construction']]
    """
    results = []

    # List of identify functions to call
    identify_functions = [
        identify_stop_sign,
        identify_yield,
        identify_construction,
        identify_warning,
        identify_rr_crossing,
        identify_services,
        identify_traffic_light
    ]

    for func in identify_functions:
        result = func(img)
        if result is not None:
            x, y, sign_name = result
            if x != -1 and y != -1 and sign_name != 'unknown':
                results.append([x, y, sign_name])
        else:
            continue

    if results:
        signs_array = np.array(results, dtype=object)
        return signs_array
    else:
        return None


def identify_signs_noisy(img: np.ndarray) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of all signs locations and name.
    Call the other identify functions to determine where that sign is if it exists.

    The images will have gaussian noise applied to them so you will need to do some blurring before detection.
    :param img: Image as numpy array
    :return: Numpy array of all signs locations and name.
             [[x, y, 'stop'],
              [x, y, 'construction']]
    """
    raise NotImplemented


def identify_signs_real(img: np.ndarray) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of all signs locations and name.
    Call the other identify functions to determine where that sign is if it exists.

    The images will be real images so you will need to do some preprocessing before detection.
    You may also need to adjust existing functions to detect better with real images through named parameters
    and other code paths

    :param img: Image as numpy array
    :return: Numpy array of all signs locations and name.
             [[x, y, 'stop'],
              [x, y, 'construction']]
    """
    raise NotImplemented