import cv2
import numpy as np
import os

print("Current working directory:", os.getcwd())
print("Directory contents:", os.listdir("."))

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(aruco_dict)
# Used markers 3 5 7 9
def detect_markers(frame):
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is None or len(ids) < 4:
        print("not all markers present")
        return None

    ids = ids.flatten()
    print("found markers:", ids)

    # 3:TL 5:TR 7:BR 9:BL
    used_markers = [3, 5, 7, 9]

    marker_positions = {}
    for i, marker_id in enumerate(ids):
        c = corners[i][0]
        center_x = np.mean(c[:, 0])
        center_y = np.mean(c[:, 1])
        marker_positions[marker_id] = (center_x, center_y)

    if all(mid in marker_positions for mid in used_markers):
        ordered_points = np.array([
            marker_positions[3],
            marker_positions[5],
            marker_positions[7],
            marker_positions[9]
        ], dtype="float32")
        return ordered_points
    else:
        print("Not all markers found")
        return None


def overlay_on_image(input_image_path, overlay_image_path, output_image_path):
    image = cv2.imread(input_image_path)
    if image is None:
        raise FileNotFoundError(f"couldn't load {input_image_path}.")

    overlay_img = cv2.imread(overlay_image_path)
    if overlay_img is None:
        raise FileNotFoundError(f"couldn't load {overlay_image_path}.")

    # warp perspective
    h, w, _ = overlay_img.shape
    src_points = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype="float32")

    dst_points = detect_markers(image)
    if dst_points is not None:
        H, _ = cv2.findHomography(src_points, dst_points)
        warped_overlay = cv2.warpPerspective(overlay_img, H, (image.shape[1], image.shape[0]))

        gray = cv2.cvtColor(warped_overlay, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img_bg = cv2.bitwise_and(image, image, mask=mask_inv)
        final = cv2.add(img_bg, warped_overlay)

        cv2.imwrite(output_image_path, final)
        print(f"img saved to {output_image_path}")
    else:
        print(f"couldn't overlay {input_image_path}")


def overlay_on_video(marker_video_path, overlay_video_path, output_video_path):
    marker_cap = cv2.VideoCapture(marker_video_path)
    overlay_cap = cv2.VideoCapture(overlay_video_path)

    if not marker_cap.isOpened():
        raise FileNotFoundError(f"couldn't open marker video {marker_video_path}.")
    if not overlay_cap.isOpened():
        raise FileNotFoundError(f"couldn't open overlay video {overlay_video_path}.")

    # save as mp4 since downloading avi player is hard
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_fps = marker_cap.get(cv2.CAP_PROP_FPS)
    out_width = int(marker_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_height = int(marker_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_video_path, fourcc, out_fps, (out_width, out_height))

    while True:
        ret_marker, marker_frame = marker_cap.read()
        ret_overlay, overlay_frame = overlay_cap.read()

        if not ret_marker:
            break
        if not ret_overlay:
            break

        # warp perspective
        dst_points = detect_markers(marker_frame)
        if dst_points is not None:
            oh, ow, _ = overlay_frame.shape
            src_points = np.array([
                [0, 0],
                [ow - 1, 0],
                [ow - 1, oh - 1],
                [0, oh - 1]
            ], dtype="float32")

            H, _ = cv2.findHomography(src_points, dst_points)
            warped_overlay = cv2.warpPerspective(overlay_frame, H, (marker_frame.shape[1], marker_frame.shape[0]))

            gray = cv2.cvtColor(warped_overlay, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            img_bg = cv2.bitwise_and(marker_frame, marker_frame, mask=mask_inv)
            final_frame = cv2.add(img_bg, warped_overlay)

            out.write(final_frame)
        else:
            # if everything fails just output original
            out.write(marker_frame)

    marker_cap.release()
    overlay_cap.release()
    out.release()

    print(f"saved video to {output_video_path}")

overlay_on_image("marker_angle1.jpg", "overlay_image.jpg", "overlayed_img1.jpg")
overlay_on_image("marker_angle2.jpg", "overlay_image.jpg", "overlayed_img2.jpg")
overlay_on_image("marker_angle3.jpg", "overlay_image.jpg", "overlayed_img3.jpg")
overlay_on_image("marker_angle4.jpg", "overlay_image.jpg", "overlayed_img4.jpg")
overlay_on_video("marker_video.avi", "overlay_video.avi", "overlayed_video.mp4")
