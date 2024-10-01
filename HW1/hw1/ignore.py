# IGNORE THIS ONE
# I DID IT ALL MANUALLY AND WRONG AT FIRST

import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_window() -> np.ndarray:
    img = 'hw1_pic1.jpg'
    pic1 = cv2.imread(img)

    return pic1


img = create_window()


def ontrackbar_changed(val) -> None:
    img1 = 'hw1_pic1.jpg'
    img2 = 'hw1_pic2.jpg'
    pic1 = cv2.imread(img1)

    green = pic1[ :, :, 1]
    # shift = 40

    # h, w, _ = pic1.shape  # for gray image
    # shift = 100  # any legal number 0 < x < h

    # pic1[:h - shift, :] = pic1[shift:, :]
    # pic1[h - shift:, :] = 0


    pic2 = cv2.imread(img2)
    src_h, src_w, _ = pic2.shape
    center_x, center_y = src_w // 2, src_h // 2
    
    start_x = center_x - 30
    start_y = center_y - 30
    end_x = center_x + 30
    end_y = center_y + 30
    
    # Crop the 60x60 region from the source image
    cropped_region = pic2[start_y:end_y, start_x:end_x]
    
    # Step 2: Determine the center of the destination image
    dst_h, dst_w, _ = pic1.shape
    center_x_dst, center_y_dst = dst_w // 2, dst_h // 2
    
    # Calculate the coordinates to place the cropped region
    start_x_dst = center_x_dst - 30
    start_y_dst = center_y_dst - 30
    end_x_dst = center_x_dst + 30
    end_y_dst = center_y_dst + 30
    
    # Step 3: Embed the cropped region into the destination image
    pic1[start_y_dst:end_y_dst, start_x_dst:end_x_dst] = cropped_region

    # pic1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # mean, stddev = cv2.meanStdDev(pic1)

    # mean_value = mean[0][0]
    # stddev_value = stddev[0][0]

    # print(f"Mean: {mean_value}")
    # print(f"Standard Deviation: {stddev_value}")
 
    cv2.imshow("HW1", green)
    return


def main() -> None:
    window = "HW1"

    cv2.namedWindow(window)
    #
    # cv2.createTrackbar("sigma", window, 1, 40, ontrackbar_changed)
    # cv2.createTrackbar("threshold", window, 1, 255, ontrackbar_changed)

    # ontrackbar_changed(0)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    pass

if __name__ == '__main__':
    main()