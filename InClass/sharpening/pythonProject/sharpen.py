import cv2
import numpy as np

def main() -> None:
    img = cv2.imread('tyu.jpg', cv2.IMREAD_GRAYSCALE)

    blurred_img = cv2.GaussianBlur(img, (5,5),1)

    mask = cv2.subtract(img, blurred_img)

    scale_val = 1.5

    sharp_img = img.astype(np.float64) + scale_val * mask.astype(np.float64)
    cv2.normalize(sharp_img, sharp_img, 0,255,cv2.NORM_MINMAX)
    sharp_img = sharp_img.astype(np.uint8)
    # print(sharp_img.astype(np.uint8))

    cv2.imshow('sharp_img', sharp_img)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
