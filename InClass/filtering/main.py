import cv2
import numpy as np
import matplotlib.pyplot as plt

def main() -> None:
    window = "BlurWindow"
    img = cv2.imread('tyu.jpg')

    gaussian_kernel = np.array([
        [1,2,1],
        [2,4,2],
        [1,2,1]
    ], dtype=np.float64)/16
    print(gaussian_kernel)

    blurred_image = cv2.filter2D(img, -1, gaussian_kernel)

    cv2.imshow(window, img)
    cv2.imshow(window+"2", blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass

if __name__ == '__main__':
    main()