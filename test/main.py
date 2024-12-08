import cv2
import numpy as np
import matplotlib.pyplot


def main() -> None:
    img = cv2.imread('File.jpg', cv2.IMREAD_GRAYSCALE)
    # print(f"Image Size: {img.shape}")
    img_slice = img[200:600, :, :]
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.figure()
    # plt.matshow(img)
    # plt.show()
    x = np.linspace(len(img[0]),0, len(img[0]))
    y = np.linspace(0, len(img), len(img))
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(dpi=len(img)/2)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, img, cmap='Greys', linewidth=1, antialiased=True, alpha=1)


    cv2.imshow('image', img_slice)
    cv2.waitKey(0)
    pass