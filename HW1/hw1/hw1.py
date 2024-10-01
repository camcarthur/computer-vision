import cv2
import numpy as np


def read_image(image_path: str) -> np.ndarray:
    """
    This function reads an image and returns it as a numpy array
    :param image_path: String of path to file
    :return img: Image array as ndarray
    """
    img = cv2.imread(image_path)
    return img


def extract_green(img: np.ndarray) -> np.ndarray:
    """
    This function takes an image and returns the green channel
    :param img: Image array as ndarray
    :return: Image array as ndarray of just green channel
    """
    green_channel=img[ :, :, 1]
    return green_channel


def extract_red(img: np.ndarray) -> np.ndarray:
    """
    This function takes an image and returns the red channel
    :param img: Image array as ndarray
    :return: Image array as ndarray of just red channel
    """
    red_channel=img[ :, :, 2]
    return red_channel


def extract_blue(img: np.ndarray) -> np.ndarray:
    """
    This function takes an image and returns the blue channel
    :param img: Image array as ndarray
    :return: Image array as ndarray of just blue channel
    """
    blue_channel=img[ :, :, 0]
    return blue_channel


def swap_red_green_channel(img: np.ndarray) -> np.ndarray:
    """
    This function takes an image and returns the image with the red and green channel
    :param img: Image array as ndarray
    :return: Image array as ndarray of red and green channels swapped
    """
    swapped_img = img.copy()
    swapped_img[:, :, 1], swapped_img[:, :, 2] = img[:, :, 2], img[:, :, 1]
        
    return swapped_img


def embed_middle(img1: np.ndarray, img2: np.ndarray, embed_size: (int, int)) -> np.ndarray:
    """
    This function takes two images and embeds the embed_size pixels from img2 onto img1
    :param img1: Image array as ndarray
    :param img2: Image array as ndarray
    :param embed_size: Tuple of size (width, height)
    :return: Image array as ndarray of img1 with img2 embedded in the middle
    """
    src_h, src_w, _ = img2.shape
    center_x, center_y = src_w // 2, src_h // 2
    
    start_x = center_x - 30
    start_y = center_y - 30
    end_x = center_x + 30
    end_y = center_y + 30
    
    cropped_region = img2[start_y:end_y, start_x:end_x]
    
    dst_h, dst_w, _ = img2.shape
    center_x_dst, center_y_dst = dst_w // 2, dst_h // 2
    
    start_x_dst = center_x_dst - 30
    start_y_dst = center_y_dst - 30
    end_x_dst = center_x_dst + 30
    end_y_dst = center_y_dst + 30
    
    img1[start_y_dst:end_y_dst, start_x_dst:end_x_dst] = cropped_region

    return img1


def calc_stats(img: np.ndarray) -> np.ndarray:
    """
    This function takes an image and returns the mean and standard deviation
    :param img: Image array as ndarray
    :return: Numpy array with mean and standard deviation in that order
    """
    mean, stddev = cv2.meanStdDev(img)
    mean_value = mean.flatten()
    stddev_value = stddev.flatten()

    return np.array([mean_value, stddev_value])



def shift_image(img: np.ndarray, shift_val: int) -> np.ndarray:
    """
    This function takes an image and returns the image shifted by shift_val pixels to the right.
    Should have an appropriate border for the shifted area:
    https://docs.opencv.org/3.4/dc/da3/tutorial_copyMakeBorder.html

    Returned image should be the same size as the input image.
    :param img: Image array as ndarray
    :param shift_val: Value to shift the image
    :return: Shifted image as ndarray
    """
    rows, cols = img.shape[:2]
    bordered_img = cv2.copyMakeBorder(img, 
                                      top=0, 
                                      bottom=0, 
                                      left=shift_val, 
                                      right=0, 
                                      borderType=cv2.BORDER_CONSTANT, 
                                      value=[0, 0, 0])
    shifted_img = bordered_img[:, :cols]

    return shifted_img


def difference_image(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    This function takes two images and returns the first subtracted from the second.
    The resulting image is normalized to the range [0, 255].
    
    :param img1: Image array as ndarray
    :param img2: Image array as ndarray
    :return: Normalized difference image as ndarray
    """
    # Ensure the images are of the same size and type
    if img1.shape != img2.shape:
        raise ValueError("Both images must have the same dimensions and number of channels.")
    
    # Compute the difference between img2 and img1
    diff = cv2.subtract(img2, img1)
    
    # Normalize the difference image to the range [0, 255]
    norm_diff = np.zeros_like(diff)
    cv2.normalize(diff, norm_diff, 0, 255, cv2.NORM_MINMAX)

    return norm_diff


def add_channel_noise(img: np.ndarray, channel: int, sigma: int) -> np.ndarray:
    """
    This function takes an image and adds noise to the specified channel.

    Should probably look at randn from numpy

    Make sure the image to return is normalized:
    https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga87eef7ee3970f86906d69a92cbf064bd

    :param img: Image array as ndarray
    :param channel: Channel to add noise to
    :param sigma: Gaussian noise standard deviation
    :return: Image array with gaussian noise added
    """
    noisy_img = img.copy()
    noise = np.random.randn(img.shape[0], img.shape[1]) * sigma
    noisy_channel = noisy_img[:, :, channel] + noise
    noisy_img[:, :, channel] = np.clip(noisy_channel, 0, 255)
    
    noisy_img = noisy_img.astype(np.uint8)
    
    return noisy_img


def add_salt_pepper(img: np.ndarray) -> np.ndarray:
    """
    This function takes an image and adds salt and pepper noise.

    Must only work with grayscale images
    :param img: Image array as ndarray
    :return: Image array with salt and pepper noise
    """

    noisy_img = img.copy()
    num_salt = 5000
    num_pepper = 5000
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
    noisy_img[salt_coords[0], salt_coords[1]] = 255  # Set to white
    
    # Add pepper (black pixels) at random locations
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape]
    noisy_img[pepper_coords[0], pepper_coords[1]] = 0  # Set to black

    return noisy_img


def blur_image(img: np.ndarray, ksize: int) -> np.ndarray:
    """
    This function takes an image and returns the blurred image

    https://docs.opencv.org/4.x/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html
    :param img: Image array as ndarray
    :param ksize: Kernel Size for medianBlur
    :return: Image array with blurred image
    """
    blurred_img = cv2.medianBlur(img, ksize)
    return blurred_img

def run_functions(img_path:str):
    img = read_image(img_path)
    # Step 2: Extract each color channel
    green_channel = extract_green(img)
    red_channel = extract_red(img)
    blue_channel = extract_blue(img)
    
    # Step 3: Swap red and green channels
    swapped_img = swap_red_green_channel(img)
    
    # Step 4: Embed part of one image into another
    # For simplicity, embed img into itself with a 60x60 pixel area
    embedded_img = embed_middle(img, img, (60, 60))
    
    # Step 5: Calculate image statistics (mean and standard deviation)
    stats = calc_stats(img)
    print(f"Mean and Standard Deviation: {stats}")
    
    # Step 6: Shift the image to the right by 50 pixels
    shifted_img = shift_image(img, 50)
    
    # Step 7: Subtract two images and normalize the result
    difference_img = difference_image(img, img)  # Self-subtraction for simplicity
    
    # RED
    rnoisy_img = add_channel_noise(img, 2, sigma=25)
    # BLUE
    bnoisy_img = add_channel_noise(img, 0, sigma=25)
    # GREEN
    gnoisy_img = add_channel_noise(img, 1, sigma=25)
    
    # Step 9: Add salt and pepper noise to a grayscale version of the image
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noisy_salt_pepper_img = add_salt_pepper(grayscale_img)
    
    # Step 10: Blur the image using a median blur with a kernel size of 5
    blurred_img = blur_image(noisy_salt_pepper_img, 5)
    
    # Display results using OpenCV (optional)
    # cv2.imshow("Original Image", img)
    # cv2.imshow("Green Channel", green_channel)
    # cv2.imshow("Red Channel", red_channel)
    # cv2.imshow("Blue Channel", blue_channel)
    # cv2.imshow("Swapped Red-Green Channels", swapped_img)
    # cv2.imshow("Embedded Image", embedded_img)
    # cv2.imshow("Shifted Image", shifted_img)
    # cv2.imshow("Difference Image", difference_img)
    cv2.imshow("Noisy Red Channel Image", rnoisy_img)
    cv2.imshow("Noisy Blue Channel Image", bnoisy_img)
    cv2.imshow("Noisy Green Channel Image", gnoisy_img)
    # cv2.imshow("Noisy Salt & Pepper Image", noisy_salt_pepper_img)
    # cv2.imshow("Blurred Image", blurred_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

run_functions('hw1_pic1.jpg')
