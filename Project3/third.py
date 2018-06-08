import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Set current directory to the directory of the file
cur_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(cur_dir)
np.set_printoptions(threshold=np.nan)

if __name__ == '__main__':

    # Load training and testing images
    """
    a)
    """
    image = cv.imread(cur_dir + r'\data\hw3_road_sign_school_blurry.jpg ',
                      cv.IMREAD_GRAYSCALE)

    NUMBER_ITER = 10
    ROW = 338
    row_intensity = np.zeros((NUMBER_ITER, image.shape[1]), dtype=np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
    for i in range(NUMBER_ITER):
        image_d = cv.dilate(image.astype(np.float32),
                            kernel.astype(np.uint8))

        image_e = cv.erode(image.astype(np.float32),
                           kernel.astype(np.uint8))

        image_h = 0.5 * (image_d + image_e)

        image = np.where(image > image_h, image_d, image_e)

        row_intensity[i, :] = image[ROW, :]

    plt.imshow(image, cmap='gray')
    plt.title("Image after iterative morphological change!")
    plt.show()

    """
    b)
    """

    for i in range(NUMBER_ITER):
        plt.subplot(NUMBER_ITER, 1, i + 1)
        plt.plot(row_intensity[i, :])

    plt.show()