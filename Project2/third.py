import cv2 as cv
import os
from matplotlib import pyplot as plt
import numpy as np

# Set current directory to the directory of the file
cur_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(cur_dir)

MAX_VAR_PARAMETER = 0.5 * 255.0 ** 2 - 127.5 ** 2
MEAN_PARAMETER = 127.5

"""
Algorithm for Otsu thresholding
"""
def otsu(image):
    threshold = 0
    max_var = 0
    height, width = image.shape
    image_size = height * width

    histogram, _ = np.histogram(image, bins=np.arange(-0.5, 256.5, 1.0))
    for i in range(1, 255):
        below_size = np.sum(histogram[:i]) / image_size
        below_mean = np.sum(np.multiply(histogram[:i], np.arange(i))) / np.sum(histogram[:i])
        above_size = np.sum(histogram[i:]) / image_size
        above_mean = np.sum(np.multiply(histogram[i:], range(i, 256))) / np.sum(histogram[i:])

        between_var = below_size * above_size * \
                      (below_mean - above_mean) ** 2

        if between_var > max_var:
            max_var = between_var
            threshold = i

    return threshold

"""
Algorithm for locally adaptive Otsu thresholding with a parameter of size of a block
and a ratio of the maximal variance
"""
def locallyAdaptiveThresholding(image, block_size, var_ratio):
    img = np.array(image, copy=True)

    rows, cols = img.shape
    x_pass = rows // block_size
    y_pass = cols // block_size

    for i in range(x_pass + 1):
        x_left = i * block_size
        x_right = min(i * block_size + block_size, rows)

        if x_left == x_right:
            break

        for j in range(y_pass + 1):
            y_left = j * block_size
            y_right = min(j * block_size + block_size, cols)

            if y_left == y_right:
                break

            block = img[x_left:x_right, y_left:y_right]
            block_var = np.var(block)

            if block_var > var_ratio * MAX_VAR_PARAMETER:
                threshold = otsu(block)
                block = (block >= threshold) * 255
            else:
                block_mean = np.mean(block)
                if block_mean >= MEAN_PARAMETER:
                    block = 255
                else:
                    block = 0

            img[x_left:x_right, y_left:y_right] = block

    return img

if __name__ == '__main__':
    # Load the images
    img1 = cv.imread(cur_dir + r'/data/hw2_book_page_1.jpg', cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(cur_dir + r'/data/hw2_book_page_2.jpg', cv.IMREAD_GRAYSCALE)

    # Apply Otsu and plot
    retval1 = otsu(img1)
    dst1 = (img1 >= retval1) * 255

    retval2 = otsu(img2)
    dst2 = (img2 >= retval2) * 255

    plt.subplot(2, 2, 1), plt.imshow(img1, 'gray')
    plt.subplot(2, 2, 2), plt.imshow(dst1, 'gray')
    plt.subplot(2, 2, 3), plt.imshow(img2, 'gray')
    plt.subplot(2, 2, 4), plt.imshow(dst2, 'gray')
    plt.show()

    # Plot original histograms and threshold from Otsu algorithm
    plt.subplot(2, 1, 1), plt.hist(img1.ravel(), 256)
    plt.axvline(x=retval1, color='r', linestyle='-')
    plt.subplot(2, 1, 2), plt.hist(img2.ravel(), 256)
    plt.axvline(x=retval2, color='r', linestyle='-')
    plt.show()

    # Apply locally adaptive thresholding and plot the resulting images
    th1 = locallyAdaptiveThresholding(img1, 10, 0.005)
    th2 = locallyAdaptiveThresholding(img2, 10, 0.05)

    plt.subplot(2, 2, 1), plt.imshow(img1, 'gray')
    plt.subplot(2, 2, 2), plt.imshow(th1, 'gray')
    plt.subplot(2, 2, 3), plt.imshow(img2, 'gray')
    plt.subplot(2, 2, 4), plt.imshow(th2, 'gray')
    plt.show()
