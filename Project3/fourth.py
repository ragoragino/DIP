import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Set current directory to the directory of the file
cur_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(cur_dir)
np.set_printoptions(threshold=np.nan)


class WeightedMedianFilter:
    def __init__(self, image, filter):
        self.image_height, self.image_width = image.shape
        filter_height, filter_width = filter.shape

        assert (filter_height == filter_width), "Filter must be square!"
        assert (filter_height % 2 != 0), "Filter must be odd!"
        assert (image.dtype == np.uint8), "Image must be bool!"
        assert (filter.dtype == np.uint8), "Filter must be bool!"

        self.image = image
        self.filter = filter
        self.filter_length = filter_height

        # Pre-allocate median holder buffer
        self.median_holder = np.zeros(np.sum(self.filter), dtype=np.uint8)

        # Create an expanded image for easy filtering
        self.expanded_image = np.zeros((self.image_height + self.filter_length - 1,
                                       self.image_width + self.filter_length - 1),
                                       dtype=np.uint8)

        filter_length_half = int(self.filter_length / 2)

        self.expanded_image[filter_length_half:(self.image_height + filter_length_half),
        filter_length_half:(self.image_width + filter_length_half)] = self.image

    def filterImage(self):
        filtered_image = np.zeros((self.image_height, self.image_width), dtype=np.uint8)

        filter_length_half = int(self.filter_length / 2)

        # Allocate block holding subsequent image blocks in the loop
        current_block = np.zeros((self.filter_length, self.filter_length), dtype=np.uint8)

        # Make the median filter operation
        for i in range(filter_length_half,
                       self.image_height + filter_length_half):
            for j in range(filter_length_half,
                           self.image_width + filter_length_half):
                current_block[:, :] = \
                    self.expanded_image[(i - filter_length_half):(i + filter_length_half + 1),
                                        (j - filter_length_half):(j + filter_length_half + 1)]

                filtered_image[i - filter_length_half, j - filter_length_half] = \
                    self.__medianFilterOperation(current_block)

        return filtered_image

    def __medianFilterOperation(self, array):
        pointer = 0
        for i in range(self.filter_length):
            for j in range(self.filter_length):
                current_value = self.filter[i, j]
                self.median_holder[pointer:(pointer + current_value)] = array[i, j]
                pointer += current_value

        return np.median(self.median_holder)


if __name__ == '__main__':

    image_building = cv.imread(cur_dir + r'\data\hw3_building.jpg',
                               cv.IMREAD_GRAYSCALE)

    image_train = cv.imread(cur_dir + r'\data\hw3_train.jpg',
                            cv.IMREAD_GRAYSCALE)

    """
    a)
    """
    image_building_median5 = cv.medianBlur(image_building, 5)
    image_building_median3 = cv.medianBlur(image_building, 3)

    image_train_median5 = cv.medianBlur(image_train, 5)
    image_train_median3 = cv.medianBlur(image_train, 3)

    plt.subplot(1, 3, 1), plt.imshow(image_building, cmap='gray')
    plt.title("Image building")
    plt.subplot(1, 3, 2), plt.imshow(image_building_median3, cmap='gray')
    plt.title("Image building with median filter 3")
    plt.subplot(1, 3, 3), plt.imshow(image_building_median5, cmap='gray')
    plt.title("Image building with median filter 5")
    plt.show()

    plt.subplot(1, 3, 1), plt.imshow(image_train, cmap='gray')
    plt.title("Image building")
    plt.subplot(1, 3, 2), plt.imshow(image_train_median3, cmap='gray')
    plt.title("Image building with median filter 3")
    plt.subplot(1, 3, 3), plt.imshow(image_train_median5, cmap='gray')
    plt.title("Image building with median filter 5")
    plt.show()

    """
    b)
    """
    filter = np.array([[0, 1, 1, 1, 0],
                       [1, 2, 2, 2, 1],
                       [1, 2, 4, 2, 1],
                       [1, 2, 2, 2, 1],
                       [0, 1, 1, 1, 0]], dtype=np.uint8)

    building_weighted_median_filter = WeightedMedianFilter(image_building, filter)
    building_weighted_median_result = building_weighted_median_filter.filterImage()
    plt.subplot(1, 3, 1), plt.imshow(image_building, cmap='gray')
    plt.title("Image building")
    plt.subplot(1, 3, 2), plt.imshow(image_building_median3, cmap='gray')
    plt.title("Image building with median filter 3")
    plt.subplot(1, 3, 3), plt.imshow(building_weighted_median_result, cmap='gray')
    plt.title("Image building with weighted median filter 5")
    plt.show()

    train_weighted_median_filter = WeightedMedianFilter(image_train, filter)
    train_weighted_median_result = train_weighted_median_filter.filterImage()
    plt.subplot(1, 3, 1), plt.imshow(image_train, cmap='gray')
    plt.title("Image building")
    plt.subplot(1, 3, 2), plt.imshow(image_train_median3, cmap='gray')
    plt.title("Image building with median filter 3")
    plt.subplot(1, 3, 3), plt.imshow(train_weighted_median_result, cmap='gray')
    plt.title("Image building with weighted median filter 5")
    plt.show()