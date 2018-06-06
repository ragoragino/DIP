import os, sys
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from string import ascii_lowercase

# Set current directory to the directory of the file
cur_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(cur_dir)
np.set_printoptions(threshold=np.nan)

class Detector:
    def __init__(self, image, filter):
        self.image = image
        self.filter = filter

class HitDetector(Detector):
    def __init__(self, image, filter):
        filter_height, filter_width = filter.shape

        assert (filter_height == filter_width), "Filter must be square!"
        assert (image.dtype == np.bool), "Image must be bool!"
        assert (filter.dtype == np.bool), "Filter must be bool!"

        super().__init__(image, filter)

    def erode(self):
        erosion = cv.erode(self.image.astype(np.float32),
                           self.filter.astype(np.uint8))

        return erosion

    def dilate(self, erosion):
        flipped_kernel = np.flip(np.flip(self.filter, 0), 1)

        dilation = cv.dilate(erosion.astype(np.float32),
                             flipped_kernel.astype(np.uint8))

        return dilation


class HitMissDetector(Detector):
    def __init__(self, image, foreground_filter, background_filter):
        filter_height, filter_width = foreground_filter.shape
        assert (filter_height == filter_width), "Foreground filter must be square!"

        filter_height, filter_width = background_filter.shape
        assert (filter_height == filter_width), "Background filter must be square!"

        assert (image.dtype == np.bool), "Image must be bool!"
        assert (foreground_filter.dtype == np.bool), "Foreground filter must be bool!"
        assert (background_filter.dtype == np.bool), "Background filter must be bool!"

        super().__init__(image, foreground_filter)
        self.background_filter = background_filter

    def erode(self):
        erosion_1 = cv.erode(self.image.astype(np.float32),
                             self.filter.astype(np.uint8))

        erosion_2 = cv.erode((self.image == False).astype(np.float32),
                             self.background_filter.astype(np.uint8))

        result = np.multiply(erosion_1, erosion_2)

        return result

    def dilate(self, erosion):
        flipped_kernel = np.flip(np.flip(self.filter, 0), 1)

        dilation = cv.dilate(erosion.astype(np.float32),
                             flipped_kernel.astype(np.uint8))

        return dilation


class RankFilter:
    def __init__(self, image, filter):
        self.image_height, self.image_width = image.shape
        self.filter_height, self.filter_width = filter.shape

        assert (self.filter_height == self.filter_width), "Filter must be square!"
        assert (image.dtype == np.bool), "Image must be bool!"
        assert (filter.dtype == np.bool), "Filter must be bool!"

        self.image = image

        # Check for odd cases
        if self.filter_height % 2 == 0:
            self.filter = self.__addRowAndColumn(filter)

            _, self.filter_length = self.filter.shape
        else:
            self.filter = filter
            _, self.filter_length = self.filter.shape

        # Create an expanded image for easy filtering
        self.expanded_image = np.zeros((self.image_height + self.filter_length - 1,
                                       self.image_width + self.filter_length - 1),
                                       dtype=np.bool)

        filter_length_half = int(self.filter_length / 2)

        self.expanded_image[filter_length_half:(self.image_height + filter_length_half),
        filter_length_half:(self.image_width + filter_length_half)] = self.image

    def __addRowAndColumn(self, filter):
        tmp = np.zeros((self.filter_height + 1, self.filter_height + 1), dtype=np.bool)
        tmp[1:, 1:] = filter

        # Interpolate values to the first column and row
        for i in range(1, self.filter_height + 1):
            tmp[i, 0] = filter[i - 1, 0]
            tmp[0, i] = filter[0, i - 1]

        tmp[0, 0] = 0.5 * (tmp[1, 0] + tmp[0, 1])

        return tmp

    def filterImage(self, p):
        filtered_image = np.zeros((self.image_height, self.image_width), dtype=np.bool)

        assert (0 <= p <= 100), "Percentile must be between 0 and 100 inclusive"

        filter_length_half = int(self.filter_length / 2)

        # Allocate block holding subsequent image blocks in the loop
        current_block = np.zeros((self.filter_length, self.filter_length), dtype=np.bool)

        # Find the p-th percentile of the filtered shape in all image blocks
        indices = np.where(self.filter == True)
        for i in range(filter_length_half,
                       self.image_height + filter_length_half):
            for j in range(filter_length_half,
                           self.image_width + filter_length_half):
                current_block[:, :] = \
                    self.expanded_image[(i - filter_length_half):(i + filter_length_half + 1),
                                        (j - filter_length_half):(j + filter_length_half + 1)]

                filtered_image[i - filter_length_half, j - filter_length_half] = \
                    self.__binaryPercentile(current_block[indices], p)

        return filtered_image

    def __binaryPercentile(self, array, p):
        percentile = 1.0 - p / 100.0
        border = np.sum(array) / array.shape[0]
        if border > percentile:
            return 1
        else:
            return 0


if __name__ == '__main__':

    """
    a)
    """
    # Load images
    training_images = list()
    training_images.append(cv.imread(cur_dir + r'\data\hw3_license_plate_clean.png',
                                     cv.IMREAD_GRAYSCALE))
    training_images.append(cv.imread(cur_dir + r'\data\hw3_license_plate_noisy.png',
                                     cv.IMREAD_GRAYSCALE))

    binary_training_images = []
    threshold, image = cv.threshold(training_images[0], 0, 255,
                                    cv.THRESH_BINARY + cv.THRESH_OTSU)
    binary_training_images.append(image == 0)
    binary_training_images.append(training_images[1] < threshold)

    for i in range(len(binary_training_images)):
        plt.imshow(binary_training_images[i], 'gray')
        plt.title("Original image!")
        plt.show()

    # Load templates
    base_template = np.ones((3, 3), dtype=np.uint8)

    number_templates = []
    for i in range(10):
        image = cv.imread(cur_dir + r'\data\templates\\' + str(i) + '.png',
                          cv.IMREAD_GRAYSCALE)

        _, image = cv.threshold(image, 0, 255,
                                cv.THRESH_BINARY + cv.THRESH_OTSU)

        image = image == 0
        erosion = cv.erode(image.astype(np.float32), base_template)

        number_templates.append(erosion)

    alphabet_templates = []
    for s in ascii_lowercase.upper():
        image = cv.imread(cur_dir + r'\data\templates\\' + s + '.png',
                          cv.IMREAD_GRAYSCALE)

        _, image = cv.threshold(image, 0, 255,
                                cv.THRESH_BINARY + cv.THRESH_OTSU)

        image = image == 0
        erosion = cv.erode(image.astype(np.float32), base_template)

        alphabet_templates.append(erosion)

    """
    b)
    """

    # Find numbers or letters in the clean image with hit filter
    for template in number_templates + alphabet_templates:
        detector = HitDetector(binary_training_images[0], template.astype(np.bool))
        erosion = detector.erode()

        if np.any(erosion == True):
            dilation = detector.dilate(erosion)

            plt.imshow(dilation, vmin=0, vmax=1, cmap='gray')
            plt.title("Detected templates in the clean image with hit filter")
            plt.show()


    """
    c)
    """
    # Find numbers or letters in the clean image with hit-miss filter
    # It seems to work better for 7 and 5 square filters, than for 5 and 3
    square7 = np.ones((7, 7), dtype=np.uint8)
    square5 = np.ones((5, 5), dtype=np.uint8)

    for template in number_templates + alphabet_templates:
        background_detector = cv.dilate(template.astype(np.float32), square7) - \
                              cv.dilate(template.astype(np.float32), square5)

        detector = HitMissDetector(binary_training_images[0],
                                   template.astype(np.bool),
                                   background_detector.astype(np.bool))

        erosion = detector.erode()

        if np.any(erosion == True):
            dilation = detector.dilate(erosion)

            plt.imshow(dilation, vmin=0, vmax=1, cmap='gray')
            plt.title("Detected templates in the clean image with hit-miss filter")
            plt.show()

    """
    d)
    """
    # Find numbers or letters in the noisy image with hit-miss filter
    for template in number_templates + alphabet_templates:
        background_detector = cv.dilate(template.astype(np.float32), square7) - \
                              cv.dilate(template.astype(np.float32), square5)

        detector = HitMissDetector(binary_training_images[1],
                                   template.astype(np.bool),
                                   background_detector.astype(np.bool))

        erosion = detector.erode()

        if np.any(erosion == True):
            dilation = detector.dilate(erosion)

            plt.imshow(dilation, cmap='gray')
            plt.title("Detected templates in the noisy image with hit-miss filter")
            plt.show()

    """
    e)
    """
    RANK_VALUE = 5
    for template in number_templates + alphabet_templates:
        # Create background detector
        background_detector = cv.dilate(template.astype(np.float32), square7) - \
                              cv.dilate(template.astype(np.float32), square5)

        # Create background and foreground rank detector and filter the image
        rank_fore = RankFilter(binary_training_images[0], template.astype(np.bool))
        detector_fore = rank_fore.filterImage(RANK_VALUE)

        rank_back = RankFilter(binary_training_images[1] == False,
                               background_detector.astype(np.bool))
        detector_back = rank_back.filterImage(RANK_VALUE)

        # Take the minimum of the foreground and background detectors
        result = np.minimum(detector_fore, detector_back)

        # Plot the result, if any non-zero point is detected
        if np.any(result == True):
            h, w = template.shape
            half = int(h / 2)
            indices = np.where(result == True)
            result[(indices[0][0] - half):(indices[0][0] + half),
            (indices[1][0] - half):(indices[1][0] + half)] = template

            plt.imshow(result, cmap='gray')
            plt.title("Detected templates in the noisy image with rank filter and "
                      "p equal to {}".format(RANK_VALUE))
            plt.show()