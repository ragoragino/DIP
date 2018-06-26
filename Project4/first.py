import cv2 as cv
import os
from matplotlib import pyplot as plt
import numpy as np

# Set current directory to the directory of the file
cur_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(cur_dir)

if __name__ == '__main__':
    """
    a)
    """
    # Load the image, execute median filtering and plot the originals together with filtered images
    image1 = cv.imread(cur_dir + r'/data/hw4_radiograph_1.jpg', cv.IMREAD_GRAYSCALE)
    image2 = cv.imread(cur_dir + r'/data/hw4_radiograph_2.jpg', cv.IMREAD_GRAYSCALE)

    medianImage1 = cv.medianBlur(image1, 11)
    medianImage2 = cv.medianBlur(image2, 11)

    plt.subplot(1, 2, 1), plt.imshow(image1, 'gray'), plt.title("Original image")
    plt.subplot(1, 2, 2), plt.imshow(medianImage1, 'gray'), \
    plt.title("Image after median filtering")
    plt.show()

    plt.subplot(1, 2, 1), plt.imshow(image2, 'gray'), plt.title("Original image")
    plt.subplot(1, 2, 2), plt.imshow(medianImage2, 'gray'), \
    plt.title("Image after median filtering")
    plt.show()

    """
    b)
    """
    # Execute FFT and plot images after it
    f1 = np.fft.fft2(image1)
    fshiftImage1 = np.fft.fftshift(f1)
    magnitude_spectrum1 = np.log(np.abs(fshiftImage1))

    f2 = np.fft.fft2(image2)
    fshiftImage2 = np.fft.fftshift(f2)
    magnitude_spectrum2 = np.log(np.abs(fshiftImage2))

    plt.subplot(1, 1, 1), plt.imshow(magnitude_spectrum1, 'gray'),\
    plt.title("FFT of the first image")
    plt.show()

    plt.subplot(1, 1, 1), plt.imshow(magnitude_spectrum2, 'gray'),\
    plt.title("FFT of the second image")
    plt.show()

    """
    c)
    """
    # Centers and ranges of points suggesting Moire patterns in images
    # Manually extracted from FFT plots
    FIRST_ERASE_POINT = [(215, 245), (124, 260)]
    FIRST_ERASE_RANGE = [10, 10]
    SECOND_ERASE_POINT = [(450, 210), (440, 280)]
    SECOND_ERASE_RANGE = [10, 10]

    # Set the neighbourhoods of points suggesting Moire pattern specified above to the median values of FFTs
    for i in range(len(FIRST_ERASE_POINT)):
        fshiftImage1[FIRST_ERASE_POINT[i][0] - FIRST_ERASE_RANGE[0] : FIRST_ERASE_POINT[i][0] + FIRST_ERASE_RANGE[0],
        FIRST_ERASE_POINT[i][1] - FIRST_ERASE_RANGE[1]: FIRST_ERASE_POINT[i][1] + FIRST_ERASE_RANGE[1]] = \
            np.median(fshiftImage1)

    # Execute inverse FFT
    f_ishift1 = np.fft.ifftshift(fshiftImage1)
    image1_back = np.fft.ifft2(f_ishift1)
    image1_back = np.abs(image1_back)

    # Run the same procedure on the second image
    for i in range(len(SECOND_ERASE_POINT)):
        fshiftImage2[SECOND_ERASE_POINT[i][0] - SECOND_ERASE_RANGE[0] : SECOND_ERASE_POINT[i][0] + SECOND_ERASE_RANGE[0],
        SECOND_ERASE_POINT[i][1] - SECOND_ERASE_RANGE[1]: SECOND_ERASE_POINT[i][1] + SECOND_ERASE_RANGE[1]] = \
            np.median(fshiftImage2)

    f_ishift2 = np.fft.ifftshift(fshiftImage2)
    image2_back = np.fft.ifft2(f_ishift2)
    image2_back = np.abs(image2_back)

    # Plot the results
    plt.subplot(1, 2, 1), plt.imshow(image1_back, 'gray'),\
    plt.title("Inverse FFT of the first image after masking")
    plt.subplot(1, 2, 2), plt.imshow(image1, 'gray'), plt.title("Original image")
    plt.show()

    plt.subplot(1, 2, 1), plt.imshow(image2_back, 'gray'),\
    plt.title("Inverse FFT of the second image after masking")
    plt.subplot(1, 2, 2), plt.imshow(image2, 'gray'), plt.title("Original image")
    plt.show()