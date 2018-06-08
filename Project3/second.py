import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from shapely.geometry import LineString


# Set current directory to the directory of the file
cur_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(cur_dir)
np.set_printoptions(threshold=np.nan)


"""
ALGORITHM:
1. With difference of two dilations, obtain an edge of the object
2. Find a convex hull of the edge
3. Find a centre of the polygon of the hull: https://en.wikipedia.org/wiki/Centroid#Centroid_of_a_polygon
4. Find q points that have max distance from the centre 
5. Measure the percentage of hit points for the square hit detector applied to test and train images
   for each q orientation to the orientation of centre - max point
6. Return an average of the maximum of the two numbers from the set of q
"""

if __name__ == '__main__':

    # Load training and testing images
    training_images = list()
    for i in range(1, 6):
        image = cv.imread(cur_dir + r'\data\hw3_leaf_training_' + str(i) + '.jpg',
                                         cv.IMREAD_GRAYSCALE)

        _, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        training_images.append(image == 0)

    image = cv.imread(cur_dir + r'\data\hw3_leaf_testing_1.jpg',
                                         cv.IMREAD_GRAYSCALE)

    _, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    testing_image = image == 0

    square7 = np.ones((7, 7), dtype=np.uint8)
    square5 = np.ones((5, 5), dtype=np.uint8)

    background_detector = cv.dilate(testing_image.astype(np.float32), square7) - \
                          cv.dilate(testing_image.astype(np.float32), square5)

    nonzero_indices = np.nonzero(background_detector)

    nonzero_indices = np.array(list(zip(nonzero_indices[0], nonzero_indices[1])))

    # Find the convex hull around the shape and unpack the indices of the control points
    testing_hull_indices = cv.convexHull(nonzero_indices)
    testing_hull_list = list()
    testing_hull = np.zeros(background_detector.shape, dtype=np.bool)
    for i in range(testing_hull_indices.shape[0]):
        testing_hull_list.append((testing_hull_indices[i][0][0],
                                  testing_hull_indices[i][0][1]))
        testing_hull[testing_hull_list[i][0],
                     testing_hull_list[i][1]] = True

    # Find the centre of the polygon
    testing_centre = LineString(testing_hull_list).centroid
    testing_hull[int(testing_centre.x), int(testing_centre.y)] = True

    # Find the 5 point with max distance

    # Translate to the centre

    # Rotate

    # Multiply

    plt.imshow(testing_hull, 'gray')
    plt.show()



