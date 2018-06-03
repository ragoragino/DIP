import os, sys
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import itertools

cur_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(cur_dir)

"""
Find all the neighbours of a point in a 3x3x3 enclosing block
"""
def findNeighbours(input_index):
    indices = []
    differences = [[0, -1, -1], [0, -1, 0], [0, -1, 1],
            [0, 0, -1], [0, 0, 0], [0, 0, 1],
            [0, 1, -1], [0, 1, 0], [0, 1, 1]]

    for shift in [-1, 0, 1]:
        for i, j, k in differences:
            if shift == i == j == k == 0:
                continue
            else:
                indices.append([i + shift, j, k])

    for index in indices:
        yield index


"""
Match a block value with a value of of the most intensive neighbour block
"""
def findHighestProbability(original, index):
    group = np.random.choice([0, 1])

    max_prob = 0.0

    for i, j, k in findNeighbours(index):
        n_not_object = original[i, j, k, 0]
        n_object = original[i, j, k, 1]

        if n_not_object == n_object == 0:
            continue

        prob_not_object = n_not_object / (n_object + n_not_object)
        prob_object = n_object / (n_object + n_not_object)
        if prob_not_object > max_prob:
            max_prob = prob_not_object
            group = 0
        elif prob_object > max_prob:
            max_prob = prob_object
            group = 1

    return group


"""
Set a probability for each block from a 16x16x16 grid
"""
def getProbabilities(original):
    probabilities = np.zeros((16, 16, 16), dtype=np.int8)

    unclassified = []

    # Set the probability of object for each individual block
    for i, j, k in itertools.product(range(16), range(16), range(16)):
                if original[i, j, k, 0] == original[i, j, k, 1] == 0:
                    unclassified.append([i, j, k])
                else:
                    if original[i, j, k, 0] >= original[i, j, k, 1]:
                        probabilities[i, j, k] = 0
                    else:
                        probabilities[i, j, k] = 1

    # Propagate values to blocks that could not be classified due to
    # scarcity of information
    for i, j, k in unclassified:
        probabilities[i, j, k] = findHighestProbability(original, [i, j, k])

    return probabilities


if __name__ == '__main__':
    training_images = []
    mask_images = []

    # Load training images
    for i in range(1, 6):
        training_images.append(cv.imread(cur_dir + r'/data/hw2_cone_training_' + str(i) + '.jpg',
                              cv.IMREAD_COLOR))

        mask_images.append(cv.imread(cur_dir + r'/data/hw2_cone_training_map_' + str(i) + '.png',
                                  cv.IMREAD_GRAYSCALE))

    # Load test images
    test_images = []
    for i in range(1, 3):
        test_images.append(cv.imread(cur_dir + r'/data/hw2_cone_testing_' + str(i) + '.jpg',
                              cv.IMREAD_COLOR))

    # Find the number of pixel values from the training set that can be assigned to object
    # we are investigating and that cannot be assigned to this object (1, 0 respectively)
    mask = np.zeros((16, 16, 16, 2), dtype=np.int32)
    indices = np.zeros(3, dtype=np.int8)
    for i in range(5):
        height, width = mask_images[i].shape
        for h, w in itertools.product(range(height), range(width)):
           indices[:] = training_images[i][h, w, :] // 16
           r, g, b = indices.tolist()
           if mask_images[i][h, w] == 0:
                mask[r, g, b, 0] += 1
           else:
                mask[r, g, b, 1] += 1

    # Find the probabilities for each block of pixels from a 16x16x16 grid
    prob = getProbabilities(mask)

    # Assign the object/non-object based on the probabilities to the testing images
    for i in range(2):
        height, width, _ = test_images[i].shape
        test = np.zeros((height, width), dtype=np.int8)
        for h, w in itertools.product(range(height), range(width)):
           indices[:] = test_images[i][h, w, :] // 16
           r, g, b = indices.tolist()
           test[h, w] = prob[r, g, b]

        plt.imshow(test, 'gray')
        plt.show()




