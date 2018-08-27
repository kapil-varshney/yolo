# import required libraries
import random
import numpy as np
from build_records import parse_annotations
from config import lisa_config as config
import cv2


def IOU(ann, centroids):
    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid

        if c_w >= w and c_h >= h:
            similarity = w*h/(c_w*c_h)
        elif c_w >= w and c_h <= h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else:
            # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape

    return np.array(similarities)


def avg_IOU(anns, centroids):
    n, d = anns.shape
    sum = 0.

    for i in range(anns.shape[0]):
        sum += max(IOU(anns[i], centroids))

    return sum/n


def print_anchors(centroids):
    anchors = centroids.copy()

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    r = "anchors: ["
    for i in sorted_indices[:-1]:
        r += '%0.5f,%0.5f, ' % (anchors[i,0], anchors[i,1])

    # there should not be comma after last anchor, that's why
    r += '%0.5f,%0.5f' % (anchors[sorted_indices[-1:],0], anchors[sorted_indices[-1:],1])
    r += "]"

    print(r)


def kmeans(ann_dims, anchor_num):
    ann_num = ann_dims.shape[0]
    prev_assignments = np.ones(ann_num)*(-1)
    iteration = 0
    old_distances = np.zeros((ann_num, anchor_num))

    indices = [random.randrange(ann_dims.shape[0]) for i in range(anchor_num)]
    centroids = ann_dims[indices]
    anchor_dim = ann_dims.shape[1]

    while True:
        distances = []
        iteration += 1
        for i in range(ann_num):
            d = 1 - IOU(ann_dims[i], centroids)
            distances.append(d)
        distances = np.array(distances) # distances.shape = (ann_num, anchor_num)

        print("iteration {}: dists = {}".format(iteration, np.sum(np.abs(old_distances-distances))))

        # assign samples to centroids
        assignments = np.argmin(distances,axis=1)

        if (assignments == prev_assignments).all() :
            return centroids

        # calculate new centroids
        centroid_sums = np.zeros((anchor_num, anchor_dim), np.float)
        for i in range(ann_num):
            centroid_sums[assignments[i]] += ann_dims[i]
        for j in range(anchor_num):
            centroids[j] = centroid_sums[j]/(np.sum(assignments == j) + 1e-6)

        prev_assignments = assignments.copy()
        old_distances = distances.copy()


def gen_anchors():
    # Get the Dictionary with image paths as keys
    # and labels and bb coordinates as values
    D = parse_annotations()
    num_anchors = config.BOXES
    bounding_boxes = []

    # Iterate over the list of image path
    for i, k in enumerate(D.keys()):

        # Read the image and extract its shape
        img = cv2.imread(k)
        (h, w, c) = img.shape

        # Calculate the ratios of resized to original dimensions
        w_ratio = config.IMAGE_W / w
        h_ratio = config.IMAGE_H / h

        # Calculate the grid cell width and height
        grid_cell_w = config.IMAGE_W / config.GRID_S
        grid_cell_h = config.IMAGE_H / config.GRID_S

        # loop over the bounding boxes + labels associated with the image
        for (label, (startX, startY, endX, endY)) in D[k]:
            # Transform bb coordinates according to resized image
            startX *= w_ratio
            startY *= h_ratio
            endX *= w_ratio
            endY *= h_ratio

            # Transform bb coord to center, w, h
            c_w = abs(endX - startX)
            c_h = abs(endY - endX)

            # Scale the coordinates to grid cell units
            c_w_grid = c_w / grid_cell_w
            c_h_grid = c_h / grid_cell_h

            bounding_boxes.append(tuple(map(float, (c_w_grid, c_h_grid))))

    bounding_boxes = np.array(bounding_boxes)
    anchors = kmeans(bounding_boxes, num_anchors)

    print('\naverage IOU for', num_anchors, 'anchors:',
          '%0.2f' % avg_IOU(bounding_boxes, anchors))
    print_anchors(anchors)


if __name__ == '__main__':
    gen_anchors()
