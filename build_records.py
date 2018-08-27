# import the required libraries
from config import lisa_config as config
from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np
from utils import BoundingBox, bbox_iou

def parse_annotations():
    # initialize a data dictionary used to map each image filename to all
    # bounding boxes associated with the image, then load the contents of the
    # annotation file
    D = {}
    rows = open(config.ANNOT_PATH).read().strip().split("\n")

    # loop over the individual rows, skipping the header
    for row in rows[1:]:
        # break the row into components
        row = row.split(",")[0].split(";")
        (imagePath, label, startX, startY, endX, endY, _) = row
        (startX, startY) = (float(startX), float(startY))
        (endX, endY) = (float(endX), float(endY))

        # if we are not interested in the label, ignore it
        if label not in config.CLASSES:
            continue

        # build the path to the input image, then grab any other bounding-boxes + labels
        # associated with that image path
        p = os.path.sep.join([config.BASE_PATH, imagePath])
        b = D.get(p, [])

        # build a tuple consisting of the label and bounding box,
        # then update the list and store it in dictionary
        b.append((label, (startX, startY, endX, endY)))
        D[p] = b

    return D


def create_dataset(D):
    # Takes in the parsed annotation file as a dictionary
    # returns train/val/test dataset

    # create the training and testing splits from our data dictionary
    (trainKeys, testKeys) = train_test_split(list(D.keys()),
                                             test_size=config.TEST_SIZE, random_state=42)

    # initialize the data split files
    datasets = [
        ("train", trainKeys, config.TRAIN_DATA),
        ("test", testKeys, config.TEST_DATA)
    ]

    return datasets


def best_anchor_box(box):
    # find the anchor that best predicts this box
    best_anchor = -1
    max_iou = -1

    shifted_box = BoundingBox(0, 0, box[2], box[3])
    anchors = [BoundingBox(0, 0, config.ANCHORS[2*i], config.ANCHORS[2*(i+1)]) for i in range(len(config.ANCHORS)//2)]

    for i in range(len(anchors)):
        anchor = anchors[i]
        iou = bbox_iou(shifted_box, anchor)

        if max_iou < iou:
            best_anchor = i
            max_iou = iou


def build_dataset():
    print("inside build_dataset")
    D = parse_annotations()
    datasets = create_dataset(D)

    # loop over the datasets
    for (dType, keys, outputPath) in datasets:

        # Array to store all images
        x = np.zeros((len(keys), config.IMAGE_H, config.IMAGE_W, 3))
        # Array to store all grid+ bb coords + labels
        y = np.zeros((len(keys), config.GRID_S, config.GRID_S, config.BOXES, 4 + 1 + config.NUM_CLASSES))

        # initialize the writer and initialize the total number
        # of examples written to file
        print("[INFO] processing '{}'...".format(dType))
        #writer_x = HDF5DatasetWriter((len(keys), config.IMAGE_H, config.IMAGE_W, 3), outputPath)
        #writer_y = HDF5DatasetWriter((len(keys), config.GRID_S, config.GRID_S, config.BOXES, 4 + 1 + config.NUM_CLASSES),'lisa/hdf5/trainY.hdf5')
        total = 0

        # loop over all the keys in the current set
        for i, k in enumerate(keys):
            # load the input image from disk
            image = cv2.imread(k)
            (h, w, c) = image.shape

            # Check if the image is grayscale, convert it to BGR
            # It will still look gray, but we need 3 channels to feed to our model
            if c == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # Resize image (e.g. 416 x 416 for YOLO) and store it in x
            img = cv2.resize(image, (config.IMAGE_H, config.IMAGE_W), interpolation=cv2.INTER_AREA)
            x[i] = img


            # Calculate the ratios of resized to original dimensions
            w_ratio = config.IMAGE_W / w
            h_ratio = config.IMAGE_H / h

            # Calculate the grid cell width and height
            grid_cell_w = config.IMAGE_W / config.GRID_S
            grid_cell_h = config.IMAGE_H / config.GRID_S

            anchor_box = 0

            # loop over the bounding boxes + labels associated with the image
            for (label, (startX, startY, endX, endY)) in D[k]:
                # Transform bb coordinates according to resized image
                startX *= w_ratio
                startY *= h_ratio
                endX *= w_ratio
                endY *= h_ratio

                # Transform bb coord to center, w, h
                c_x = (startX + endX) * 0.5
                c_y = (startY + endY) * 0.5
                c_w = abs(endX - startX)
                c_h = abs(endY - endX)

                # Scale the coordinates to grid cell units
                c_x_grid = c_x / grid_cell_w
                c_y_grid = c_y / grid_cell_h
                c_w_grid = c_w / grid_cell_w
                c_h_grid = c_h / grid_cell_h
                box = [c_x_grid, c_y_grid, c_w_grid, c_h_grid]

                # Determine the grid cell to place the center coordinates in
                grid_x = int(c_x_grid)
                grid_y = int(c_y_grid)

                # increment the total number of examples(bb)
                total += 1

                # Figure out which anchor box to use

                # Figure the index of the label
                label_index = config.CLASSES[label]

                # Create the Y array
                y[i, grid_x, grid_y, anchor_box, :4] = box
                y[i, grid_x, grid_y, anchor_box, 4] = 1
                y[i, grid_x, grid_y, anchor_box, 4+label_index] = 1

                # Find the most suitable anchor box
                anchor_box = best_anchor_box(box)
                #anchor_box += 1

                cv2.rectangle(img, (int(startX), int(startY)), (int(endX), int(endY)), (0, 255, 0), 2)
                cv2.imshow('Image', img)
                cv2.waitKey(0)

            # add the image and label to the HDF5
            #writer.add([image], [label])

        # close the writer and print the diagnostics information to the user
        #writer.close()
        #print("[INFO] {} examples saved for '{}'".format(total, dType))


if __name__ == '__main__':
    build_dataset()
