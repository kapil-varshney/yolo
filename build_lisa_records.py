#import
from config import lisa_config as config
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
import os
import cv2


def main(_):
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
        # associated with the image path, labels, and bounding box lists, respectively
        p = os.path.sep.join([config.BASE_PATH, imagePath])
        b = D.get(p, [])

        # build a tuple consisting of the label and bounding box,
        # then update the list and store it in dictionary
        b.append((label, (startX, startY, endX, endY)))
        D[p] = b

    # create the training and testing splits from our data dictionary
    (trainKeys, testKeys) = train_test_split(list(D.keys()),
        test_size=config.TEST_SIZE, random_state=42)

    # initialize the data split files
    datasets = [
        ("train", trainKeys, config.TRAIN_RECORD),
        ("test", testKeys, config.TEST_RECORD)
    ]


    # loop over the datasets
    for(dType, keys, outputPath) in datasets:

        x = np.zeros()  # Store all images here
        y = np.zeros()  # Store all grid+ bb coords + labels info here

        # initialize the writer and initialize the total number
        # of examples written to file
        print("[INFO] processing '{}'...".format(dType))
        writer = HDF5DatasetWriter((len(keys), 416, 416, 3), outputPath)
        total = 0

        # loop over all the keys in the current set
        for k in keys:
            # load the input image from disk
            image = cv2.imread(k)
            (w, h) = image.size[:2]

            # Resize image to 416 x 416 (for YOLO)
            # Transform bb coordinates according to image resize
            # Transform bb coord to center,w,h
            # Scale the bb coord

            # loop over the bounding boxes + labels associated with the image
            for (label, (startX, startY, endX, endY)) in D[k]:
                #TensorFlow assumes all  bounding boxes are in the range[0,1]
                # so we need to scale them
                xMin = startX / w
                xMax = endX / w
                yMin = startY / h
                yMax = endY / h

                #update the bounding boxes + labels lists
                tfAnnot.xMins.append(xMin)
                tfAnnot.xMaxs.append(xMax)
                tfAnnot.yMins.append(yMin)
                tfAnnot.yMins.append(yMax)
                tfAnnot.textLabels.append(label.encode("utf8"))
                tfAnnot.classes.append(config.CLASSES[label])
                tfAnnot.difficult.append(0)

                # increment the total number of examples(bb)
                total += 1
            """
            #loop over the bounding boxes + labels associated with the image
            for (label, (startX, startY, endX, endY)) in D[k]:
                #TensorFlow assumes all  bounding boxes are in the range[0,1]
                # so we need to scale them
                xMin = startX / w
                xMax = endX / w
                yMin = startY / h
                yMax = endY / h

                #load the input image from disk and denormalize the bounding box
                # coordinates
                image = cv2.imread(k)
                startX = int(xMin*w)
                startY = int(yMin*h)
                endX = int(xMax*w)
                endY = int(yMax*h)

                #draw the bounding box on the image
                cv2.rectangle(image, (startX, startY), (endX, endY), (0,255,0), 2)
                cv2.imshow("Image", image)
                cv2.waitKey(0)
            """

            # add the image and label to the HDF5
            writer.add([image], [label])

        #close the writer and print the diagnostics information to the user
        writer.close()
        print("[INFO] {} examples saved for '{}'".format(total, dType))

# check to see if the main thread should be started
if __name__ == '__main__':
    tf.app.run()
