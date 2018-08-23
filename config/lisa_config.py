# import the packages
import os

# initialize the base path for the LISA dataset
BASE_PATH = "lisa"

# build the path to the annotations file
ANNOT_PATH = os.path.sep.join([BASE_PATH, "allAnnotations.csv"])

# build the path to the output training and testing record files,
# along with the class labels file

TRAIN_DATA = os.path.sep.join([BASE_PATH, "hdf5/train.hdf5"])
VAL_DATA = os.path.sep.join([BASE_PATH, "hdf5/val.hdf5"])
TEST_DATA = os.path.sep.join([BASE_PATH, "hdf5/test.hdf5"])

# initialize the test-train split
TEST_SIZE = 0.25

# initialize the class labels dictionary
CLASSES = {"pedestrianCrossing": 1, "signalAhead": 2, "stop": 3}
