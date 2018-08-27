

class BoundingBox:
    # Class to define a Bounding Box with diagonal opposite coordinates
    # (x_min, y_min) and (x_max, y_max)
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max


def _interval_overlap(box1_min, box1_max, box2_min, box2_max):

    # Check to see if box 2 starts before box 1
    if box2_min < box1_min:
        # If yes, then see if box 2 ends before box 1
        if box2_max < box1_min:
            # This means no overlap
            return 0
        else:
            # Otherwise, check for the min between box1 and box2 end points
            # and take the difference with box1 start point
            return min(box1_max, box2_max) - box1_min
    else:
        if box1_max < box2_min:
            return 0
        else:
            return min(box1_max, box2_max) - box2_min


def bbox_iou(box1, box2):

    # Calculate the intersection of the two boxes
    intersect_w = _interval_overlap(box1.x_min, box1.x_max, box2.x_min, box2.x_max)
    intersect_h = _interval_overlap(box1.y_min, box1.y_max, box2.y_min, box2.y_max)
    intersect = intersect_w * intersect_h

    # Calculate the union of the two boxes
    w1, h1 = box1.x_max - box1.x_min, box1.y_max - box1.y_min
    w2, h2 = box2.x_max - box2.x_min, box2.y_max - box2.y_min
    union = (w1 * h1) + (w2 * h2) - intersect

    return intersect/union


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)

    if np.min(x) < t:
        x = x / np.min(x) * t

    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)