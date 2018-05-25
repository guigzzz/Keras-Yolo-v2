import numpy as np


def logistic(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_out = np.exp(x - np.max(x, axis=-1)[..., None])
    return exp_out / np.sum(exp_out, axis=-1)[..., None]

def getClassInterestConf(yolov2_out, class_of_interest):
    class_logits = yolov2_out[..., 5:]
    class_preds = softmax(class_logits)

    logits_of_interest = class_preds[..., class_of_interest]

    out = np.concatenate((
        yolov2_out[..., :5], logits_of_interest[..., None]
    ), axis=-1)

    return out

TINY_YOLOV2_ANCHOR_PRIORS = np.array([
    [1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]
])
CELL_SIZE = 32

def getBoundingBoxesFromNetOutput(output, image_size, confidence_threshold):

    NUM_CELLS = image_size // CELL_SIZE
    print(image_size)
    
    cell_coords = np.array([
        (i, j)
        for i in range(NUM_CELLS)
        for j in range(NUM_CELLS)
    ])

    clf = output.reshape(NUM_CELLS ** 2, 5, 6)

    tx = clf[:, :, 0]
    ty = clf[:, :, 1]
    tw = clf[:, :, 2]
    th = clf[:, :, 3]
    to = clf[:, :, 4]
    class_confidences = clf[:, :, 5]

    pw, ph = TINY_YOLOV2_ANCHOR_PRIORS[:, 0], TINY_YOLOV2_ANCHOR_PRIORS[:, 1]
    cy, cx = cell_coords[:, 0], cell_coords[:, 1]

    bx = (logistic(tx) + cx[..., None]) * CELL_SIZE
    by = (logistic(ty) + cy[..., None]) * CELL_SIZE
    bw = (pw * np.exp(tw)) * CELL_SIZE
    bh = (ph * np.exp(th)) * CELL_SIZE
    object_confidences = logistic(to)

    final_confidence = class_confidences * object_confidences

    left = bx - bw / 2
    right = bx + bw / 2
    top = by - bh / 2
    bottom = by + bh / 2

    boxes = np.stack((
        left, bottom, right, top
    ), axis=-1).astype(np.int32)

    boxes = boxes[final_confidence > confidence_threshold].reshape(-1, 4)

    return boxes

