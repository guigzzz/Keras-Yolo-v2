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

CELL_SIZE = 32

def getBoundingBoxesFromNetOutput(clf, anchors, confidence_threshold):
    pw, ph = anchors[:, 0], anchors[:, 1]
    cell_inds = np.arange(clf.shape[1])

    tx = clf[..., 0]
    ty = clf[..., 1]
    tw = clf[..., 2]
    th = clf[..., 3]
    to = clf[..., 4]
    class_confidences = clf[..., 5]

    bx = logistic(tx) + cell_inds[None, :, None]
    by = logistic(ty) + cell_inds[:, None, None]
    bw = pw * np.exp(tw) / 2
    bh = ph * np.exp(th) / 2
    object_confidences = logistic(to)

    left = bx - bw
    right = bx + bw
    top = by - bh
    bottom = by + bh

    boxes = np.stack((
        left, top, right, bottom
    ), axis=-1) * CELL_SIZE
    
    final_confidence = class_confidences * object_confidences
    boxes = boxes[final_confidence > confidence_threshold].reshape(-1, 4).astype(np.int32)

    return boxes

