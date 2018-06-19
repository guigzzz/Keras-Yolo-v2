import tensorflow as tf
import numpy as np

def bbToYoloFormat(bb):
    """
    converts (left, top, right, bottom) to
    (center_x, center_y, center_w, center_h)
    """
    x1, y1, x2, y2 = np.split(bb, 4, axis=1) 
    w = x2 - x1
    h = y2 - y1
    c_x = x1 + w / 2
    c_y = y1 + h / 2
    
    return np.concatenate([c_x, c_y, w, h], axis=-1)

def findBestPrior(bb, priors):
    """
    Given bounding boxes in yolo format and anchor priors
    compute the best anchor prior for each bounding box
    """
    w1, h1 = bb[:, 2], bb[:, 3]
    w2, h2 = priors[:, 0], priors[:, 1]
    
    # overlap, assumes top left corner of both at (0, 0)
    horizontal_overlap = np.minimum(w1[:, None], w2)
    vertical_overlap = np.minimum(h1[:, None], h2)
    
    intersection = horizontal_overlap * vertical_overlap
    union = (w1 * h1)[:, None] + (w2 * h2) - intersection
    iou = intersection / union
    return np.argmax(iou, axis=1)

def processGroundTruth(bb, labels, priors, network_output_shape):
    """
    Given bounding boxes in normal x1,y1,x2,y2 format, the relevant labels in one-hot form,
    the anchor priors and the yolo model's output shape
    build the y_true vector to be used in yolov2 loss calculation
    """
    bb = bbToYoloFormat(bb) / 32
    best_anchor_indices = findBestPrior(bb, priors)
    
    responsible_grid_coords = np.floor(bb).astype(np.uint32)[:, :2]
    
    values = np.concatenate((
        bb, np.ones((len(bb), 1)), labels
    ), axis=1)
    
    x, y = np.split(responsible_grid_coords, 2, axis=1)
    y = y.ravel()
    x = x.ravel()
    
    y_true = np.zeros(network_output_shape)    
    y_true[y, x, best_anchor_indices] = values
    
    return y_true

class YoloLossKeras:
    def __init__(self, priors, B=5, n_classes=20):
        self.priors = priors
        self.B = B
        self.n_classes = n_classes

    def loss(self, y_true, y_pred):

        n_cells = y_pred.get_shape().as_list()[1]
        y_pred = tf.reshape(y_pred, [-1, n_cells, n_cells, self.B, 4 + 1 + self.n_classes], name='y_pred')
        y_true = tf.reshape(y_true, tf.shape(y_pred), name='y_true')

        #### PROCESS PREDICTIONS ####
        # get x-y coords (for now they are with respect to cell)
        predicted_xy = tf.nn.sigmoid(y_pred[..., :2])

        # convert xy coords to be with respect to image
        cell_inds = tf.range(n_cells, dtype=tf.float32)
        predicted_xy = tf.stack((
            predicted_xy[..., 0] + tf.reshape(cell_inds, [1, -1, 1]), 
            predicted_xy[..., 1] + tf.reshape(cell_inds, [-1, 1, 1])
        ), axis=-1)

        # compute bb width and height
        predicted_wh = self.priors * tf.exp(y_pred[..., 2:4])

        # compute coords for 
        predicted_min = predicted_xy - predicted_wh / 2
        predicted_max = predicted_xy + predicted_wh / 2

        predicted_objectedness = tf.nn.sigmoid(y_pred[..., 4])
        predicted_logits = tf.nn.softmax(y_pred[..., 5:])


        #### PROCESS TRUE ####
        true_xy = y_true[..., :2]
        true_wh = y_true[..., 2:4]
        true_logits = y_true[..., 5:]

        true_min = true_xy - true_wh / 2
        true_max = true_xy + true_wh / 2

        #### compute iou between ground truth and predicted (used for objectedness) ####
        intersect_mins  = tf.maximum(predicted_min, true_min)
        intersect_maxes = tf.minimum(predicted_max, true_max)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = predicted_wh[..., 0] * predicted_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = intersect_areas / union_areas

        #### Compute loss terms ####
        responsibility_selector = y_true[..., 4]

        xy_diff = tf.square(true_xy - predicted_xy) * responsibility_selector[..., None]
        xy_loss = tf.reduce_sum(xy_diff, axis=[1, 2, 3, 4])

        wh_diff = tf.square(tf.sqrt(true_wh) - tf.sqrt(predicted_wh)) * responsibility_selector[..., None]
        wh_loss = tf.reduce_sum(wh_diff, axis=[1, 2, 3, 4])

        obj_diff = tf.square(iou_scores - predicted_objectedness) * responsibility_selector
        obj_loss = tf.reduce_sum(obj_diff, axis=[1, 2, 3])

        best_iou = tf.reduce_max(iou_scores, axis=-1)
        no_obj_diff = tf.square(0 - predicted_objectedness) * tf.to_float(best_iou < 0.6)[..., None] * (1 - responsibility_selector)
        no_obj_loss = tf.reduce_sum(no_obj_diff, axis=[1, 2, 3])

        clf_diff = tf.square(true_logits - predicted_logits) * responsibility_selector[..., None]
        clf_loss = tf.reduce_sum(clf_diff, axis=[1, 2, 3, 4])

        object_coord_scale = 5
        object_conf_scale = 1
        noobject_conf_scale = 1
        object_class_scale = 1

        loss = object_coord_scale * (xy_loss + wh_loss) + \
                object_conf_scale * obj_loss + noobject_conf_scale * no_obj_loss + \
                object_class_scale * clf_loss

        # loss = tf.Print(loss, [xy_loss, wh_loss, obj_loss, no_obj_loss, clf_loss])
        return loss 
        












    
















# Yolov2 loss
# S x S = number of cells
# B = number of anchors
# C = number of classes to predict
# logits = N x S x S x B * (4 + 1 + C)

"""
https://arxiv.org/pdf/1506.02640.pdf

Formally we define confidence as Pr(Object) ∗ IOU^truth_pred. 
If no object exists in that cell, the confidence scores should be
zero. Otherwise we want the confidence score to equal the
intersection over union (IOU) between the predicted box
and the ground truth.

Each bounding box consists of 5 predictions: x, y, w, h,
and confidence. The (x, y) coordinates represent the center
of the box relative to the bounds of the grid cell. The width
and height are predicted relative to the whole image. Finally
the confidence prediction represents the IOU between the
predicted box and any ground truth box. (specific to v1)


"""

# labels = N x S x S x B * (4 + 1 + C)
# labels contains ground truth x, y, w, h information 
# the label information should be normalised 
# - If a cell is responsible for an object,
# then its object confidence logit should be one
# - If a cell isn't responsible for an object,
# then its object confidence logit should be zero
# - If a cell contains an object, then the logit for the class of the object
# should be one

# S_loc = for all the reponsible cells, sum together the 
#       squared l2 distances between the label and prediction
# S_size = for all the responsible cells, sum together the 
#       squared l2 distances between the root of the label and the root of the prediction
# S_obj = for all the reponsible cells, sum together the
#       squared l2 distance between 1 and the object confidences
# S_noobj = for all the not responsible cells, sum together the
#       squared l2 distance between 0 and the object confidences
# S_class = for all cells containing an object, sum together the
#       squared l2 distance between the one hot labels and the network logits

# l_coord = 5
# l_noobj = 0.5
# loss = l_coord * S_loc + l_coord * S_size + S_obj + l_noobj * S_noobj + S_class








