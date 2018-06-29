import cv2
import numpy as np

voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 
                'bus', 'train', 'truck', 'boat', 'traffic light', 
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 
                'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
                'sports ball', 'kite', 'baseball bat', 'baseball glove', 
                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 
                'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 
                'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 
                'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
                'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def annotate_image(image, boxes, labels, class_names):
    image = (image.copy() * 255).astype(np.uint8)

    for (left, top, right, bottom), label in zip(boxes, labels):
        cv2.rectangle(image, (left, top), 
                (right, bottom), color=(255, 0, 0), thickness=2)
        cv2.putText(
            image, class_names[label], (left, top-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0)
        )

    return image


