import numpy as np

def overlap(a, b):
    (ya1, xa1, ya2, xa2) = a
    (yb1, xb1, yb2, xb2) = b
    horizontal_overlap = np.maximum(0, np.minimum(xa2, xb2) - np.maximum(xa1, xb1))
    vertical_overlap = np.maximum(0, np.minimum(ya2, yb2) - np.maximum(ya1, yb1))
    return horizontal_overlap * vertical_overlap

def area(a):
    (y1, x1, y2, x2) = a
    return (y2 - y1) * (x2 - x1)

def IOU(a, b):
    intersection = overlap(a, b)
    return intersection / (area(a) + area(b) - intersection)  