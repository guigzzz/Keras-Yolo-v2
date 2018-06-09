import cv2
from tiny_yolo_v2 import TinyYOLOv2
from yolo_v2 import YOLOv2
from scipy.misc import imresize
from time import time
import numpy as np

IM_SIZE = 416

# net = TinyYOLOv2(IM_SIZE)
# net.loadWeightsFromKeras('yolov2_tiny_keras_model')

net = YOLOv2(IM_SIZE)
net.loadWeightsFromKeras('yolov2_keras_model')

webcam = cv2.VideoCapture(0)
webcam.read()

n_frame = 1
buffer_size = 30
frame_time_buffer = np.zeros(buffer_size)

while True:
    try:
        t = time()
        
        ok, frame = webcam.read()
        frame = imresize(frame, (IM_SIZE, IM_SIZE)) / 255
        r = net.forward(frame)[0]

        if len(r) > 0:
            boxes, _ = r
            for left, top, right, bottom in boxes:
                cv2.rectangle(frame, (left, top), (right, bottom), color=(255, 0, 0), thickness=3)


        frame_time_buffer[n_frame % buffer_size] = time() - t
        cv2.putText(
            frame, 'fps: {:.2f}'.format(buffer_size / np.sum(frame_time_buffer)), (0, 15), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0)
            )

        cv2.imshow('frame', frame)
        cv2.waitKey(1)

        n_frame += 1

    except KeyboardInterrupt:
        print('exiting...')
        break
