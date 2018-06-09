# Keras-Yolo-v2
A very basic re-implementation of Yolo v2 in Keras.
Both normal and tiny backbone models can be used. 

### Links to necessary weight files:
- [Tiny Yolo v2](https://pjreddie.com/media/files/yolov2-tiny-voc.weights)
- [Yolo v2](https://pjreddie.com/media/files/yolov2-voc.weights)

# Usage

### Loading the model:
```py
from tiny_yolo_v2 import TinyYOLOv2
# or from yolo_v2 import YOLOv2

IM_SIZE = 13*32
net = TinyYOLOv2(IM_SIZE)
net.loadWeightsFromDarknet(tiny_yolo_darknet_weight_file)
```
## Inference:
Output bounding boxes are in [left, top, right, bottom] format. The origin is at the top-left corner of the image. 

The forward function expects an array of dimensions `[None, IM_SIZE, IM_SIZE, 3]` or `[IM_SIZE, IM_SIZE, 3]`. 

- Regardless of the input, `results` is a list of tuples of length the number of images. 
- Each tuple contains a list of bounding boxes and a list of integer labels, where each label pairs up with a bounding box
```py
resized_image = resize(image, (IM_SIZE, IM_SIZE))
results = net.forward(resized_image)
```

## Once loaded from darknet files, weights can be saved to keras format:
```py
net.m.save(desired_keras_save_path)
```

## Then the models can be loaded without interacting with the darknet files:
(This is also quite a bit faster than the other method)
```py
from tiny_yolo_v2 import TinyYOLOv2
# or from yolo_v2 import YOLOv2

IM_SIZE = 13*32
net = TinyYOLOv2(IM_SIZE)
net.loadWeightsFromKeras(tiny_yolo_keras_weight_file)
```

# Example:

#### Ground truth:
![alt text](images/example_ground_truth.JPG)

#### Detections:
![alt text](images/example_detection.JPG)

(As we can see, the INRIA dataset annotations are pretty bad. In this case, the model detects an unannotated object)