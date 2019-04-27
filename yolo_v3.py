from keras.layers import Conv2D, Input, Concatenate, Reshape, add
from keras import backend as K
from keras import Model
import numpy as np

from postprocessing import yoloPostProcess
from yolo_layer_utils import conv_batch_lrelu, NetworkInNetwork, residual, Upsample

YOLOV3_ANCHOR_PRIORS = np.array([
    10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
]).reshape(3, 3, 2)[::-1]

class YOLOv3:
    def __init__(self, image_size, B, n_classes, is_learning_phase=False):
        K.set_learning_phase(int(is_learning_phase))
        K.reset_uids()

        self.image_size = image_size
        self.n_cells = self.image_size // 32
        self.B = B
        self.n_classes = n_classes

        self.outs_per_cell = self.B * (4 + 1 + self.n_classes)

        self.m = self.buildModel()

    def loadWeightsFromKeras(self, file_path):
        self.m.load_weights(file_path)

    def buildModel(self):
        model_in = Input((self.image_size, self.image_size, 3))

        model = conv_batch_lrelu(model_in, 32, 3)
        model = conv_batch_lrelu(model, 64, 3, strides=2)

        model = residual(model, 1, [(32, 1), (64, 3)])
        model = conv_batch_lrelu(model, 128, 3, strides=2)

        model = residual(model, 2, [(64, 1), (128, 3)])
        model = conv_batch_lrelu(model, 256, 3, strides=2)

        model = shortcut2 = residual(model, 8, [(128, 1), (256, 3)])
        model = conv_batch_lrelu(model, 512, 3, strides=2)

        model = shortcut1 = residual(model, 8, [(256, 1), (512, 3)])
        model = conv_batch_lrelu(model, 1024, 3, strides=2)

        model = residual(model, 4, [(512, 1), (1024, 3)])

        #############

        model = NetworkInNetwork(model, [(512, 1), (1024, 3), (512, 1), (1024, 3), (512, 1)])
        out1 = conv_batch_lrelu(model, 1024, 3)
        out1 = Conv2D(self.outs_per_cell // 3, 1)(out1)

        model = conv_batch_lrelu(model, 256, 1)
        model = Upsample(2)(model)
        model = Concatenate()([model, shortcut1])

        model = NetworkInNetwork(model, [(256, 1), (512, 3), (256, 1), (512, 3), (256, 1)])
        out2 = conv_batch_lrelu(model, 512, 3)
        out2 = Conv2D(self.outs_per_cell // 3, 1)(out2)

        model = conv_batch_lrelu(model, 128, 1)
        model = Upsample(2)(model)
        model = Concatenate()([model, shortcut2])

        model = NetworkInNetwork(model, [(128, 1), (256, 3), (128, 1), (256, 3), (128, 1)])
        out3 = conv_batch_lrelu(model, 256, 3)
        out3 = Conv2D(self.outs_per_cell // 3, 1)(out3)

        return Model(inputs=model_in, outputs=[out1, out2, out3])

    def forward(self, images):
        if len(images.shape) == 3:
            # single image
            images = images[None]

        output = self.m.predict(images)

        all_out = []

        for outs in zip(*output):

            boxes = []
            labels = []

            for i, (o, p) in enumerate(zip(outs, YOLOV3_ANCHOR_PRIORS)):
                o = o.reshape(1, *o.shape[:2], self.B // 3, self.n_classes + 5)
                cell_size = 32 / 2**i
                b, l = yoloPostProcess(o, p / cell_size, classthresh=0.7,
                                cell_size=cell_size, maxsuppressionthresh=0.4)[0]
                boxes.extend(b + int(cell_size))
                labels.extend(l)
                
            all_out.append((
                np.array(boxes), 
                np.array(labels)
            ))

        return all_out
    