from keras.layers import Conv2D, MaxPooling2D, Input, Concatenate
from keras import backend as K
from keras import Model
import numpy as np

from darknet_weight_loader import load_weights
from classifcation_utils import non_max_suppression
from yolov2_tools import getClassInterestConf, getBoundingBoxesFromNetOutput
from yolo_utils import conv_batch_lrelu, NetworkInNetwork, reorg

YOLOV2_ANCHOR_PRIORS = np.array([
    1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071
]).reshape(5, 2)

class YOLOv2:
    def __init__(self, image_size):
        K.set_learning_phase(0)
        K.reset_uids()

        self.image_size = image_size

        self.m = self.buildModel()
        self.has_weights = False

    def loadWeightsFromDarknet(self, file_path):
        load_weights(self.m, file_path)
        self.has_weights = True

    def loadWeightsFromKeras(self, file_path):
        self.m.load_weights(file_path)
        self.has_weights = True

    def buildModel(self):
        model_in = Input((self.image_size, self.image_size, 3))

        model = conv_batch_lrelu(model_in, 32, 3)
        model = MaxPooling2D(2, padding='valid')(model)

        model = conv_batch_lrelu(model, 64, 3)
        model = MaxPooling2D(2, padding='valid')(model)

        #### NIN 1 ####
        model = NetworkInNetwork(model, [(128, 3), (64, 1), (128, 3)])
        model = MaxPooling2D(2, padding='valid')(model)

        #### NIN 2 ####
        model = NetworkInNetwork(model, [(256, 3), (128, 1), (256, 3)])
        model = MaxPooling2D(2, padding='valid')(model)

        #### NIN 3 ####
        model = NetworkInNetwork(model, [(512, 3), (256, 1), (512, 3), (256, 1), (512, 3)])
        skip = model
        model = MaxPooling2D(2, padding='valid')(model)

        #### NIN 4 ####
        model = NetworkInNetwork(model, [(1024, 3), (512, 1), (1024, 3), (512, 1), (1024, 3)])

        #### Detection Layers ####
        model = conv_batch_lrelu(model, 1024, 3)
        model = conv_batch_lrelu(model, 1024, 3)

        convskip = conv_batch_lrelu(skip, 64, 1)
        model = Concatenate()([reorg(convskip, 2), model])
        
        model = conv_batch_lrelu(model, 1024, 3)
        model_out = Conv2D(125, (1, 1), padding='same', activation='linear')(model)

        m = Model(inputs=model_in, outputs=model_out)
        
        return m

    def forward(self, images):
        if not self.has_weights:
            raise ValueError("Network needs to be initialised before being executed")

        if len(images.shape) == 3:
            # single image
            images = images[None]

        output = self.m.predict(images).reshape(
            -1, self.image_size // 32, self.image_size // 32, 5, 25)

        allboxes = []
        for o in output:
            out = getClassInterestConf(o, 14) # people class
            boxes = getBoundingBoxesFromNetOutput(out, YOLOV2_ANCHOR_PRIORS, confidence_threshold=0.3)
            allboxes.append(non_max_suppression(boxes, 0.3))

        return allboxes




    