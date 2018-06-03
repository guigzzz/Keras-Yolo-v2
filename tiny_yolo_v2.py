from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D
from keras import backend as K

from darknet_weight_loader import load_weights

import tensorflow as tf
from classifcation_utils import non_max_suppression
from yolov2_tools import getClassInterestConf, getBoundingBoxesFromNetOutput

class TinyYOLOv2:
    def __init__(self, image_size):
        K.set_learning_phase(0)
        self.sess = K.get_session()
        self.image_size = image_size

        self.m = self.buildModel()
        self.has_weights = False

    def loadWeightsFromDarknet(self, file_path):
        self.sess.run(tf.global_variables_initializer())
        load_weights(self.m, file_path)

        self.has_weights = True

    def loadWeightsFromKeras(self, file_path):
        self.sess.run(tf.global_variables_initializer())
        self.m.load_weights(file_path)

        self.has_weights = True

    def buildModel(self):
        m = Sequential()
        m.add(Conv2D(16, (3, 3), padding='same', input_shape=(self.image_size, self.image_size, 3)))
        m.add(BatchNormalization())
        m.add(LeakyReLU(alpha=0.1))
        m.add(MaxPooling2D(2, padding='valid'))
        
        for i in range(0, 4):
            m.add(Conv2D(32 * 2 ** i, (3, 3), padding='same'))
            m.add(BatchNormalization())
            m.add(LeakyReLU(alpha=0.1))
            m.add(MaxPooling2D(2, padding='valid'))

        m.add(Conv2D(512, (3, 3), padding='same'))
        m.add(BatchNormalization())
        m.add(LeakyReLU(alpha=0.1))
        m.add(MaxPooling2D(2, 1, padding='same'))
        
        for _ in range(2):
            m.add(Conv2D(1024, (3, 3), padding='same'))
            m.add(BatchNormalization())
            m.add(LeakyReLU(alpha=0.1))
        
        m.add(Conv2D(125, (1, 1), padding='same', activation='linear'))
        
        return m

    def forward(self, images):
        if not self.has_weights:
            raise ValueError("Network needs to be initialised before being executed")

        if len(images.shape) == 3:
            # single image
            images = images[None]
        
        output = self.sess.run(self.m.output, {self.m.input: images}).reshape(-1, self.image_size // 32, self.image_size // 32, 5, 25)

        allboxes = []
        for o in output:
            out = getClassInterestConf(o, 14) # people class
            boxes = getBoundingBoxesFromNetOutput(out, confidence_threshold=0.3)
            allboxes.append(non_max_suppression(boxes, 0.3))

        return allboxes




    