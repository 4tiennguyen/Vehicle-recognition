from resnet_152 import resnet152_model
import cv2 as cv
import keras.backend as K
import numpy as np
import scipy.io
import sys
import json
import os
import random

def load_model():
    model_weights_path = 'models/model.23-0.89.hdf5'
    img_width, img_height = 224, 224
    num_channels = 3
    num_classes = 196
    model = resnet152_model(img_height, img_width, num_channels, num_classes)
    model.load_weights(model_weights_path, by_name=True)
    return model

if __name__ == '__main__':
    img_width, img_height = 224, 224
    model = load_model()
    model.load_weights('models/model.23-0.89.hdf5')

    cars_meta = scipy.io.loadmat('devkit/cars_meta')
    class_names = cars_meta['class_names']  # shape=(1, 196)
    class_names = np.transpose(class_names)

    if len(sys.argv) > 1:
        arg = sys.argv[1:][0]
        print(arg)
        image = cv.imread(arg)
        image = cv.resize(image, (img_width, img_height), cv.INTER_CUBIC)
        rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgb_img = np.expand_dims(rgb_img, 0)
        preds = model.predict(rgb_img)
        prob = np.max(preds)
        class_id = np.argmax(preds)
        text = ('Predict: {}, prob: {}'.format(class_names[class_id][0][0], prob))
        print(text)


    K.clear_session()