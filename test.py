import os
import cv2
import keras
import numpy as np 

def load_model():
    model_weights_path = 'models/model.96-0.89.hdf5'
    img_width, img_height = 224, 224
    num_channels = 3
    num_classes = 196
    model = resnet152_model(img_height, img_width, num_channels, num_classes)
    model.load_weights(model_weights_path, by_name=True)
    return model

def predict(imRGB):
    pred = model.predict(imRGB)
    return np.argmax(pred)

def load_imgRGB(pathIMG):
    imBGR = cv2.imread(pathIMG)
    imRGB = cv2.cvtColor(imBGR, cv2.COLOR_BGR2RGB)
    return np.expand_dims(imRGB, 0)

if __name__ == '__main__':
    model = load_model()
    num_samples = 8041
    results = open('results.txt', 'a') #replace w output filename

    for i in range(num_samples):
        imRGB = load_imgRGB(os.path.join('data/test', '%05d.jpg' % (i + 1)))  #replace w input filename
        res = predict(imRGB)
        results.write(str(res + 1)+'\n')

    results.close()
    keras.backend.clear_session()
