import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Lambda, LeakyReLU, Conv2D, Input, Layer, Dense, Reshape, GaussianNoise, LayerNormalization, Multiply, Add
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.backend import int_shape, ones
import numpy as np
import os
import uuid
import shutil
import subprocess
import json
import cv2
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Input, BatchNormalization, Reshape, UpSampling2D, PReLU, ReLU, LeakyReLU, Lambda, GlobalAveragePooling2D, LayerNormalization, Multiply, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import to_categorical, plot_model
import tensorflow as tf


from tqdm.notebook import tqdm
from multiprocessing import Pool
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from timeout_decorator import timeout, TimeoutError


def SubpixelConv2D(input_shape, scale=2):

    # Copyright (c) 2017 Jerry Liu
    # Released under the MIT license
    # https://github.com/twairball/keras-subpixel-conv/blob/master/LICENSE

    def subpixel(x):
        return tf.nn.depth_to_space(x, scale)

    return Lambda(subpixel)


def ResBlock(layer_input, out_channel):
    d = Conv2D(out_channel, 3, strides=1, padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    # d = Conv2D(out_channel, 3, strides=1, padding='same')(d)
    # d = Add()([d, layer_input])
    # d = LeakyReLU(alpha=0.2)(d)
    return d


def visualize(g, d=None, epoch=None, row=4, col=5, save=None):
    noise_length = g.layers[0].input.shape[1]
    plt.figure(figsize=(col * 3, row * 3))
    noise = np.random.uniform(-1, 1, size=(row * col, noise_length))
    pred = g.predict(noise, verbose=0)
    if d:
        losses = d.predict(pred)
        if type(losses) == list:
            losses = np.asarray(losses)
            losses = np.mean(losses, axis=0)
    if epoch:
        plt.suptitle("epoch={0}".format(epoch), fontsize=15)
    for i in range(row * col):
        plt.subplot(row, col, i + 1)
        plt.imshow((pred[i] * 127.5 + 127.5).astype(np.uint8))
        if d:
            plt.title(np.mean(losses[i]))
        plt.axis('off')
    if save:
        plt.savefig(save)
    plt.show()
    return noise

def Combined_model(g, d):
    noise_length = g.layers[0].input.shape[1]
    inputs = Input((noise_length,))
    x = g(inputs)
    outputs = d(x)
    return Model(inputs=inputs, outputs=outputs)


def Combined_model2(g, d):
    noise_length = g.layers[0].input.shape[1]
    inputs = Input((noise_length,))
    inputs2 = Input((1,))
    x = g([inputs, inputs2])
    outputs = d(x)
    return Model(inputs=[inputs,inputs], outputs=outputs)

def check(img):
    likelihood = 0
    uid = str(uuid.uuid4())
    tmpFolder = "tmp-" + uid
    os.makedirs(tmpFolder)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(tmpFolder, "tmp.jpg"), img)
    cmd = "ruby /animeface-2009/animeface-ruby/proc_folder.rb {0} {0}/out.json".format(
        tmpFolder)
    subprocess.run(cmd, shell=True)
    with open("{0}/out.json".format(tmpFolder), "r") as fp:
        for line in fp:
            data = json.loads(line)
            likelihood = data["likelihood"]
    shutil.rmtree(tmpFolder)
    return likelihood


def create(generator, likelihood=0.999, size=None):
    noise_length = generator.layers[0].input.shape[1]
    while True:
        noise = np.random.uniform(-1, 1, size=(1, noise_length))
        img = generator.predict(noise)[0]
        img = img * 127.5 + 127.5
        img[img > 255] = 255
        img[img < 0] = 0
        img = img.astype(np.uint8)
        if check(img) < likelihood:
            continue
        if size:
            img = cv2.resize(img, size)
        img = cv2.bilateralFilter(img, 5, 35, 35)
        return img
  
"""
#https://github.com/penny4860/keras-adain-style-transfer
class AdaIN(Layer):
    def __init__(self, alpha=1.0, **kwargs):
        self.alpha = alpha
        super(AdaIN, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        assert input_shape[0] == input_shape[1]
        return input_shape[0]
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'alpha': self.alpha
        })
        return config

    def call(self, x):
        assert isinstance(x, list)
        # Todo : args
        content_features, style_features = x[0], x[1]
        style_mean, style_variance = tf.nn.moments(style_features, [1,2], keepdims=True)
        content_mean, content_variance = tf.nn.moments(content_features, [1,2], keepdims=True)
        epsilon = 1e-5
        normalized_content_features = tf.nn.batch_normalization(content_features, content_mean,
                                                                content_variance, style_mean, 
                                                                tf.sqrt(style_variance), epsilon)
        normalized_content_features = self.alpha * normalized_content_features + (1 - self.alpha) * content_features
        return normalized_content_features

"""
def parseshape(c):
    shape=int_shape(c)[1::]
    return shape


def AdaINBlock(x,z):
    x = LayerNormalization(axis=[1,2])(x)
    shape = parseshape(x)
    assert len(shape)==3
    ch = shape[2]
    ysi = Dense(ch)(z)
    ysi = Reshape((1,1,ch))(ysi)
    ysi = Lambda(K.tile, arguments={'n':(1,shape[0],shape[1],1)})(ysi)
    ybi = Dense(ch)(z)
    ybi = Reshape((1,1,ch))(ybi)
    ybi = Lambda(K.tile, arguments={'n':(1,shape[0],shape[1],1)})(ybi)
    x = Multiply()([x,ysi])
    x = Add()([x,ybi])
    return x

def D_model_core(Height, Width, channel=3):
    inputs = Input((Height, Width, channel))
    x = Conv2D(64, (5, 5), padding="same", strides=(2,2))(inputs)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (5, 5), padding="same", strides=(2,2))(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, (5, 5), padding="same", strides=(2,2))(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    x = Dropout(0.5) (x)
    x = Dense(1024)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def D_model(Height, Width, channel=3, n_estimators=3):
    inputs = Input((Height, Width, channel))
    outputs = []
    for i in range(n_estimators):
        outputs.append(D_model_core(Height, Width, channel)(inputs))
    model = Model(inputs=inputs, outputs=outputs, name="discriminator")
    return model

def read_file(filepath,width,height):
    img = cv2.imread(filepath)
    if img is None:
        print("no such file {0}".format(filepath))
        return
    _width, _height, _ = img.shape
    if _width<256 or _height < 256:
        return None
    img = cv2.resize(img, (width,height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def read_file_wrap(dict):
    filepath=dict["filepath"]
    width=dict["width"]
    height=dict["height"]
    return read_file(filepath,width,height)