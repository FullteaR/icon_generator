import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Lambda, LeakyReLU, Conv2D, Input, Layer, Dense, Reshape, GaussianNoise, LayerNormalization, Multiply, Add, Flatten, LeakyReLU, UpSampling2D, GlobalMaxPooling2D, BatchNormalization, MaxPooling2D, Dropout
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

from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import to_categorical, plot_model
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
    fig = plt.figure(figsize=(col * 3, row * 3))
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
    plt.close(fig)
    return noise

def calcscore(g, d, sample=1000):
    noise_length = g.layers[0].input.shape[1]
    noise = np.random.uniform(-1, 1, size=(sample, noise_length))
    pred = g.predict(noise, verbose=0)
    scores = d.predict(pred)
    return np.mean(scores)
    
    

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
    ysi = Dense(ch, name="adain_ysi")(z)
    ysi = Reshape((1,1,ch))(ysi)
    ysi = Lambda(K.tile, arguments={'n':(1,shape[0],shape[1],1)})(ysi)
    ybi = Dense(ch, name="adain_ybi")(z)
    ybi = Reshape((1,1,ch))(ybi)
    ybi = Lambda(K.tile, arguments={'n':(1,shape[0],shape[1],1)})(ybi)
    x = Multiply()([x,ysi])
    x = Add()([x,ybi])
    return x

def D_model_core(Height, Width, channel=3):
    inputs = Input((Height, Width, channel))
    x = Conv2D(64, (5, 5), padding="same", strides=(2,2))(inputs)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling2D(padding="same")(x)
    x = Conv2D(128, (5, 5), padding="same", strides=(2,2))(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling2D(padding="same")(x)
    x = Conv2D(256, (5, 5), padding="same", strides=(2,2))(x)
    x = GlobalMaxPooling2D()(x)
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

def D_model_core2(Height, Width, channel=3):
    inputs = Input((Height, Width, channel))
    x = Conv2D(64, (5, 5), padding="same", strides=(2,2))(inputs)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling2D(padding="same")(x)
    x = Conv2D(128, (5, 5), padding="same", strides=(2,2))(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling2D(padding="same")(x)
    x = Conv2D(256, (5, 5), padding="same", strides=(2,2))(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(0.2)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def D_model2(Height, Width, channel=3, n_estimators=3):
    inputs = Input((Height, Width, channel))
    outputs = []
    for i in range(n_estimators):
        outputs.append(D_model_core2(Height, Width, channel)(inputs))
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


class Constant(Layer):
    def __init__(self, output_dim, trainable=True, **kwargs):
        self.output_dim = output_dim
        self.trainable=trainable
        super(Constant, self).__init__(**kwargs)
    def build(self, input_shape):
        self.bias = self.add_weight(shape=(1, self.output_dim), initializer="uniform", trainable=self.trainable)
    def call(self, x):
        return self.bias
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    def get_config(self):
        config = super().get_config().copy()
        config.update({"output_dim":self.output_dim})
        return config


def AdaINBlock(x,z, use_bias=True):
    x = LayerNormalization(axis=[1,2])(x)
    shape = parseshape(x)
    assert len(shape)==3
    ch = shape[2]
    ysi = Dense(ch, use_bias=use_bias)(z)
    ysi = GaussianNoise(0.5)(ysi)
    ysi = Reshape((1,1,ch))(ysi)
    ysi = Lambda(K.tile, arguments={'n':(1,shape[0],shape[1],1)})(ysi)
    ybi = Dense(ch, use_bias=use_bias)(z)
    ybi = GaussianNoise(0.5)(ybi)
    ybi = Reshape((1,1,ch))(ybi)
    ybi = Lambda(K.tile, arguments={'n':(1,shape[0],shape[1],1)})(ybi)
    x = Multiply()([x,ysi])
    x = Add()([x,ybi])
    return x




#https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate. In a standard GAN, the
    discriminator has a sigmoid output, representing the probability that samples are
    real or generated. In Wasserstein GANs, however, the output is linear with no
    activation function! Instead of being constrained to [0, 1], the discriminator wants
    to make the distance between its output for real and generated samples as
    large as possible.
    The most natural way to achieve this is to label generated samples -1 and real
    samples 1, instead of the 0 and 1 used in normal GANs, so that multiplying the
    outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be)
    less than 0."""
    return K.mean(y_true * y_pred)


#https://gist.github.com/a-re/8ac20f2abd00be917d18eab7b76dde96
class MinibatchStd(Layer):
    # initialize the layer
    def __init__(self, **kwargs):
        super(MinibatchStd, self).__init__(**kwargs)

    # perform the operation
    def call(self, inputs, **kwargs):
        # calculate the mean value for each pixel across channels
        mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
        # calculate the average of the squared differences (variance) and add a small constant for numerical stability
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=0, keepdims=True) + 1e-8
        # calculate the mean standard deviation across each pixel coord. stddev = sqrt of variance
        average_stddev = tf.reduce_mean(tf.sqrt(variance), keepdims=True)
        # Scale this up to be the size of one input feature map for each sample
        shape = tf.shape(inputs)
        minibatch_stddev = tf.tile(average_stddev, (shape[0], shape[1], shape[2], 1))
        # concatenate minibatch std feature map with the input feature maps (axis=-1 if data_format=NHWC)
        return tf.concat([inputs, minibatch_stddev], axis=-1)

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[-1] += 1  # batch-wide std adds one additional channel
        return tuple(input_shape)