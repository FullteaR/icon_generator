import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Lambda, LeakyReLU, Conv2D, Input, Layer, Dense, Reshape, GaussianNoise
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.backend import int_shape, ones
import numpy as np
import os
import uuid
import shutil
import subprocess
import json
import cv2


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

def parseshape(c):
    shape=int_shape(c)[1::]
    uints=1
    for s in shape:
        uints*=s
    return uints, shape

