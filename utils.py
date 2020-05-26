import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Lambda, LeakyReLU, Conv2D
from tensorflow.keras.models import Sequential
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


def visualize(g, d=None, epoch=None, row=4, col=5):
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
    plt.show()
    return noise


def Combined_model(g, d):
    model = Sequential()
    model.add(g)
    model.add(d)
    return model


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


def create(generator, size=None):
    noise_length = generator.layers[0].input.shape[1]
    while True:
        noise = np.random.uniform(-1, 1, size=(1, noise_length))
        img = generator.predict(noise)[0]
        img = img * 127.5 + 127.5
        img[img > 255] = 255
        img[img < 0] = 0
        img = img.astype(np.uint8)
        if check(img) < 0.999:
            continue
        if size:
            img = cv2.resize(img, size)
        img = cv2.bilateralFilter(img, 5, 35, 35)
        return img
