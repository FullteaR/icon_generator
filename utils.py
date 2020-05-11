import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Lambda, LeakyReLU, Conv2D 
from keras.models import Sequential
import numpy as np
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

def visualize(g,d,epoch=None, row=4, col=5):
    noise_length = g.layers[0].input.shape[1]
    plt.figure(figsize=(col*3,row*3))
    noise = np.random.uniform(-1, 1, size=(row*col, noise_length))
    pred = g.predict(noise, verbose=0)
    losses = d.predict(pred)
    if epoch:
        plt.suptitle("epoch={0}".format(epoch),fontsize=15)
    for i in range(row * col):
        plt.subplot(row, col, i+1)
        plt.imshow((pred[i]*127.5+127.5).astype(np.uint8))
        plt.title(losses[i][0])
        plt.axis('off')
    plt.show()


def Combined_model(g, d):
    model = Sequential()
    model.add(g)
    model.add(d)
    return model
