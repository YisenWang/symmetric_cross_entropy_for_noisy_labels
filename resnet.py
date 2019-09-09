"""Some code sections are taken from
https://github.com/raghakot/keras-resnet
"""

import sys

import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Activation, merge, Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.merge import add
from keras import backend as K

sys.setrecursionlimit(10000)

BN_AXIS = 3


def cifar10_resnet(depth, num_classes):
    # how many layers this is going to create?
    # 3 + 6 * depth

    img_channels = 3
    img_rows = 32
    img_cols = 32
    num_conv = 3
    decay = 1e-2

    input = Input(shape=(img_rows, img_cols, img_channels))

    # 1 conv + BN + relu
    filters = 16
    b = Conv2D(filters=filters, kernel_size=(num_conv, num_conv),
               kernel_initializer="he_normal", padding="same",
               kernel_regularizer=l2(decay), bias_regularizer=l2(0))(input)
    b = BatchNormalization(axis=BN_AXIS)(b)
    b = Activation("relu")(b)

    # 1 res, no striding
    b = residual(num_conv, filters, decay, first=True)(b)  # 2 layers inside
    for _ in np.arange(1, depth):  # start from 1 => 2 * depth in total
        b = residual(num_conv, filters, decay)(b)

    filters *= 2

    # 2 res, with striding
    b = residual(num_conv, filters, decay, more_filters=True)(b)
    for _ in np.arange(1, depth):
        b = residual(num_conv, filters, decay)(b)

    filters *= 2

    # 3 res, with striding
    b = residual(num_conv, filters, decay, more_filters=True)(b)
    for _ in np.arange(1, depth):
        b = residual(num_conv, filters, decay)(b)

    b = BatchNormalization(axis=BN_AXIS)(b)
    b = Activation("relu")(b)

    b = AveragePooling2D(pool_size=(8, 8), strides=(1, 1),
                         padding="valid")(b)

    b = Flatten(name='lid')(b)

    out = Dense(units=num_classes, kernel_initializer="he_normal",
                  kernel_regularizer=l2(decay), bias_regularizer=l2(0))(b)

    act = Activation(tf.nn.softmax)(out)

    return Model(inputs=input, outputs=act)


def residual(num_conv, filters, decay, more_filters=False, first=False):
    def f(input):
        # in_channel = input._keras_shape[1]
        out_channel = filters

        if more_filters and not first:
            # out_channel = in_channel * 2
            stride = 2
        else:
            # out_channel = in_channel
            stride = 1

        if not first:
            b = BatchNormalization(axis=BN_AXIS)(input)
            b = Activation("relu")(b)
        else:
            b = input

        b = Conv2D(filters=out_channel,
                   kernel_size=(num_conv, num_conv),
                   strides=(stride, stride),
                   kernel_initializer="he_normal", padding="same",
                   kernel_regularizer=l2(decay), bias_regularizer=l2(0))(b)
        b = BatchNormalization(axis=BN_AXIS)(b)
        b = Activation("relu")(b)
        res = Conv2D(filters=out_channel,
                     kernel_size=(num_conv, num_conv),
                     kernel_initializer="he_normal", padding="same",
                     kernel_regularizer=l2(decay), bias_regularizer=l2(0))(b)

        # check and match number of filter for the shortcut
        input_shape = K.int_shape(input)
        residual_shape = K.int_shape(res)
        if not input_shape[3] == residual_shape[3]:
            stride_width = int(round(input_shape[1] / residual_shape[1]))
            stride_height = int(round(input_shape[2] / residual_shape[2]))

            input = Conv2D(filters=residual_shape[3], kernel_size=(1, 1),
                           strides=(stride_width, stride_height),
                           kernel_initializer="he_normal",
                           padding="valid", kernel_regularizer=l2(decay))(input)

        return add([input, res])

    return f