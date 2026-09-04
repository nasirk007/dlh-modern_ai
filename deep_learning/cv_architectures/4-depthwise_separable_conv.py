#!/usr/bin/env python3
"""
Build a Depthwize Separable Convolution block
with batch norm and ReLU
"""
from tensorflow import keras


def depthwise_separable_conv(X, filters, stride=1):
    """
    Apply depthwise separable convolution
    """

    # 1. Depthwize convolution
    x = keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=stride,
        padding='same',
        use_bias=False
    )(X)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    # 2. Pointwise convolution
    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        strides=1,
        padding='same',
        use_bias=False
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    return x
