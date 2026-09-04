#!/usr/bin/env python3
"""
Extract features backbone
MobileNet
"""
from tensorflow import keras

depthwise_separable_conv = __import__(
    '4-depthwise_separable_conv'
).depthwise_separable_conv


def mobilenet_backbone(inputs):
    """
    Extract features
    """
    x = keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=2,
        padding='same',
        use_bias=False
    )(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    block_configs = [
        (64, 1),
        (128, 2),
        (128, 1),
        (256, 2),
        (256, 1),
        (512, 2),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (1024, 2),
        (1024, 1)
    ]

    for filters, stride in block_configs:
        x = depthwise_separable_conv(
            x, filters=filters, stride=stride
        )

    return x
