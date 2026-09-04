#!/usr/bin/env python3
"""
Build a ResNet bottleneck residual block
"""
from tensorflow import keras as K


def bottleneck_block(
        x,
        filters,
        stride=1,
        downsample=False,
        name=None
):
    """
    Build and return a bottleneck residual block
    """
    prefix = '' if name is None else f'{name}_'

    shortcut = x

    x = K.layers.Conv2D(
        filters,
        kernel_size=(1, 1),
        strides=stride,
        padding='same',
        use_bias=False,
        name=f'{prefix}conv1'
    )(x)
    x = K.layers.BatchNormalization(
        name=f'{prefix}bn1'
    )(x)
    x = K.layers.ReLU(
        name=f'{prefix}relu1'
    )

    x = K.layers.Conv2D(
        filters,
        kernel_size=(3, 3),
        strides=1,
        padding='same',
        use_bias=False,
        name=f'{prefix}conv2'
    )(x)
    x = K.layers.BatchNormalization(
        name=f'{prefix}bn2'
    )(x)
    x = K.layers.ReLU(
        name=f'{prefix}relu2'
    )(x)

    x = K.layers.Conv2D(
        filters,
        kernel_size=(1, 1),
        strides=1,
        padding='same',
        use_bias=False,
        name=f'{prefix}conv3'
    )(x)
    x = K.layers.BatchNormalization(
        name=f'{prefix}bn3'
    )(x)

    if downsample:
        shortcut = K.layers.Conv2D(
            filters * 4,
            kernel_size=(1, 1),
            strides=stride,
            padding='same',
            use_bias=False,
            name=f'{prefix}shortcut_conv'
        )(shortcut)
        shortcut = K.layers.BatchNormalization(
            name=f'{prefix}shortcut_bn'
        )(shortcut)

    x = K.layers.Add(
        name=f'{prefix}add'
    )([x, shortcut])
    x = K.layers.ReLU(
        name=f'{prefix}out'
    )(x)

    return x
