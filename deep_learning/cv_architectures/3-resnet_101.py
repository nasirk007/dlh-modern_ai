#!/usr/bin/env python3
"""
Build a ResNet-101 model.
"""
from tensorflow import keras

bottleneck_block = __import__('2-bottleneck_block').bottleneck_block


def make_layer(x, blocks, filters, stride=1, name=None):
    """
    Build one ResNet stage.
    """
    x = bottleneck_block(
        x,
        filters,
        stride=stride,
        downsample=True,
        name=f'{name}_block1'
    )
    for i in range(1, blocks):
        x = bottleneck_block(
            x,
            filters,
            stride=1,
            downsample=False,
            name=f'{name}_block{i + 1}'
        )
    return x


def build_resnet101(input_shape=(224, 224, 3), num_classes=1000):
    """
    Build and return the ResNet-101 architecture.
    """
    inputs = keras.Input(shape=input_shape)

    x = keras.layers.Conv2D(
        64,
        kernel_size=(7, 7),
        strides=2,
        padding='same',
        use_bias=False,
        name='conv1'
    )(inputs)
    x = keras.layers.BatchNormalization(name='bn1')(x)
    x = keras.layers.ReLU(name='relu1')(x)
    x = keras.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=2,
        padding='same',
        name='maxpool'
    )(x)

    x = make_layer(x, 3, 64, stride=1, name='layer1')
    x = make_layer(x, 4, 128, stride=2, name='layer2')
    x = make_layer(x, 23, 256, stride=2, name='layer3')
    x = make_layer(x, 3, 512, stride=2, name='layer4')

    x = keras.layers.GlobalAveragePooling2D(name='avgpool')(x)
    outputs = keras.layers.Dense(
        num_classes,
        activation='softmax',
        name='fc'
    )(x)

    return keras.Model(inputs, outputs, name='resnet101')
