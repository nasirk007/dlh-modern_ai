#!/usr/bin/env python3
"""
Build complete MobileNetV1 model
"""
from tensorflow import keras

mobilenet_backbone = __import__(
    '5-mobilenet_backbone'
).mobilenet_backbone


def mobilenet(input_shape=(224, 224, 3),
              num_classes=1000):
    """
    Assemble the complete MobileNet architecture
    """

    inputs = keras.Input(shape=input_shape)

    # Backbone feature extractor
    x = mobilenet_backbone(inputs)

    # Classification head
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(
        num_classes,
        activation='softmax'
    )(x)

    return keras.Model(inputs, outputs, name='mobilenetv1')
