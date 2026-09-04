#!/usr/bin/env python3
"""
Pull a awift brain and preserve it's knowledge
"""
from tensorflow import keras


def build_feature_extractor():
    """
    Loads MobileNetV2 without top layer,
    freezes its weights, and
    attaches a GlobalAveragePooling2D layer on top.
    """
    base_model = keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    features = base_model(inputs, training=False)
    outputs = keras.layers.GlobalAveragePooling2D()(features)

    return keras.Model(inputs, outputs)
