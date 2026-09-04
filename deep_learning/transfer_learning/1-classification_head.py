#!/usr/bin/env python3
"""
Add a classification head
on top of one with the brains (feature extractor)
"""
from tensorflow import keras


def add_classification_head(
        base_model,
        num_classes
):
    """
    Add hidden and output classification
    layers to a feature extractor
    """
    inputs = keras.Input(
        shape=base_model.input_shape[1:]
    )
    features = base_model(inputs)
    hidden = keras.layers.Dense(
        128,
        activation='relu'
    )(features)
    outputs = keras.layers.Dense(
        num_classes,
        activation='softmax'
    )(hidden)

    return keras.Model(inputs, outputs)
