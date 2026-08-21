#!/usr/bin/env python3
"""
Build functional shallow newral network
"""
from tensorflow import keras


def build_model(input_dim, neurons_h):
    """
    Perform multiclass classification
    without using the Sequential class
    """

    inputs = keras.layers.Input(shape=(input_dim,))
    hidden = keras.layers.Dense(neurons_h, activation='sigmoid')(inputs)
    outputs = keras.layers.Dense(10, activation='softmax')(hidden)

    model = keras.Model(inputs, outputs)

    return model
