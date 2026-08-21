#!/usr/bin/env python3
"""
Shallow model (one layer of neirones)
Performs multi-class classification
"""
from tensorflow import keras


def build_model(input_dim, neurons_h):
    """
    Use Sequential class
    """

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(neurons_h, activation='sigmoid'),
            keras.layers.Dense(10, activation='softmax')
        ]
    )

    return model
