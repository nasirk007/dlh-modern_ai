#!/usr/bin/env python3
"""
Shallow model (one layer of neurons)
Performs multi-class classification
"""
from tensorflow import keras


def build_model(input_dim, neurons_h):
    """
    Use Sequential class
    """
    # Fix 1: Use 'input_shape' (tuple) and 'relu' activation
    model = keras.Sequential(
        [
            keras.layers.Dense(
                neurons_h,
                activation='Sigmoid',
                input_shape=(input_dim,)  # Must be a tuple
            ),
            keras.layers.Dense(10, activation='softmax')
        ]
    )

    return model
