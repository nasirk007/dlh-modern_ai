#!/usr/bin/env python3
"""
Configure model for training
"""
from tensorflow import keras


def compile_model(model, learning_rate=0.01):
    """
    Stochastic gradient descent,
    Binary cross-entropy loss,
    Accuracy metric for monitoring
    """
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
