#!/usr/bin/env python3
"""
Generate predictions using a trained model
"""
import tensorflow as tf


def predict(model, X, verbose=0):
    """
    Predict class labels for input data X
    """
    probabilities = model.predict(X, verbose=verbose)

    return tf.argmax(probabilities, axis=1)
