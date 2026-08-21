#!/usr/bin/env python3
"""
Save space, move only weights
"""


def save_model_weights(model, filepath):
    """
    Don's need to save the entire model
    Just take from what you can replicate it
    """
    model.save_weights(filepath)


def load_model_weights(model, filepath):
    """
    Git saved waits to the model
    """
    model.load_weights(filepath)
