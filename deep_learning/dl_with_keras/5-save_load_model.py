#!/usr/bin/env python3
"""
Save and reload a kerashmodel
including its architecture, wights,
and optimizer state
"""
from tensorflow import keras


def save_model(model, filepath):
    """
    Host your bitch with her beautycase
    in the secret place, you know the path
    """
    model.save(filepath)


def load_model(filepath):
    """
    Call your bitch when you need her
    """
    return keras.models.load_model(filepath)
