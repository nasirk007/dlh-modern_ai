#!/usr/bin/env python3
"""
Train keras model
"""


def train_model(model, X, Y, epochs, verbose=1):
    """
    Buy fitness memborship to your bitch!
    """
    model.fit(
        X, Y,
        epochs=epochs,
        verbose=verbose
    )
