#!/usr/bin/env python3
"""
Observe model performance with
TENSORBOARD
"""
import datetime
from tensorflow import keras


def log_to_tensorboard(
        log_dir,
        model,
        X, Y, epochs, verbose=1
):
    """
    Train a model while logging
    TensorBoard data
    """

    timestamp = datetime.datetime.now().strftime(
        '%Y%m%d-%H%M%S'
    )
    full_log_dir = f'{log_dir}/{timestamp}'

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=full_log_dir,
        histogram_freq=1
    )

    model.fit(
        X, Y,
        epochs=epochs,
        verbose=verbose,
        callbacks=[tensorboard_callback]
    )
