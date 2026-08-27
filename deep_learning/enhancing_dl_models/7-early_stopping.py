#!/usr/bin/env python3
"""
Regularization with Early Stopping
"""
from tensorflow import keras


def get_early_stopping_callback(
        patience,
        monitor='val_loss',
        verbose=1
):
    """
    Returns Keras ErlyStopping callback
    monitors one of the metrics:
    'val_loss' or 'val_accuracy
    """

    for (name, value) in {
        'patience': patience,
        'verbose': verbose
    }.items():
        if not isinstance(value, int):
            raise TypeError(
                f'{name} must be int, got {type(value).__name__}'
            )
    if not isinstance(monitor, str):
        raise TypeError('Arg monitor shall be str')

    if monitor not in ['val_loss', 'val_accuracy']:
        raise ValueError(
            'Check func docs for accepted values of monitor'
        )
    else:
        callback_with_best_wights = keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            verbose=verbose,
            restore_best_weights=True
        )
    return callback_with_best_wights
