#!/usr/bin/env python3
"""
Compile and train a CNN
"""
from tensorflow import keras


def complile_and_train_cnn(
        model,
        x_train,
        y_train,
        epochs,
        batch_size,
        optimizer_name='adam',
        optimizer_params=None
):
    """
    Regurn a fited CNN with it's training history
    """
    if optimizer_params is None:
        optimizer_params = {}

    optimizers = {
        'adam': keras.optimizers.Adam,
        'sgd': keras.optimizers.SGD,
        'rmsprop': keras.optimizers.RMSprop
    }

    if optimizer_name not in optimizers:
        raise ValueError(
            "Allowed optimizers: adam, sgd, or rmsprop"
        )

    optimizer = optimizers[optimizer_name](**optimizer_params)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size
    )

    return model, history
