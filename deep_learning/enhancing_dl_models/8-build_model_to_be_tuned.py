#!/usr/bin/env python3
"""
Build a keras model for mltcls classification,
where the architecture and training params
are tuned with keras tuner
"""
from tensorflow import keras


def build_model(hp):
    """
    Returns sequential model based on
    hyperparameters defined in
    the 'hp' object
    """

    model = keras.Sequential()
    model.add(
        keras.layers.Input(shape=(784,))
    )

    num_layers = hp.Int(
        'num_layers',
        min_value=1,
        max_value=2
    )
    units = hp.Int(
        'units',
        min_value=4,
        max_value=12,
        step=4
    )
    activation = hp.Choice(
        'activation',
        values=['relu', 'sigmoid']
    )

    for _ in range(num_layers):
        model.add(
            keras.layers.Dense(
                units=units,
                activation=activation
            )
        )

    model.add(keras.layers.Dense(10, activation='softmax'))

    learning_rate = hp.Choice(
        'learning_rate',
        values=[1e-2, 1e-3]
    )
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
