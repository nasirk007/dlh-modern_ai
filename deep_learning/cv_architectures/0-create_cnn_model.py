#!/usr/bin/env python3
"""
Create and compile a configurable
convolutional neural network (CNN)
"""
from tensorflow import keras


def create_cnn_model(
        input_shape,
        filters,
        kernel_sizes,
        activations,
        pooling_type='max'
):
    """
    Build and compile a CNN with
    a single pooling layer per confolution
    """
    if not (len(
        filters
        ) == len(
            kernel_sizes
            ) == len(
                activations
                )):
        raise ValueError(
            'filters, kernel_sizes, activation must me eq len'
            )
    if pooling_type not in ('max', 'avg'):
        raise ValueError(
            'pooling_type must be "max" or "avg"'
        )

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=input_shape))

    if pooling_type == 'max':
        pooling_layer = (
            keras.layers.MaxPooling2D
        )
    else:
        pooling_layer = (
            keras.layers.AveragePooling2D
        )

    for layer_filters, kernel_size, activation in zip(
            filters, kernel_sizes, activations):
        model.add(keras.layers.Conv2D(
            filters=layer_filters,
            kernel_size=kernel_size,
            activation=activation,
            padding='valid'
        ))
        model.add(pooling_layer(
            pool_size=(2, 2)
        ))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(
         10, activation='softmax'
    ))

    model.compile(
         optimizer='adam',
         loss='sparse_categorical_crossentropy',
         metrics=['accuracy']
    )

    return model
