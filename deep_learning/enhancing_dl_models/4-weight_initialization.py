#!/usr/bin/env python3
"""
Initialize weights (regularization)
"""
from tensorflow import keras


def build_model_initializer_by_activation(
        input_dim,
        hidden_units,
        activation
):
    """
    Use an appropriate weight initializer
    based on the activation function:
    'sigmoid', 'tanh', 'relu', or 'leaky_relu'
    returns a fucking shallow model
    """

    if not isinstance(input_dim, int):
        raise TypeError('Var input_dim shall be int')
    if not isinstance(hidden_units, int):
        raise TypeError('Var hidden_units shall be int')
    if not isinstance(activation, str):
        raise TypeError('You need to name it a special word')

    if activation in ('sigmoid', 'tanh'):
        initializer = keras.initializers.GlorotUniform()
    elif activation in ('relu', 'leaky_relu'):
        initializer = keras.initializers.HeNormal()
    else:
        raise ValueError('Check function docs for activation names')

    inputs = keras.layers.Input(shape=(input_dim,))

    if activation == 'leaky_relu':
        hidden = keras.layers.Dense(
            units=hidden_units,
            activation=keras.layers.LeakyReLU(),
            kernel_initializer=initializer
        )(inputs)
    else:
        hidden = keras.layers.Dense(
            units=hidden_units,
            activation=activation,
            kernel_initializer=initializer
        )(inputs)

    outputs = keras.layers.Dense(10, activation='softmax')(hidden)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
