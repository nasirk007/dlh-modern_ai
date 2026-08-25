#!/usr/bin/env python3
"""
Compare Gradient Descent Variants
-optimization-
"""
from tensorflow import keras


def train_with_gradient_descent_variant(
        variant,
        learning_rate,
        x_train,
        batch_size
):
    """
    Returns gradient descent 'optimizer' configured with
    specified learning rate, and
    the batch sise 'bs' based on the selected variant:
    'batch', 'stochastic', oder 'mini_batch'
    """

    if not isinstance(variant, str):
        raise TypeError('Argument "variant" shall be string')
    if not isinstance(learning_rate, (float, int)):
        raise TypeError('Argument "learning_rate" shall be float')
    if not hasattr(x_train, '__len__'):
        raise TypeError('Argument "x_train" shall be array-like')
    if not isinstance(batch_size, int):
        raise TypeError('Argument "batch_size" shall be integer')
    if batch_size <= 0:
        raise ValueError('Argument "batch_size" shall be greater than 0')

    optimizer = keras.optimizers.SGD(
        learning_rate=learning_rate,

    )

    if variant == 'batch':
        bs = len(x_train)
    elif variant == 'stochastic':
        bs = 1
    elif variant == 'mini_batch':
        bs = batch_size
    else:
        raise ValueError('Argument "variant" smels funny')

    return optimizer, bs
