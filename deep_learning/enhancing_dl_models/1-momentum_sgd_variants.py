#!/usr/bin/env python3
"""
Stochastic Gradient Descent (SDG) Variants
based on momentum. WTF momentum and WTF momentum
"""
from tensorflow import keras


def get_optimizer_SGD(
        name, lr,
        momentum=0.0,
        nesterov=False
):
    """
    Returns SGD optimiser based on complectation and names:
    'SGD', 'SGD+Momentum', or 'SGD+Momentum+Nesterov'
    """

    if not isinstance(name, str):
        raise TypeError('First attr "name" shall be string')
    if not isinstance(lr, float):
        raise TypeError('Second attr "learning rate" shall be float')
    if not isinstance(momentum, float):
        raise TypeError('Third attr "momentum" shall be float')
    if not isinstance(nesterov, bool):
        raise TypeError('Fourth attr "nesterov" shall be boolean')

    if name == 'SGD':
        momentum, nesterov = 0.0, False
    elif name == 'SGD+Momentum':
        nesterov = False
    elif name == 'SGD+Momentum+Nesterov':
        nesterov = True
    else:
        raise ValueError('Check func documentation for possible names')

    optimizer = keras.optimizers.SGD(
        learning_rate=lr,
        momentum=momentum,
        nesterov=nesterov
    )

    return optimizer
