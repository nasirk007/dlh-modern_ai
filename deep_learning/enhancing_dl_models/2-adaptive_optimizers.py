#!/usr/bin/env python3
"""
Show what is adapting for different optimizers named:
- 'sgd': Stochastic Gradient Descent
- 'adam': Adaptive Moment Estimation
- 'rmsprop': Root Mean Square Propagation
"""
from tensorflow import keras


def get_optimizer(
        name,
        learning_rate,
        momentum,
        beta_1,
        beta_2,
        rho
):
    """
    Activates attributes based on name:
    'sgd', 'adam', 'rmsprop'
    """

    if not isinstance(name, str):
        raise TypeError('First attr "name" shall be str')
    if not isinstance(learning_rate, float):
        raise TypeError('Second attr "learning rate" shall be float')
    if not isinstance(momentum, float):
        raise TypeError('Third attr "momentum" shall be float')
    if not isinstance(beta_1, float):
        raise TypeError('Fourth attr "Beta 1" shall be float')
    if not isinstance(beta_2, float):
        raise TypeError('Fifth attr "Beta 2" shall be float')
    if not isinstance(rho, float):
        raise TypeError('Sixth attr "rho" shall be float')

    if learning_rate <= 0:
        raise ValueError('Lerning rate shall be positive float')

    if name == 'sgd':
        return keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum
        )
    elif name == 'adam':
        return keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2
        )
    elif name == 'rmsprop':
        return keras.optimizers.RMSprop(
            learning_rate=learning_rate,
            rho=rho
        )
    else:
        raise ValueError('Check function doc for name values')
