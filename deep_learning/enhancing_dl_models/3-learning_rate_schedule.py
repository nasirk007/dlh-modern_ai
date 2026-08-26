#!/usr/bin/env python3
"""
Discover learning rate scheduling
"""
from tensorflow import keras


def get_optimizer_SGD_with_schedule(
        schedule_type,
        initial_lr,
        decay_steps,
        decay_rate,
        momentum
):
    """
    Scholastic Gradient Descent with
    learning rate schedules of types:
    'exponential' or 'inverse_time'
    """

    if not isinstance(schedule_type, str):
        raise TypeError('First arg "Schedule type" shall be str')
    if not isinstance(initial_lr, float):
        raise TypeError('Second arg "Initial LR" shall be float')
    if not isinstance(decay_steps, int):
        raise TypeError('Third arg "Decay steps" shall be int')
    if not isinstance(decay_rate, float):
        raise TypeError('Fourth arg "Decate rate" shall be float')
    if not isinstance(momentum, float):
        raise TypeError('Fifth arg "Momentum" shall be float')

    if schedule_type == 'exponential':
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )
    elif schedule_type == 'inverse_time':
        lr_schedule = keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )
    else:
        raise ValueError('Check function doc for shedule types')

    optimizer = keras.optimizers.SGD(
        learning_rate=lr_schedule,
        momentum=momentum
        )

    return optimizer, lr_schedule
