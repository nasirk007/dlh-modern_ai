#!/usr/bin/env python3
"""
Initialize keras tuner for tuning
of hyperparameters
"""
import keras_tuner


def initiate_tuner(
        tuner_type,
        build_model,
        seed,
        hyperband_iterations,
        max_trials,
        objective
):
    """
    Returns keras tuner object ready to use
    for hyperparameter optimization, based on tuner types:
    'Hyperband', 'RandomSearch', or 'BayesianOptimization'
    """

    overwrite = True

    if tuner_type == 'Hyperband':
        tuner = keras_tuner.Hyperband(
            hypermodel=build_model,
            objective=objective,
            max_epochs=10,
            hyperband_iterations=hyperband_iterations,
            seed=seed,
            overwrite=overwrite
        )
    elif tuner_type == 'RandomSearch':
        tuner = keras_tuner.RandomSearch(
            hypermodel=build_model,
            objective=objective,
            max_trials=max_trials,
            seed=seed,
            overwrite=overwrite
        )
    elif tuner_type == 'BayesianOptimization':
        tuner = keras_tuner.BayesianOptimization(
            hypermodel=build_model,
            objective=objective,
            max_trials=max_trials,
            seed=seed,
            overwrite=overwrite
        )
    else:
        raise ValueError(
            'Check func docs for accepted tuner types'
        )

    return tuner
