#!/usr/bin/env python3
"""
Find the best model and hyperparameters
for a particular case
"""


def search_and_return_best_model(
        tuner,
        x_train,
        y_train,
        epochs,
        validation_split,
        verbose=0
):
    """
    Run hyperparameter search and return best hyperparameters
    """

    tuner.search(
        x_train,
        y_train,
        epochs=epochs,
        validation_split=validation_split,
        verbose=verbose
    )

    best_hyperparameters = tuner.get_best_hyperparameters(
        num_trials=1
    )[0]

    return best_hyperparameters
