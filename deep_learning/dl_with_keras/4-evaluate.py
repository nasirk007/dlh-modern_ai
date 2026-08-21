#!/usr/bin/env python3
"""
Evaluate keras model
gosh, how that keras sounds like keiras
it's good to be sabra as the hacker's name
"""


def evaluate_model(model, X, Y, verbose=0):
    """
    Return loss and accuracy, which is, suprisingly, different shit
    """

    loss, accuracy = model.evaluate(X, Y, verbose=verbose)

    return loss, accuracy
