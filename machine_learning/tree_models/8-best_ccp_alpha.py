#!/usr/bin/env python3
"""(Post-Pruning) find the Best alpha for Pruning."""
from sklearn import tree


def get_best_alpha(clfs, train_scores, test_scores, ccp_alphas):
    """Function that selects the best pruning alpha value for a set
    of trained decision trees. This function first identifies the model(s)
    that achieve the highest test accuracy. If multiple models share same
    test accuracy, it selects the one with the smallest difference between
    training and test accuracy to favor better generalization.
    In the event of a further tie, the model associated with the largest
    ccp_alpha is chosen to promote a simpler, more regularized tree.

    Arguments:
    clfs: List of trained DecisionTreeClassifier instances, each trained with
    a different ccp_alpha.
    train_scores: List of training accuracy scores corresponding to each
    classifier in clfs.
    test_scores: List of test accuracy scores corresponding to each classifier
    in clfs as well.
    ccp_alphas: List or array of ccp_alpha values used to train the classifiers.

    Returns:
    best_alpha: The most appropriate ccp_alpha value based on test accuracy
    and generalization.
    best_clf: The trained classifier associated with the best alpha.
    Find the best alpha value based on test performance.
    """

    # Identify the maximum test accuracy achieved
    max_test_score = max(test_scores)
    


    

