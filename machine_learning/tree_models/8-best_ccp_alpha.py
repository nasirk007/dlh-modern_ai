#!/usr/bin/env python3
"""
This module selects the best pruning value
ccp_alpha for a set of trained decision trees.
"""


def get_best_alpha(clfs, train_scores, test_scores, ccp_alphas):
    """
    Select the best pruning value ccp_alpha for
    a set of trained decision trees.

    - This function first identifies the model(s) that achieve
    the highest test accuracy.
    - If multiple models share this same test accuracy,
    it selects the one with the smallest difference between
    training and test accuracy to favor better generalization.
    - In the event of a further tie, the model associated
    with the largest ccp_alpha is chosen to promote a simpler,
    more regularized tree.

    Args:
        clfs: List of trained DecisionTreeClassifier instances,
              each trained with a different ccp_alpha.
        train_scores: List of training accuracy scores corresponding
                      to each classifier in clfs.
        test_scores: List of test accuracy scores corresponding
                     to each classifier in clfs as well.
        ccp_alphas: List or array of ccp_alpha values used
                    to train the classifiers.

    Returns:
        best_alpha: The most appropriate ccp_alpha value based
                    on test accuracy and generalization.
        best_clf: The trained classifier associated with the best alpha.
    """
    best_index = 0

    for i in range(1, len(clfs)):
        current_test = test_scores[i]
        best_test = test_scores[best_index]

        current_gap = abs(train_scores[i] - test_scores[i])
        best_gap = abs(
            train_scores[best_index] -
            test_scores[best_index]
            )

        if current_test > best_test:
            best_index = i
        elif current_test == best_test:
            if current_gap < best_gap:
                best_index = i
            elif current_gap == best_gap:
                if ccp_alphas[i] > ccp_alphas[best_index]:
                    best_index = i

    best_alpha = ccp_alphas[best_index]
    best_clf = clfs[best_index]

    return best_alpha, best_clf
