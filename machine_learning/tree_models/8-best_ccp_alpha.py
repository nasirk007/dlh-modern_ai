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
    # Step 1: highest test accuracy
    highest_test = max(test_scores)

    best_alpha = None
    best_clf = None
    smallest_gap = float("inf")

    for clf, train, test, alpha in zip(
            clfs,
            train_scores,
            test_scores,
            ccp_alphas):

        # Ignore models that do not have the highest test score
        if test != highest_test:
            continue

        gap = abs(train - test)

        # Better gap found
        if gap < smallest_gap:
            smallest_gap = gap
            best_alpha = alpha
            best_clf = clf

        # Same gap -> choose larger alpha
        elif gap == smallest_gap:

            if alpha > best_alpha:
                best_alpha = alpha
                best_clf = clf

    return best_alpha, best_clf
