#!/usr/bin/env python3
"""(Post-Pruning) Train and Evaluate Decision Trees
with Pruning.
"""
from sklearn import tree
train_tree = __import__('1-train').train_tree


def prune_and_evaluate_trees(
        X_train,y_train, X_test, y_test,
        ccp_alphas, random_state, min_samples_leaf,
        min_samples_split
        ):
    """function that trains multiple decision trees using range
    of alpha values and evaluate performance of each tree on
    the test set.

    Arguments:
    X_train, y_train: Training data and labels
    X_test, y_test: Testing data and labels
    ccp_alphas: A NumPy array of pruning alpha values to use for training different trees.
    random_state: Integer seed for reproducibility.
    min_samples_leaf: Mini no of samples required at a leaf node
    min_samples_split: Mini no of samples required to split an internal node

    Returns:
    clfs: A list of trained DecisionTreeClassifier instances,
    each corresponding to a ccp_alpha value.
    train_scores: A list of training accuracy scores for each classifier.
    test_scores: A list of testing accuracy scores for each classifier.

    Note with zero gini = pure node and further split needed
    means we arive the classification decision.
    Impurities is simply the gini index of the node. In task-6, we reach impurity of 0.66
    while alpha value on that path was 0.241, and if you remember
    0.664 is gini of root node in our original tree (task-1). It means with this alpha (0.245)
    we are going to chop whole tree except the roots. this alpha is too high, not good, create
    overfit model. 
    """
    clf = []
    train_score = []
    test_score = []
    for alpha in clf:






    return clf, train_score, test_score