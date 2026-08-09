#!/usr/bin/env python3
"""perform a Grid Search to find best hyperparameters
to pre-prune decision tree classifier."""
from sklearn import model_selection


def prepruning(X, y, clf):
    """Preprunning is a technique used to prevent overfitting
    in decision trees by limiting the growth of the tree during
    training. This function performs a grid search to find the
    best combination of hyperparameters for a decision tree
    classifier. The grid search evaluates different combinations
    of the following hyperparameters:
    1. criterion: "gini" or "entropy"
    2. max_depth: integer values in the range [2, 5)
    3. min_samples_leaf: integer values in the range [2, 5)
    4. min_samples_split: integer values in the range [2, 5)

    E.g. if student understand minor details of concept, and get
    confused in exam. this is called prunning and likewise the model
    memorise minor details of training data and get confused
    in the exam. This is called overfitting. When tree is overfit, it
    become more deep and with alot of leaf nodes which contain 1 0r 2
    datapoints (samples). To avoid this, we can limit the growth of the
    tree by setting its bouundaries.

    Arguments:
    X: Input features
    y: Target labels
    clf: An untrained DecisionTreeClassifier instance
    Returns:
    A dictionary containing the best combination of
    hyperparameters found during the grid search.
    """
    # create a parameter grid of hyperparameters to be tuned
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": range(2, 5),
        "min_samples_leaf": range(2, 5),
        "min_samples_split": range(2, 5),
    }
    # create a GridSearchCV object
    grid_search = model_selection.GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
    )
    # train the model using grid search to improve model
    # performance in terms of precision, recall and F1-score.
    grid_search.fit(X, y)
    return grid_search.best_params_
