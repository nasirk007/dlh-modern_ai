#!/usr/bin/env python3
"""Retrieve the Pruning Path of a Decision Tree."""


def get_pruning_path(clf, X, y):
    """Which branch we need to prune first and next, so on
    and this is why we need to define the pruning path
    in the tree.

    Arguments:
    clf: A DecisionTreeClassifier instance
    X: Input features
    y: Target labels

    Returns:
    ccp_alphas: A NumPy array containing the effective alpha
    values used for pruning (post prunning).
    impurities: A NumPy array containing the total impurity of
    leaves at each corresponding alpha value.

    alpha is simply a penalty added to reduce complexity of tree.
    and to counter overfitting of the model.
    R(T) = R(T) + alpha * |T|,
    where R(T) is the error of the tree,
    and |T| is the number of leaves in the tree.

    ccp_alphas is a hyperparameter that controls the complexity of
    the model after being built. When model is overfit, it learned every detail
    and highly accurate, and to penalise this accuracy
    we need to increase the model error by adding alpha to above equation.
    becuase the goal is to minimise the accuracy of the model to the level
    model it start learning rather memorising the data.
    if ccp_alpha is too high, the model may become too simple and underfit.

    In each node (decision, leaf, root) model has gini value, higher
    the gini, node is more impure (means all classes of target variable exist
    and we cannot task classification decision to say this nodes
    belongs to yes, no, n/a class). Therefore, if alpha increased,
    tree get prunned and node become less impure. Therefore we need to find
    optimal alpha value, and We call it effective alpha value.
    See next task where model trained with effective alpha values.

    How prunning path work becuase it return 11 alpha/impurties or gini value
    rather 13, becuase the tree has total 13 nodes (Task-1). Prune is not going
    to romve nodes rather it remove 1 decision node and 2 leaf nodes.
    """
    path = clf.cost_complexity_pruning_path(X, y)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    return ccp_alphas, impurities
