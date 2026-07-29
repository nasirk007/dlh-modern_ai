#!/usr/bin/env python3
"""
this module contains functions to split data into train and test data.
"""
from sklearn import model_selection


def split_data(df, target='Churn', test_size=0.2, random_state=42):
    """ Split the dataset into training and testing sets.
    """
    X = df.drop(columns=[target])
    Y = df[target]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, Y, test_size=test_size, random_state=random_state)
    # chrun was encoded through lable encoder, values are 0 & 1 now 
    # and X_train.mean() will give you % customer churn rate in
    # train data and so on........ see checker 17.main.py file
    return X_train, X_test, y_train, y_test
