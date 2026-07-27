#!/usr/bin/env python3
"""
this module contains functions to encode categorical
features into numerical value using scikit-learn, library
famous for ML work using python.
"""
import pandas as pd
from sklearn import preprocessing


def encode_features(df):
    """
    Scikit-learn library provide tools for data preprocessing,
    feature encoding, traninig models, prediction and evaluation.
    In this function, we will use preprocessing module of scikit-learnt
    to encode categorical features into numeric. Becuase, ML model
    understand numerical rather strings type data. Therefore function
    needs to returns DataFrame with encoded features.
    Preprocessing isto clean and transform data before training ML model
    on the data. 
    """
