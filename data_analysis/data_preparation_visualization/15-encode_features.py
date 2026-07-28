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
    Feature and target variable encoding is one of the preprocessing step
    in EDA and ML model training.
    For practice see 15-practice.ipyn notebook on local machine.
    """
    df_enc = df.copy()
    target_le = preprocessing.LabelEncoder()
    df_enc["Churn"] = target_le.fit_transform(df_enc["Churn"])

    binary_oe = preprocessing.OrdinalEncoder(categories=[["No", "Yes"]])
    binary_columns = [
        "Partner", "Dependents", "PaperlessBilling", "SeniorCitizen"]
    for col in binary_columns:
        df_enc[col] = binary_oe.fit_transform(df_enc[[col]])
        df_enc[col] = df_enc[col].astype("int64")

    df_enc = pd.get_dummies(df_enc, columns=["Contract", "PaymentMethod"],
                            drop_first=True, dtype="int64")

    TG_oe = preprocessing.OrdinalEncoder()
    df_enc["TenureGroup"] = df_enc["TenureGroup"].astype("str")
    df_enc[["TenureGroup"]] = TG_oe.fit_transform(df_enc[[
        "TenureGroup"]]).astype("int64")

    return (df_enc, target_le, binary_oe, TG_oe)
