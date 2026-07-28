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
    # label encoding for target variable (y = dependent variable and x = independent variable)
    # in data preprocessing, we drop target variable like churn from the feature set and encode it separately. 
    # This is because we want to predict the target variable based on the features.
    # lable encoder works in ID array so it expect input as 1D array
    # fit_transform() method is used to fit the encoder and transform the data in one step.
    # e.g. it will convert 'Yes' to 1 and 'No' to 0 for the target variable 'Churn'. 
    target_le = preprocessing.LabelEncoder()
    df_enc["Churn"] = target_le.fit_transform(df_enc["Churn"])

    binary_oe = preprocessing.OrdinalEncoder(categories=[["No", "Yes"]])
    # in contrast OrdinalEncoder is used for 2D or for multiple categorial/feature columns.
    # ordinal encoder is transformer of sklearn library which is used to convert
    # categorical features into numerical values.
    # it explicitly like to know categories of the categorical features in advance,
    # so we need to pass categories parameter to the OrdinalEncoder() method.
    # afterward loop is required to encode all the categories in binary columns into numerical values.
    binary_columns = [
        "Partner", "Dependents", "PaperlessBilling", "SeniorCitizen"]
    for col in binary_columns:
        df_enc[col] = binary_oe.fit_transform(df_enc[[col]])
        df_enc[col] = df_enc[col].astype("int64")

    # onhot encoder is another transformer of sklearn library which is used
    # to convert more than 2 categorical features into numerical values.
    # one hot encoding is used to convert categorical variables into numerical variables.
    # also it create dummy variables for each category in the categorical variable.
    # e.g. contract has 3 categories: Month-to-month, One year, Two year. 
    # One hot encoding will create 2 dummy variables for each category.
    # if we dont apply drop_first=True, it will create 3 dummy variables for each category. 
    # But we can drop one dummy variable to avoid multicollinearity.
    # this is illustrated in 15-practice.ipyn notebook on local machine.
    # there are two way for onehot encoding, either use sklearn preprocessing encoder
    # or pandas get_dummies() method. if we use sklearn for this, its quite complex and 
    # below code line through value error becuase onehot encoder expect 2D array and
    # secondly it will produce sparse matrix which is not compatible with pandas dataframe.
    # of df_enc[["Contract", "PaymentMethod"]], means output is matrix or table of 6 columns and 7043 rows.
    # or 4 columns if we used drop_first=True parameter otherwise. 
    # Means we cannot store this matrix in pandas dataframe directly.
    # So we use pandas get_dummies() method which is simple and easy to use.
    # df_enc[["Contract", "PaymentMethod"]] = preprocessing.OneHotEncoder().fit_transform(df_enc[["Contract", "PaymentMethod"]]) 
    df_enc = pd.get_dummies(df_enc, columns=["Contract", "PaymentMethod"],
                            drop_first=True, dtype="int64")

    TG_oe = preprocessing.OrdinalEncoder()
    # ordinalencoder expect 2D array or 2 column table as input and work on numeric data type
    # Tenure Group is single column and we just use two square bracket to convert into 2D array
    # Secondly, TenureGroup is of object data type (string), so we need to convert it into
    # string data type before applying ordinalencoder. To convert into numeric
    # we can use pd.to_numeric method as well but this will produce NaN values for
    # non-numeric data type and we will critical data before encoding and modelling
    # pd.to_numeric() method in initial stage of data clearning and transforming 
    # rather encoding feature stage, see below for understanding of preprocessing steps
    # https://www.geeksforgeeks.org/data-analysis/data-preprocessing-machine-learning-python/
    df_enc["TenureGroup"] = df_enc["TenureGroup"].astype("str")
    df_enc[["TenureGroup"]] = TG_oe.fit_transform(df_enc[["TenureGroup"]])

    return (df_enc, target_le, binary_oe, TG_oe)
