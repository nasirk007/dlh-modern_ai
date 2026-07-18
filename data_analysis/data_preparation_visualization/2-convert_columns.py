#!/usr/bin/env python3
"""visualise convert column datatype dataframe"""
import pandas as pd


def convert_columns(df):
    """This task isto convert columns in DF into different data type."""
    # to_nunmeric is pandas method and cannot be applied over dataframe
    # after df.info(), we noted totalcharges column = object type, means string
    # convert string into numeric (integer, float) for analysis
    # "28.95" is string rather integer and numeric function will work
    # but for "abc" or " " it will return error and there are different option
    # to treat error, raise it, ignore it or convert to NaN so
    # coerce will do conversion NaN and its required per task requirement
    # map function syntax used dictionary style for replacement
    # oldvalue ==> newvalue, and so one
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors="coerce")
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: "No", 1: "Yes"})
    return df
