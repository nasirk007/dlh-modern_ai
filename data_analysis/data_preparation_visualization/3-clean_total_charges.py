#!/usr/bin/env python3
"""this module handle missing values in DF"""


def def clean_total_charges(df, method='drop'):
    """Task to do modification 4 missing values in particular column of DF."""
    # to see output of below, we need to have object
    # like what is df, like we need to read csv
    # after df.info() tell that datatype of info sitting under each col
    # make sure, those data type make sense otherwise convert them into
    # other suitable data type, as required for analysis
    # dropna will remove rows with NaN rather empty in totalcharges
    # to get NaN, we need to convert this col to numeric becuase df.info()
    # told us this col contain string type data (object or "O")
    # this change is done at DF level for particular column
    # means all those row from dataframe will be remove based on
    # if cell are empty in one particular columns like totalcharges
    if method == "drop":
        df = df.dropna(subset=['TotalCharges'])
    elif method == "median":
        # this change is only for particular col, so stored in that col only
        df["TotalCharges"] = df["TotalCharges"].fillna(df[
            "TotalCharges"].median())
    elif method == "impute":
        df["TotalCharges"] = df["TotalCharges"].fillna(df[
            "MonthlyCharges"] * ["tenure"])
    return df
