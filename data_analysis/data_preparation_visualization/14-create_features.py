#!/usr/bin/env python3
"""
Module for creating new features from existing ones in a DataFrame.
"""
import pandas as pd


def create_features(df):
    """Create new features based on existing ones.
    """
    df = df.drop(columns=["PhoneService"])
    # create an empty column for number of services
    df["NumService"] = 0
    # create a new feature for number of services
    service_cols = ["MultipleLines", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV",
                    "StreamingMovies"]
    for col in service_cols:
        if col == "InternetService":
            df["NumService"] += df[col].map({"DSL": 1,
                                             "Fiber optic": 1, "No": 0})
        else:
            df["NumService"] += df[col].apply(lambda x: 1 if x == "Yes" else 0)
    df["TenureGroup"] = 0
    # create a new feature for tenure group
    # use pd.cut to create bins for tenure
    df["TenureGroup"] = pd.cut(df["tenure"],
                               bins=[0, 12, 24, 48, 60, float("inf")],
                               labels=[
                                   '0-12', '13-24', '25--48', '49-60', '60+'],
                               right=True)
    # labling was giving an idea to create bin for tenure like
    # 0, 12, 24 .........as it was not explicitly given
    # secondly this can be confirm that how many bins we may needed
    # df["tenure"].value_counts()
    # labling will replace internal bin index (0, 1..) with above strings
    # setting right at true means interval close on the right side
    df.drop(columns=service_cols + ["tenure"], inplace=True)
    return df
