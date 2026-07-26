#!/usr/bin/env python3
"""
Module for creating new features from existing ones in a DataFrame.
"""
import pandas as pd


def create_features(df):
    """Create new features based on existing ones.
    """
    df = df.drop(columns=["PhoneService"])
    df["NumService"] = 0
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
    df["TenureGroup"] = pd.cut(df["tenure"],
                               bins=[0, 12, 24, 48, 60, float("inf")],
                               labels=[
                                   '0-12', '13-24', '25--48', '49-60', '60+'
                                   ],
                               right=True)
    df = df.drop(columns=service_cols + ["tenure"])
    return df
