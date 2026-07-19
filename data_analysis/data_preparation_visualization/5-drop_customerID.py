#!/usr/bin/env python3
"""this module handle droping col in DF"""
import pandas as pd


def drop_customerID(df):
    """Task to drop customerID column."""
    df = df.drop(columns=[]"customerID"])
    return df
