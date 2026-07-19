#!/usr/bin/env python3
"""this module handle duplicates rows in DF"""


def remove_duplicates(df):
    """Task to remove duplicate rows from DF."""
    # duplicate method by default turn column into row and check duplicates
    # but if you want to check duplicates by column, use transpose before
    # duplicate function e.g df.T.duplicated()
    df = df.drop_duplicates()
    return df
