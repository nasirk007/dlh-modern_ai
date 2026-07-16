#!/usr/bin/env python3
import pandas as pd
df = pd.read_csv('Telco-Customer-Churn.csv')
shape = df.shape
data_types = df.dtypes
head = df.head(5)
missing_count = df.isnull().sum()
duplicates = df.duplicated().sum()
print("Shape:", shape)
print("Dtypes:\n", data_types)
print("First rows:\n", head)
print("Missing values:\n", missing_count)
print("Duplicates:", duplicates)
