from data_generator import load_dataset
import pandas as pd


df = load_dataset()
print(df.head())
print(df.describe())
