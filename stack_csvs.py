import pandas as pd
import os

current_path = os.listdir(os.getcwd())
concat_csvs = []

for filename in current_path:
    if filename[-5:] == "y.csv":
        df = pd.read_csv(filename, index_col=None, header=0)
        concat_csvs.append(df)
    else:
        continue 

frame = pd.concat(concat_csvs, axis=0, ignore_index=True)
frame.to_csv('testSummaryConsol.csv')