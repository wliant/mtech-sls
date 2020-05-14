import os
import pandas as pd


def load_dataset(name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, 'data', name + '.csv')
    df = pd.read_csv(path)
    df["date_time"] = df[['Date', 'Hour']].agg(' '.join, axis=1)
    df.set_index("date_time")
    df = df.drop(columns=['Date', 'Hour'])
    return df