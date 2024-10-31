import pandas as pd

def get_dataset():
    df = pd.read_csv('/home/naka/code/egfr_binder_rd2/data/fold_df.csv', index_col=0)
    return df