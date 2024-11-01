import pandas as pd
import polaris as po
from egfr_binder_rd2 import DATA_DIR

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, GroupShuffleSplit
import numpy as np
import pandas as pd

def get_data():
    fp = 'https://raw.githubusercontent.com/adaptyvbio/egfr_competition_1/refs/heads/main/results/replicate_summary.csv'
    df = pd.read_csv(fp)

    # Load the dataset from the Hub
    dataset = po.load_dataset("adaptyv-bio/egfr-binders-v0")

    merged_df = dataset.table.merge(df, how='outer')

    merged_df['length'] = merged_df['sequence'].apply(len)
    return merged_df

def get_data_with_scraped_expression():
    df = get_data()
    expression = pd.read_csv(DATA_DIR / 'scraped_egfr_binder_expression.csv')
    expression['expression'] = expression['expression'].fillna('None')
    
    individuals = expression[expression['designer'].isna()].iloc[1:]
    individuals[['name', 'replicate']] = individuals['design_name'].str.split(' #', expand=True)
    
    clean_exp = individuals[['name', 'replicate', 'expression']]
    clean_exp['replicate'] = clean_exp['replicate'].astype(int)
    
    mdf = df.drop(columns=['expression']).merge(clean_exp, on=['name', 'replicate'], how='outer')
    mdf.loc[:1, 'expression'] = 'Low'

    expression_map = {'Low': 1, 'Medium': 2, 'High': 3, 'None': 0}
    mdf['encoded_expression'] = mdf['expression'].map(expression_map)
    
    return mdf



def create_data_splits_grouped(df, test_size=0.2, val_size=0.2, random_state=42):

    X = df['sequence'].tolist()
    y = df['encoded_expression'].tolist()
    groups = df['username'].fillna('adaptyv').tolist()

    # Calculate train size
    train_size = 1 - test_size - val_size

    # Split the data into train, validation, and test sets
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size + val_size, random_state=random_state)
    train_idx, temp_idx = next(gss.split(X, y, groups))

    # Split the temporary set into validation and test sets
    val_test_ratio = val_size / (test_size + val_size)
    gss_val_test = GroupShuffleSplit(n_splits=1, test_size=1-val_test_ratio, random_state=random_state)
    val_idx, test_idx = next(gss_val_test.split(np.array(X)[temp_idx], np.array(y)[temp_idx], np.array(groups)[temp_idx]))

    # Adjust indices for the original dataset
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    # Create DataFrames for each split
    train_df = df.iloc[train_idx].assign(stage='train')
    val_df = df.iloc[val_idx].assign(stage='valid')
    test_df = df.iloc[test_idx].assign(stage='test')

    # Combine splits
    split_df = pd.concat([train_df, val_df, test_df])

    return split_df

def create_data_splits_stratified(df, test_size=0.2, val_size=0.2, random_state=42):

    # Create a combined stratification column
    df['strat'] = df['username'].fillna('adaptyv') + '_' + df['encoded_expression'].astype(str)

    try:
        # Attempt stratified split
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size + val_size, random_state=random_state)
        train_idx, temp_idx = next(sss1.split(df, df['strat']))

        # Second split: val vs test
        val_test_ratio = val_size / (test_size + val_size)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1-val_test_ratio, random_state=random_state)
        val_idx, test_idx = next(sss2.split(df.iloc[temp_idx], df.iloc[temp_idx]['strat']))

        # Adjust indices for the original dataset
        val_idx = temp_idx[val_idx]
        test_idx = temp_idx[test_idx]

    except ValueError:
        print("Warning: Stratified split not possible. Falling back to random split.")
        # Fallback to random split
        train_idx, temp_idx = train_test_split(
            range(len(df)), test_size=test_size + val_size, random_state=random_state
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=test_size / (test_size + val_size), random_state=random_state
        )

    # Create DataFrames for each split
    train_df = df.iloc[train_idx].assign(stage='train')
    val_df = df.iloc[val_idx].assign(stage='valid')
    test_df = df.iloc[test_idx].assign(stage='test')

    # Combine splits and rename 'encoded_expression' to 'label'
    split_df = pd.concat([train_df, val_df, test_df]).rename(columns={'encoded_expression': 'label'})

    return split_df
