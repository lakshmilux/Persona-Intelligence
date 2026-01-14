import pandas as pd
from typing import List

## TO DO:
## if number of obs per level <10 then group it

def create_stratification_df(X: pd.DataFrame, stratify_key: pd.Series, dataset_name: str, fold_number: int) -> pd.DataFrame:
    """
    Generates a DataFrame of stratification proportions for a given dataset
    without using the target variable.

    Args:
        X (pd.DataFrame): The feature DataFrame.
        stratify_key (pd.Series): The Series used for stratification.
        dataset_name (str): The name of the dataset (e.g., "Full Train", "Inner Fold 1").
        fold_number (int): The fold number for this dataset.

    Returns:
        pd.DataFrame: A DataFrame with columns 'Category', 'Proportion', 'Fold', and 'Dataset'.
    """
    # Combine the stratification key with the index to ensure alignment
    combined_df = pd.DataFrame({
        'stratify_key': stratify_key.loc[X.index]
    })

    # Calculate the size of each unique stratify_key group
    counts = combined_df.groupby('stratify_key').size().reset_index(name='Count')

    # Calculate the total number of items
    total_count = counts['Count'].sum()

    # Calculate proportions
    counts['Proportion'] = counts['Count'] / total_count
    
    # Rename and add the dataset name column
    counts = counts.rename(columns={'stratify_key': 'Category'})
    counts['Dataset'] = dataset_name
    counts['Fold'] = fold_number

    # Drop the intermediate 'Count' column
    return counts[['Category', 'Proportion', 'Fold', 'Dataset']]

