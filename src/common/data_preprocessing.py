import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
import json
from typing import List, Dict, Any, Tuple, Optional, Union
from src.common.config import ModelConfig


def get_champion_prefix_from_metadata(config: ModelConfig) -> Optional[str]:
    """Retrieves the champion model prefix from the saved metadata file."""
    
    # Reconstruct the expected path to the metadata file
    metadata_dir = os.path.dirname(config.shap_path)
    metadata_file_path = os.path.join(metadata_dir, "champion_metadata.json")

    if not os.path.exists(metadata_file_path):
        print(f"Metadata file not found at: {metadata_file_path}")
        return None
    
    try:
        with open(metadata_file_path, 'r') as f:
            metadata = json.load(f)
            # We return the prefix (e.g., CatBoost) to be used in load_and_split_data
            return metadata.get("champion_model_prefix")
    except Exception as e:
        print(f"Error reading metadata file: {e}")
        return None
      
def read_dataset(filepath: str) -> pd.DataFrame:
    """
    Reads a dataset from a file based on its extension and returns it as a pandas DataFrame.
    Supported file types:
    - .csv
    - .xlsx (Excel files)
    - .xls (older Excel files)
    - .parquet (Parquet files)
    - .json (JSON files)
    Args:
        filepath (str): The full path to the file, including the filename and extension.
    Returns:
        pd.DataFrame: The dataset loaded into a pandas DataFrame.
    Raises:
        ValueError: If the file type is unsupported or the file cannot be read.
    """
    # Extract the file extension
    file_extension = os.path.splitext(filepath)[1].lower()  # Get the file extension (e.g., '.csv')

    try:
        # Read the file based on its extension
        if file_extension == ".csv":
            return pd.read_csv(filepath)
        elif file_extension in [".xlsx", ".xls"]:
            return pd.read_excel(filepath,engine='openpyxl')
        elif file_extension == ".parquet":
            return pd.read_parquet(filepath)
        elif file_extension == ".json":
            return pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    except Exception as e:
        raise ValueError(f"Error reading the file: {filepath}. Details: {e}")

def load_and_split_data(
    config: Optional[ModelConfig] = None,
    *,
    data_path: Optional[str] = None,    
    shap_path: Optional[str] = None,
    id_col: Optional[str] = None,
    target_col: Optional[Union[List[str], str]] = None,
    random_state: Optional[int] = None,
    rawtest_datapath: Optional[str] = None,
    testdata_originalpred: Optional[str] = None,
    total_testdata_path: Optional[str] = None,
    test_size: Optional[float] = None,
    stratify_key = None,
    champion_model_name: Optional[str] = None
):
    """
    Load and split data for model training with improved readability.
    
    Parameters:
    -----------
    config : ModelConfig, optional
        ModelConfig object containing all configuration (preferred method).
        If provided, individual parameters are ignored.
    data_path : str, optional
        Path to the main data CSV file (required if config is None)
    shap_path : str, optional
        Path to the SHAP values Excel file (required if config is None)
    id_col : str, optional
        Name of the ID column (required if config is None)
    target_col : str or list, optional
        Name(s) of the target column(s) (required if config is None)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    rawtest_datapath : str, optional
        Path to save the raw test data
    testdata_originalpred : str, optional
        Path to save the test data with original predictions
    total_testdata_path : str, optional
        Path to save the complete test dataset
    test_size : float, optional
        Proportion of data to use for testing (default: 0.2)
    stratify_key : str, list, or None
        Column name(s) to use for stratification, or None to auto-generate
    
    Returns:
    --------
    tuple
        X_with_ids, X_train_full, y_train_full, X_test_final, y_test_final, 
        numerical_cols, categorical_cols, feature_ranking, stratify_key
    """
    # Extract values from config if provided, otherwise use individual parameters
    if config is not None:
        data_path = config.processesed_datapath_for_shap
        shap_path = config.shap_path
        id_col = config.id_col
        target_col = config.target_col
        random_state = config.random_state
        test_size = config.test_size
        rawtest_datapath = config.rawtest_datapath
        testdata_originalpred = config.testdata_originalpred
        total_testdata_path = config.total_testdata_path
    else:
        # Validate required parameters
        if data_path is None or shap_path is None or id_col is None or target_col is None:
            raise ValueError("Either 'config' must be provided, or all of 'data_path', 'shap_path', 'id_col', and 'target_col' must be provided.")
        if random_state is None:
            random_state = 42
        if test_size is None:
            test_size = 0.2
            
    # 1. Retrieve the champion name from the metadata file
    champion_model_name = get_champion_prefix_from_metadata(config) 

    if champion_model_name:  
        print(f"Dynamically loading champion_model_name: {champion_model_name}")
        # 2. Use the name to construct the path
        shap_path = shap_path.replace(".csv", f"_{champion_model_name}.csv")
        print(shap_path)
    else:
        print(f"Champion name not found in metadata. Attempting to load default path: {shap_path}")  
    
    main_data = read_dataset(data_path)
    feature_importance_data = read_dataset(shap_path)
      
    # Step 2: Organize columns based on SHAP feature importance
    ordered_features = feature_importance_data["Feature"].to_list()
    
    expected_features = ordered_features
    
    # The features we have in the loaded data
    available_columns = set(main_data.columns)

    # Find the features that are in the expected list but NOT in the available columns
    missing_features = [col for col in expected_features if col not in available_columns]

    if missing_features:
        print("\n" + "="*80)
        print("⚠️ WARNING: Feature Mismatch Detected!")
        print(f"The following {len(missing_features)} features are in the expected list ('ordered_features'/'feature_ranking') but are MISSING in the loaded data (main_data.columns):")
        for feature in missing_features:
            print(f"    - {feature}")
        print("\nACTION: These features will be **DROPPED** from the feature list to allow the pipeline to run.")
        print("="*80 + "\n")
    
    # Step 2: Update the feature list by dropping the missing ones
    # Filter the expected features to keep only the available ones
    feature_ranking = [col for col in expected_features if col in available_columns]
    ordered_features = feature_ranking
    if isinstance(target_col, list):  # Multi-output case
        column_order = [id_col] + target_col + ordered_features
    else:  # Single-output case
        column_order = [id_col, target_col] + ordered_features
    
    organized_data = main_data[column_order]
    
    # Step 3: Split data into features (X) and target (y)
    X_all = organized_data.drop(columns=target_col) 
    y_all = organized_data[target_col]
    X_features = X_all.drop(columns=[id_col])
    X_ids = X_all[id_col]
    
    # Step 4: Create stratification key combining target and country
    # This ensures balanced distribution across both target values and countries
    # old version: stratify_key = y_all.astype(str) + '_' + X_features['country'].astype(str)
    if stratify_key is None:
        # No stratify_key passed; dynamically generate it
        if isinstance(target_col, list):  # Multi-output case
            stratify_key = y_all[target_col[0]].astype(str) + '_' + X_features['country'].astype(str)
        else:  # Single-output case
            stratify_key = y_all.astype(str) + '_' + X_features['country'].astype(str)
    elif isinstance(stratify_key, str):  # Single column for stratification
        stratify_key = main_data[stratify_key].astype(str)
    elif isinstance(stratify_key, list):  # Multiple columns for stratification
        stratify_key = main_data[stratify_key].astype(str).agg('_'.join, axis=1)
    else:
        raise ValueError("`stratify_key` must be a string (single column) or a list of column names.")
        
    # Step 5: Perform train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, 
        y_all, 
        test_size=test_size,
        stratify=stratify_key, 
        random_state=random_state
    )
    print("features",X_features.columns)
    
    # Step 6: Preserve IDs for the test set
    X_with_all_ids = X_all.copy()
    X_test[id_col] = X_with_all_ids.loc[X_test.index, id_col]
    
    # Step 7: Save test data to CSV
    test_data = X_test.copy()
    test_data = test_data.reset_index(drop=True)
    test_data.to_csv(rawtest_datapath,index=False)
    
    print("y_test",rawtest_datapath)
    # Step 8: Create and save test data with original predictions
    if isinstance(y_test, pd.Series):
        test_predictions = y_test.to_frame(name='Original_pred')
    elif isinstance(y_test, pd.DataFrame):
        test_predictions = y_test.copy()
    else:
         raise TypeError(f"y_test is neither a Series nor a DataFrame. Type: {type(y_test)}")
    test_predictions[id_col] = X_with_all_ids.loc[X_test.index, id_col]
#    test_predictions.to_csv(testdata_originalpred, index=False)
    
    # Step 9: Identify numerical and categorical columns for preprocessing
    numerical_columns = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    feature_ranking = X_train.columns.tolist()
    
    # Step 10: Create and save combined test dataset
    complete_test_data = pd.merge(test_data, test_predictions, on=id_col)
    complete_test_data.to_csv(total_testdata_path, index=False)
    
    # Return all required data for model training and evaluation
    return (
        X_with_all_ids, 
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        numerical_columns, 
        categorical_columns, 
        feature_ranking,
        stratify_key
    )
  
  
def convert_data_for_weighted_training(X_train: pd.DataFrame, y_train: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert data for weighted training, creating arrays for train_data, train_labels, and train_weights.

    Args:
        X_train (pd.DataFrame): Features for training.
        y_train (pd.DataFrame): Target probabilities for training.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - train_data: Expanded feature data for weighted training.
            - train_labels: Corresponding labels for each observation.
            - train_weights: Weights for each observation.
    """
    train_data = []
    train_labels = []
    train_weights = []

    for i in range(y_train.shape[0]):
        for class_idx, prob in enumerate(y_train.iloc[i]):
            # Append row as a pandas Series
            train_data.append(X_train.iloc[i])
            train_labels.append(class_idx)
            train_weights.append(prob)

    # Convert the lists to pandas DataFrame and Series
    train_data = pd.DataFrame(train_data)  # Keep as DataFrame
    train_labels = pd.Series(train_labels, name="Labels")  # Convert to Series
    train_weights = pd.Series(train_weights, name="Weights")  # Convert to Series

    return train_data, train_labels, train_weights


#from src.adoption_cohort import config as cohort_config
#from src.adoption import config as inference_config
#from src.common.config import ModelConfig
#config = ModelConfig.from_module(cohort_config)
#config.shap_path
#X_with_ids, X_train_full, y_train_full, X_test_final, y_test_final, numerical_cols, categorical_cols, feature_ranking, stratify_key = load_and_split_data(config=config,stratify_key=config.stratify_key )
#main_data = read_dataset(config.processesed_datapath_for_shap)
#cntry_cols = main_data.filter(like="Country_")
#cntry_cols.head() 
#main_data.columns
#feature_importance_data = read_dataset(config.shap_path)
#ordered_features = feature_importance_data["Feature"].to_list()
#ordered_features
