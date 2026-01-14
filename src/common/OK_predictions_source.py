import pandas as pd
import numpy as np
from typing import Optional, List
import os
import cloudpickle
import sys

from src.common.utils import setup_project_paths
project_root = setup_project_paths()
sys.path.insert(0, os.path.abspath(os.path.join(project_root, "src")))


#from src.common.config import ModelConfig
#from src.digital_cohort import config as cohort_config
#from src.digital import config as inference_config


def cohort_preprocess_data(
    config,
    df: pd.DataFrame,
    country_column: str = "Country",
    new_column_name: str = "Country_clean",
    min_obs: int = 10,
    id_columns: Optional[List[str]] = None
) -> tuple:
    """
    Preprocess the data for cohort modeling.
    """
    random_state = config.random_state  
    target_col = config.target_col
    print("TARGET_COL",target_col)
      
    #Determine Model Type and Optimization Direction
    is_softweights_cohort = isinstance(target_col, list)  
    print(is_softweights_cohort)

      
    # Step 1: Drop unnecessary columns
    df = df.loc[:, ~df.columns.str.contains(" Count")]
    df = df.loc[:, ~df.columns.str.contains("Count_")]

    # Step 2: Create the modified country column
    country_counts = df[country_column].value_counts()
    df[new_column_name] = df[country_column].apply(lambda x: x if country_counts[x] >= min_obs else "Other")

    # Step 3: Define features (X) and targets (y)
    if not target_col:
        raise ValueError("Target columns must be specified in config.target_col.")

    y = df[target_col]
    X = df.drop(columns= target_col + [new_column_name]) 

    # Normalize probabilities for target columns
    y = y.div(y.sum(axis=1), axis=0)

    # Perform one-hot encoding on relevant columns
    cols_to_encode = [col for col in ['Specialty (CDV2)', country_column] if col in X.columns]
    X_encoded = pd.get_dummies(X, columns=cols_to_encode, drop_first=True)

    # Step 4: Rename X columns
    X_encoded.columns = X_encoded.columns.str.replace(r"[^\w\d_]+", "_", regex=True)
    X_encoded.columns = X_encoded.columns.str.replace(r"_+", "_", regex=True)

    # Step 5: Add ID column and create processed_data DataFrame
    country_clean_series = df[new_column_name]

    if id_columns:
        if not all(col in df.columns for col in id_columns):
            raise ValueError(f"One or more columns in `id_columns` are not present in the DataFrame: {id_columns}")
        df["ID"] = df[id_columns].astype(str).agg("_".join, axis=1)
        processed_data = pd.concat([df["ID"].reset_index(drop=True), country_clean_series.reset_index(drop=True), y.reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1)
    else:
        processed_data = pd.concat([country_clean_series.reset_index(drop=True), y.reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1)
    print("X_encoded",X_encoded.shape)
    
    Country_Specialty = pd.DataFrame(df[['Specialty (CDV2)', 'Country']])    
    return X_encoded, y, country_clean_series, processed_data, Country_Specialty 


def inference_preprocess_data(config,df: pd.DataFrame):
        print("Ok dataloaded for Inference module")
        df.rename(columns={ 'Specialty_V2': 'OK_Spec_Map_CD_Spec',
                    #'accepts_remote_detailing_by_web_conference_workplace': 'accepts_remote_detailing_by_webconf_workplace',
                    'onekey_database_id':'onekey id'}, inplace=True)
        return df

    

  
def cohort_predictions_ok_data(config, ok_data: pd.DataFrame,Country_Specialty:pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess ok_data and predict probabilities, handling differences in profile names,
    and including the predicted hard label and the 'ID' column for distribution analysis.
    
    Returns:
        pd.DataFrame: Predicted probabilities, the predicted hard label ('predictions'),
                      and the 'ID' column for the new dataset.
    """
    expected_profile_names = config.target_col
    model_path = config.cloudpickle_filename_step3
    feature_names_path = config.metrics_filename_step3
    sheet_name = 'Final_Features'
    
    # --- Feature Loading and Model Loading (Simplified for brevity) ---
    try:
        # ... (Model and Feature Loading logic, unchanged) ...
        # 
        with open(model_path, "rb") as f:
            model = cloudpickle.load(f)
            
        features_df = pd.read_excel(feature_names_path, sheet_name=sheet_name)
        feature_names = features_df.iloc[:, 0].astype(str).tolist()
        valid_feature_names = [f for f in feature_names if f in ok_data.columns]

        if not valid_feature_names:
             raise ValueError("No valid feature names found in the data after alignment.")
             
        ok_model_features = ok_data[valid_feature_names]
        
    except Exception as e:
        print(f"  > **ERROR:** Feature loading failed: {e}")
        raise AttributeError("FATAL: Could not determine feature names.")
        
    # --- Prediction ---
    print("\nPredicting probabilities on ok_data...")
    predicted_probabilities_raw = model.predict_proba(ok_model_features)

    if len(expected_profile_names) != predicted_probabilities_raw.shape[1]:
        raise ValueError("Number of expected profile names does not match the number of model output profiles.")

    predicted_probabilities = pd.DataFrame(
        predicted_probabilities_raw,
        columns=expected_profile_names,
        index=ok_data.index
    )    
    

    # --- New Logic to find the correct column name ---
    if not Country_Specialty.empty:
        # Concatenate the 'ID' column (whatever its name) from the input data with the prediction results
        print(f"  > Using column '{Country_Specialty.columns}' to have Countrywise distributions.")

        # We must ensure both DataFrames have matching indices for concatenation
        country_data_reset = Country_Specialty.reset_index(drop=True)
        predicted_probabilities_reset = predicted_probabilities.reset_index(drop=True)

        ok_prob_final = pd.concat([country_data_reset, predicted_probabilities_reset], axis=1)
        print("predicted_probabilities",predicted_probabilities.shape)
        print("ok_prob_final",ok_prob_final.shape)
        print(ok_prob_final.head())

    else:
        # If no identifying column is found, country-wise distribution cannot be performed.
        print(f"  > WARNING: None of the expected columns ({', '.join(POSSIBLE_ID_COLUMNS)}) found in input data.")
        print("  > Cannot calculate country distribution.")
        ok_prob_final = predicted_probabilities
    
    return ok_prob_final
    
  
def inference_predictions_ok_data(config,ok_data: pd.DataFrame): 
    
    model_path = config.cloudpickle_filename_step3
    target_col = config.target_col
    print(f"  > Loading model from: {model_path}")
    with open(model_path, "rb") as f:
          model = cloudpickle.load(f)
        
    best_selected_features = []
    
    if hasattr(model, 'feature_names_in_'):
        best_selected_features = list(model.feature_names_in_)
        print(f"Features from model: {best_selected_features}")
    else:
        print("Warning: Could not get feature names from the model. Skipping prediction.")
        return ok_data, []
    
    missing_features = [feat for feat in best_selected_features if feat not in ok_data.columns]
    
    if missing_features:
        print(f"Error: Missing features in input DataFrame: {missing_features}")
        return ok_data, []

    data_for_prediction = ok_data[best_selected_features]
    predictions_array = model.predict(data_for_prediction)
    
    predictions_df = pd.DataFrame(predictions_array, columns=['predictions'], index=ok_data.index)
    
    # Merge predictions back to the main DataFrame
    ok_data['predictions'] = predictions_df['predictions']
    
    print("\nPredicted classes for new data:")
    print(ok_data[['onekey id', 'predictions']].head())
    
    return ok_data, best_selected_features
  
  
def inference_calculate_distribution(config, df: pd.DataFrame, data_type: str):
    """
    Calculates overall and country-wise prediction distributions for a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing 'predictions' and 'country' columns.
        data_type (str): Label for printing and sheet naming (e.g., 'OneKey', 'Test Data').

    Returns:
        tuple: (overall_distribution_df, country_distribution_df)
    """
    if df.empty or 'predictions' not in df.columns:
        print(f"Error: {data_type} DataFrame is empty or missing 'predictions' column.")
        return pd.DataFrame(), pd.DataFrame()

    # Calculate overall prediction distribution
    overall_distribution = df['predictions'].value_counts(normalize=True) * 100
    overall_distribution_df = overall_distribution.reset_index(name='percentage').rename(columns={'index': 'prediction'})
    
    print(f"\nOverall Prediction Distribution ({data_type}):")
    print(overall_distribution_df)
    
    if 'ID' in df.columns:
            # Use .loc to avoid SettingWithCopyWarning
            df.loc[:, 'country'] = df['ID'].str.split('_', n=1, expand=True)[1]

    # Calculate country-wise prediction distribution
    if 'country' in df.columns:
        print(f"\nCalculating prediction distribution by country ({data_type})...")
        
        country_counts = df.groupby('country').size().rename('Total Count')
        country_predictions = df.groupby(['country', 'predictions']).size().unstack(fill_value=0)
        
        country_distribution_df = country_predictions.div(country_counts, axis=0) * 100
        country_distribution_df = country_distribution_df.reset_index()
        
        print(f"\nCountry-wise Prediction Distribution ({data_type}):")
        print(country_distribution_df.head())
    else:
        print(f"Warning: 'country' column not found in {data_type} DataFrame. Skipping country-wise distribution.")
        country_distribution_df = pd.DataFrame()

    return overall_distribution_df, country_distribution_df

    
def cohort_calculate_distribution(df: pd.DataFrame, data_type: str):
    """
    1. Extracts Country from ID if necessary.
    2. Cleans 'y_pred_' prefix from columns.
    3. Returns Overall Dist (Vertical/Long format) and Country Dist (Wide format).
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # --- Step 1: Handle Country Column ---
    # If Country is missing (common in Test Data), extract from ID
    if 'Country' not in df.columns and 'ID' in df.columns:
        # Splits 'ProviderID_Algeria' -> 'Algeria'
        df['Country'] = df['ID'].str.split('_', n=1, expand=True)[1]

    # --- Step 2: Clean Column Names ---
    # Identify and rename columns starting with 'y_pred_' to empty strings
    rename_map = {col: col.replace('y_pred_', '') for col in df.columns if 'y_pred_' in col}
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    
    # Identify probability columns (exclude metadata)
    # If we renamed columns, those are our targets; otherwise, filter standard metadata
    prob_cols = list(rename_map.values()) if rename_map else \
                [col for col in df.columns if col not in ['Specialty (CDV2)', 'Country', 'ID', 'Country_clean']]

    # --- Step 3: Calculate Distributions ---
    # A. Country-wise Distribution (Wide format for Excel)
    country_dist_df = df.groupby('Country')[prob_cols].mean().reset_index()

    # B. Overall Distribution (Melted to Vertical format as requested)
    # Calculate mean -> Transpose -> Reset Index to get labels as rows
    overall_mean = country_dist_df[prob_cols].mean()
    overall_distribution_df = overall_mean.to_frame().reset_index()
    
    # Rename columns to your specific requirements
    overall_distribution_df.columns = ['Adoption Profile', 'Distribution%']
    
    # Optional: Convert 0.3112 to 31.12 for percentage display
    overall_distribution_df['Distribution%'] = overall_distribution_df['Distribution%'] * 100

    print(f"✅ Processed {data_type} distribution.")
    return overall_distribution_df, country_dist_df    