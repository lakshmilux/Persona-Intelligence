import os
import sys
import pandas as pd
import numpy as np
import cloudpickle
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer 
from sklearn.metrics import make_scorer,get_scorer
from sklearn.preprocessing import LabelEncoder
import time
from typing import Dict, Any, List, Union
import warnings
# --- Setup and Imports ---

from src.common.utils import setup_project_paths
setup_project_paths()
sys.path.insert(0, os.path.abspath(os.path.join(setup_project_paths(), "src")))

from src.digital_cohort import config as cohort_config
from src.digital import config as inference_config
from src.common.config import ModelConfig
from src.common.metrics import geometric_mean_f1_scorer
from typing import Union
import numpy as np

def custom_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Custom implementation of log loss for probability distributions."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    log_loss_per_sample = -np.sum(y_true * np.log(y_pred), axis=1)
    return np.mean(log_loss_per_sample)

gmean_f1_scorer_obj = make_scorer(geometric_mean_f1_scorer, greater_is_better=True)
custom_log_loss_obj = make_scorer(custom_log_loss, greater_is_better=False)#, response_method='predict_proba') 
accuracy_scorer_obj = get_scorer('accuracy')

# Define the dictionary of scorers to use, ensuring consistency
scorers: Dict[str, Any] = {
    'Accuracy': accuracy_scorer_obj,
    'GeometricMean': gmean_f1_scorer_obj,
    'Custom_Log_Loss': custom_log_loss_obj
}
scorer_keys = list(scorers.keys())

# Define common parameters for permutation_importance
perm_imp_params: Dict[str, Any] = {
    'n_repeats': 5,
    'n_jobs': -1
}

# --- Permutation Importance Functions ---

def get_overall_permutation_importance(
    estimator: Any, 
    X_test: pd.DataFrame, 
    y_test: Union[pd.Series, pd.DataFrame], 
    scorers: Dict[str, Any],
    perm_imp_params: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    
    print(f"\n--- Starting Overall Permutation Importance Calculation ---")
    overall_importance_results: Dict[str, pd.DataFrame] = {}
    start_time = time.time()

    is_softweights_cohort = isinstance(y_test, pd.DataFrame) and y_test.shape[1] > 1
    print("soft", is_softweights_cohort)
    
    # Filter scorers based on the model type
    if is_softweights_cohort:
        print("Model Type Detected: **Soft Weights Cohort** (using Multi-Target Scorers)")
        scorers_to_use = {k: v for k, v in scorers.items() if k == 'Custom_Log_Loss'}
    else:
        print("Model Type Detected: **Standard Classification** (using Hard Label Scorers)")
        scorers_to_use = {k: v for k, v in scorers.items() if k in ['Accuracy', 'GeometricMean']}
        
    if not scorers_to_use:
        print("WARNING: No appropriate scorers found for the detected model type. Skipping permutation importance.")
        return overall_importance_results

    # Continue passing DataFrames/Series objects directly
    for metric_name, scorer_func in scorers_to_use.items(): 
        print(f"  > Calculating for **{metric_name}**...")
        try:
            overall_perm_importance = permutation_importance(
                estimator, 
                X_test, 
                y_test, 
                scoring=scorer_func,
                **perm_imp_params
            )

            df = pd.DataFrame({
                "Feature": X_test.columns, 
                "Permutation Importance Mean": overall_perm_importance.importances_mean,
                "Permutation Importance Std": overall_perm_importance.importances_std
            }).sort_values(by="Permutation Importance Mean", ascending=False)

            overall_importance_results[metric_name] = df
            print(f"  - Overall Permutation Importance for {metric_name} calculated successfully.")

        except Exception as e:
            print(f"  - **ERROR:** Could not compute overall permutation importance for {metric_name}: {e}")
            overall_importance_results[metric_name] = pd.DataFrame()

    elapsed_time = time.time() - start_time
    print(f"\nOverall importance calculation took **{elapsed_time:.2f} seconds**.")
    return overall_importance_results
    
def get_countrywise_permutation_importance(
    model: Any, 
    data: pd.DataFrame, 
    feature_names: List[str], 
    target_col: Union[str, List[str]],
    scorers: Dict[str, Any],
    perm_imp_params: Dict[str, Any]
) -> Dict[str, List[pd.DataFrame]]:
    """
    Calculates country-wise permutation importance for all specified scorers.
    """
    print(f"\n--- Starting Country-Wise Permutation Importance Calculation ---")
    country_importance_results: Dict[str, List[pd.DataFrame]] = {metric_name: [] for metric_name in scorers.keys()}
    start_time = time.time()
    label_encoder = LabelEncoder()
    
    is_softweights_cohort = isinstance(target_col, list)
    print("soft", is_softweights_cohort)
    
    # Filter scorers based on the model type
    if is_softweights_cohort:
        print("Model Type Detected: **Soft Weights Cohort** (using Multi-Target Scorers)")
        scorers_to_use = {k: v for k, v in scorers.items() if k == 'Custom_Log_Loss'}
    else:
        print("Model Type Detected: **Standard Classification** (using Hard Label Scorers)")
        scorers_to_use = {k: v for k, v in scorers.items() if k in ['Accuracy', 'GeometricMean']}
        
    if not scorers_to_use:
        print("WARNING: No appropriate scorers found for the detected model type. Skipping permutation importance.")
        return country_importance_results

    # Continue passing DataFrames/Series objects directly
    for metric_name, scorer_func in scorers_to_use.items(): 
        
        print(f"\n  > Calculating Country-Wise Importance for **{metric_name}**...")
        
        for country, group_data in data.groupby('country'):
            print(f"    >> Computing importance for **{country}**...")
            
            y_country = group_data[target_col]
            X_country_model_features = group_data[feature_names]
            
            # Initialize debug variables
            country_variance_check = np.nan
            country_baseline_score = np.nan
            country_prediction_std_sum = np.nan
            
            if is_softweights_cohort:
                has_variation = y_country.std().sum() > 1e-6 
                is_valid_target_variation = has_variation  
            else:
                is_valid_target_variation = len(y_country.unique()) >= 2
                
            if X_country_model_features.empty or not is_valid_target_variation:
                print(f"    - Skipping {country}: Data is empty or target column has < 2 unique classes.")
                continue
            
            
            # 1. Check Feature Variance
            numeric_X = X_country_model_features.select_dtypes(include=np.number)
            if not numeric_X.empty:
                country_variance_check = numeric_X.std().sum()

            # 2. Check Baseline Score
            try:
                if metric_name == 'Custom_Log_Loss':
                    # Log Loss: use model.predict_proba and original y_country (assumed multi-column/one-hot)
                    predictions = model.predict_proba(X_country_model_features)
                    y_score_input = y_country.values
                    predictions_for_std = predictions # For variance calculation
                    
                else:
                    # Hard-label metrics (Accuracy, GeometricMean): requires consistent encoding for baseline
                    predictions_raw = model.predict(X_country_model_features)
                    
                    if predictions_raw.ndim > 1:
                        predictions_raw = predictions_raw.ravel()
                    
                    # --- FIX: Ensure true labels and predictions are encoded consistently ---
                    
                    # 1. Encode true labels (y_country) and use this fit for the predictions
                    # We refit/transform y_country for two reasons:
                    # a) To get the numerical target (y_score_input)
                    # b) To learn the mapping to correctly encode the predictions
                    y_score_input = label_encoder.fit_transform(y_country.values.ravel())
                    
                    # 2. Transform the raw predictions using the learned mapping
                    predictions_encoded = label_encoder.transform(predictions_raw)
                    
                    # 3. Use the encoded arrays for baseline calculation and variance
                    predictions = predictions_encoded.astype(int)
                    y_score_input = y_score_input.astype(int)
                    predictions_for_std = predictions # For variance calculation

                # Calculate the baseline score using the consistently encoded numerical arrays
                country_baseline_score = scorer_func._score_func(y_score_input, predictions)
                print(f"    - DEBUG: Baseline Score: **{country_baseline_score:.6f}**")
                
                # Calculate prediction std sum
                if predictions_for_std.ndim == 1:
                    country_prediction_std_sum = predictions_for_std.std()
                else:
                    country_prediction_std_sum = predictions_for_std.std(axis=0).sum()
                
                print(f"    - DEBUG: Prediction STD Sum: **{country_prediction_std_sum:.6f}**")
                
            except Exception as e:
                print(f"    - WARNING: Could not calculate baseline score: {e}")

            
            try:
                # Permutation Importance uses the scorer object, which handles the necessary encoding/predict calls internally
                country_perm_importance = permutation_importance(
                    model,
                    X_country_model_features,
                    y_country, # Pass original y_country (Pandas object) to permutation_importance
                    scoring=scorer_func,
                    **perm_imp_params
                )
                
                num_features = len(X_country_model_features.columns)
                
                df = pd.DataFrame({
                    "Country": country,
                    "Feature": X_country_model_features.columns,
                    "Permutation Importance Mean": country_perm_importance.importances_mean, # Using built-in mean
                    "Permutation Importance Std": country_perm_importance.importances_std,
                    f"{metric_name}_Baseline_Score": [country_baseline_score] * num_features,
                    "Feature_Variance": [country_variance_check] * num_features,
                    "Predictions_variance":[country_prediction_std_sum]*num_features
                })
                
                country_importance_results[metric_name].append(df)
                print(f"    - Importance for {country} calculated successfully.")
            
            except Exception as e:
                print(f"    - **ERROR:** Could not compute importance for {country} with {metric_name}: {e}")
                continue
                
    elapsed_time = time.time() - start_time
    print(f"\nCountry-wise importance calculation took **{elapsed_time:.2f} seconds**.")
    return country_importance_results

# --- The evaluate_model_importance function is the same as provided by the user ---

def evaluate_model_importance(config):
    """
    Loads a trained model and a test dataset, then orchestrates the calculation and saving
    of detailed permutation importance for all specified metrics, both overall
    and country-wise.
    """
    print(f"\n--- Starting model importance evaluation ---")

    # Load and Destructure Configuration ⚙️
    model_config = ModelConfig.from_module(config)
    is_softweights_cohort = isinstance(model_config.target_col, list)

    # Centralized variable definition for paths and columns
    model_path = model_config.cloudpickle_filename_step3
    data_path = model_config.processesed_datapath_for_shap
    output_dir = model_config.countrywise_perm_importance_dir
    target_col = model_config.target_col
    feature_names_path = model_config.metrics_filename_step3
    output_filename = model_config.countrywise_perm_importance_filename

    # Initialize variables to ensure they exist after try/except
    X_test_model_features = pd.DataFrame()
    y_test = pd.Series()
    feature_names = []
    data = pd.DataFrame()
    country_importance_results = {}
    full_pipeline = None 

    # --- Step 1: Load the model and data ---
    try:
        print(f"  > Loading model from: {model_path}")
        with open(model_path, "rb") as f:
            model = cloudpickle.load(f)
        
        data = pd.read_csv(data_path)
        
        # Ensure the target column name is correct based on config
        if not is_softweights_cohort and isinstance(target_col, str):
             data.rename({'Original_pred': target_col}, axis=1, inplace=True, errors='ignore')
        
        data.rename({'Country':'country'}, axis=1, inplace=True)

        # Country ID extraction logic
        if 'ID' in data.columns:
            # Use .loc to avoid SettingWithCopyWarning
            data.loc[:, 'country'] = data['ID'].str.split('_', n=1, expand=True)[1]
            
        # Check for required columns
        if 'country' not in data.columns:
            raise ValueError("The 'country' column is missing from the data.")

        # Determine columns to drop
        if is_softweights_cohort:
            y_test = data[target_col]
            cols_to_drop = [c for c in target_col if c in data.columns] + ['ID', 'country']
            X_test = data.drop(columns=cols_to_drop, errors='ignore')
        else:
            y_test = data[target_col]
            X_test = data.drop(columns=[target_col])

        if is_softweights_cohort:
            sheet_name = 'Final_Features'
            try:
                print(f"  > Attempting to load raw feature names from Excel file: {feature_names_path}")
                features_df = pd.read_excel(feature_names_path, sheet_name=sheet_name)
                feature_names = features_df.iloc[:, 0].astype(str).tolist()
                print(f"  > Successfully loaded **{len(feature_names)}** feature names from the '{sheet_name}' sheet.")
                valid_feature_names = [f for f in feature_names if f in X_test.columns]
                missing_in_data = set(feature_names) - set(valid_feature_names)
                if missing_in_data:
                    print(f"  > WARNING: {len(missing_in_data)} names loaded from Excel are NOT in X_test and will be ignored.")

                if not valid_feature_names:
                    raise ValueError("No valid feature names found in the data after alignment.")

                X_test_model_features = X_test[valid_feature_names]
                feature_names = valid_feature_names
                expected_features = feature_names  
                print(f"  > X_test aligned successfully with **{len(X_test_model_features.columns)}** features.")
            except Exception as e:
                print(f"  > **ERROR:** Manual feature loading failed: {e}")
                raise AttributeError("FATAL: Could not determine feature names, even by manual Excel lookup.")
        else:
            if isinstance(model, Pipeline):
                # Try to get feature names after preprocessing, or fall back to all columns
                try:
                    feature_names = model.named_steps.get('pre', model).get_feature_names_out().tolist()
                except AttributeError:
                    feature_names = X_test.columns.tolist() # Fallback
            elif hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_.tolist()
            else:
                raise AttributeError("Could not determine feature names from the model.")
                
            valid_feature_names = [f for f in feature_names if f in X_test.columns]
            if not valid_feature_names:
                raise ValueError("No valid features found after aligning hard-label model names with data.")

        X_test_model_features = X_test[valid_feature_names]
        feature_names = valid_feature_names    
        print("modelfeatures",X_test_model_features.head())    
        print(f"Model and test data loaded successfully with **{len(feature_names)} features**.")
            
    except Exception as e:
        print(f"**FATAL ERROR** loading model or data: {e}. Aborting.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Step 2: Run Overall Permutation Importance ---
    overall_importance_results = get_overall_permutation_importance(
        model, 
        X_test_model_features, 
        y_test, 
        scorers, 
        perm_imp_params
    )  
    
    country_importance_results = get_countrywise_permutation_importance(
        model, 
        data, 
        feature_names, 
        target_col,
        scorers, 
        perm_imp_params
    )
    
    # --- Step 4: Save all results to a single Excel file with multiple sheets ---
    print("\n--- Saving all permutation importance results to Excel ---")
    output_file = os.path.join(output_dir, output_filename)
    
    # Check if there is any data to save
    has_overall_data = any(df is not None and not df.empty for df in overall_importance_results.values())
    has_country_data = any(df_list for df_list in country_importance_results.values())
    
    if not has_overall_data and not has_country_data:
        print("No results to save. Aborting Excel creation.")
        return

    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Save overall importance
            for metric_name, df in overall_importance_results.items():
                if df is not None and not df.empty:
                    sheet_name = f"Overall_{metric_name}"
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"  - Overall importance for {metric_name} saved to **'{sheet_name}'** sheet.")

            # Save country-wise importance
            for metric_name, df_list in country_importance_results.items():
                if df_list:
                    # Concatenate all country results for this metric
                    all_country_df = pd.concat(df_list, ignore_index=True)
                    all_country_df.sort_values(
                        by=['Country', 'Permutation Importance Mean'], 
                        ascending=[True, False], 
                        inplace=True
                    )
                    sheet_name = f"Countrywise_{metric_name}"
                    all_country_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"  - Country-wise importance for {metric_name} saved to **'{sheet_name}'** sheet.")
                else:
                    print(f"  - No country-wise importance results to save for {metric_name}.")
                    
        print(f"\n**SUCCESS:** All permutation importance reports saved to **{output_file}**")
        
    except Exception as e:
        print(f"**FATAL ERROR** saving data to Excel file: {e}")
        
# --- Main execution block ---
if __name__ == "__main__":
  
    warnings.filterwarnings(
        "ignore", 
        message="X does not have valid feature names, but LGBMClassifier was fitted with feature names", 
        category=UserWarning,
        module='sklearn'
    )
    config = cohort_config
    evaluate_model_importance(config)
    print("\n--- COHORT Permutation Importance completed. All reports generated. ---")
   
    config = inference_config
    evaluate_model_importance(config)
    print("\n--- Inference Permutation Importance completed. All reports generated. ---")