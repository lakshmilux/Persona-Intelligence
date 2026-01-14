#!pip install pandas openpyxl shap seaborn scikit-learn imbalanced-learn catboost lightgbm xgboost optuna optuna-integration[lightgbm,xgboost,catboost] scipy category-encoders --upgrade --no-cache-dir
#!pip install numpy --upgrade
#pip install 'numpy<2'
#pip install shap==0.48.0 --force-reinstall
#pip install xgboost==3.0.5 --force-reinstall
import pandas as pd
import numpy as np
import seaborn as sns
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from imblearn.ensemble import BalancedRandomForestClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
import shap
import xgboost as xgb
from sklearn.inspection import permutation_importance
from typing import Tuple, Dict, Optional, List, Union
import optuna
from optuna.integration import LightGBMPruningCallback
import sys
import os
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder

from src.common.utils import setup_project_paths
project_root = setup_project_paths()
sys.path.insert(0, os.path.abspath(os.path.join(project_root, "src")))


from src.common.model_factory_enhanced import EnhancedModelFactory
from src.common.new_optunaenhanced import (OptunaObjectiveEnhanced, optimize_lgb_inference, optimize_brf_inference, 
                                           optimize_catboost_inference,optimize_xgboost_inference,
                                           optimize_xgboost_cohort,optimize_lgb_cohort, 
                                           optimize_brf_cohort, optimize_catboost_cohort)
from src.common.metrics import calculate_fold_metrics, evaluate_models, generate_results_table
from src.common.data_preprocessing import convert_data_for_weighted_training


# Import config module and create ModelConfig
from src.common.config import ModelConfig
from src.digital_cohort import config as cohort_config
from src.digital import config as inference_config
inference_model_config = ModelConfig.from_module(inference_config)
cohort_model_config = ModelConfig.from_module(cohort_config)

# ======================================================================
## ✂️ Cohort-Specific Functions
# ======================================================================

def preprocess_data(
    config: ModelConfig,
    df: pd.DataFrame,
    country_column: str = "Country",
    new_column_name: str = "Country_clean",
    min_obs: int = 10,
    id_columns: Optional[List[str]] = None
) -> tuple:
    """
    Preprocess the data for cohort modeling.
    """
    target_columns = config.target_col
    
    if isinstance(target_columns, str):
        target_columns = [target_columns]

    # Step 1: Drop unnecessary columns
    df = df.loc[:, ~df.columns.str.contains(r" Coverage \(Base: Count_UserID\)", regex=True)]
    df.columns = df.columns.str.replace(" (Base: Count_OneKey ID)", "", regex=False)
    df = df.loc[:, ~df.columns.str.contains(" Count")]
    df = df.loc[:, ~df.columns.str.contains("Count_")]
        
    # Step 2: Create the modified country column
    country_counts = df[country_column].value_counts()
    df[new_column_name] = df[country_column].apply(lambda x: x if country_counts[x] >= min_obs else "Other")

    # Step 3: Define features (X) and targets (y)
    if not target_columns:
        raise ValueError("Target columns must be specified in config.target_col.")
        
    y = df[target_columns]
    X = df.drop(columns=target_columns + [new_column_name]) 
    
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
        
    return X_encoded, y, country_clean_series, processed_data

def split_data(
    config: ModelConfig,
    X: pd.DataFrame,
    y: pd.DataFrame,
    country_column: pd.Series,
    val_size: float = 0.2,
) -> Tuple:
    """
    Split the data into train, validation, and test sets using stratified sampling.
    """
    
    test_size = config.test_size
    random_state = config.random_state

    # 1. First split: Train + Validation vs Test (Stratified by country_column)
    X_train_val, X_test, y_train_val, y_test, train_val_country, test_country = train_test_split(
        X, y, country_column, 
        test_size=test_size, 
        stratify=country_column,
        random_state=random_state,
    )

    # 2. Second split: Train vs Validation (Stratified by country_column)
    X_train, X_val, y_train, y_val, train_country, val_country = train_test_split(
        X_train_val, y_train_val, train_val_country, 
        test_size=val_size,
        stratify=train_val_country,
        random_state=random_state,
    )
    
    # Create a split indicator vector
    split_indicator = pd.Series(index=X.index, dtype="object")
    split_indicator.loc[X_train.index] = "train"
    split_indicator.loc[X_val.index] = "valid"
    split_indicator.loc[X_test.index] = "test"

    return X_train, X_val, X_test, y_train, y_val, y_test, train_country, val_country, test_country, split_indicator



def calculate_feature_importance_unified(
    model, 
    X_test: pd.DataFrame, 
    y_test: Union[pd.Series, np.ndarray], # Added y_test for Permutation Importance
    X_data_train_cols: pd.Index, # Added for aligning BRF features/Permutation
    findf: pd.DataFrame, # Added findf for Coverage calculation
    profile_names: list = None, 
    evaluation_type: str = "softweights" # New parameter to switch logic
) -> Union[Tuple[pd.DataFrame, List[pd.DataFrame]], Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]]:
    """
    Calculate and return feature importance based on evaluation type.

    Args:
        model: Trained model instance (LGBM, CatBoost, etc.).
        X_test (pd.DataFrame): Test features used for calculating importance.
        y_test (Union[pd.Series, np.ndarray]): True labels for Permutation Importance.
        X_data_train_cols (pd.Index): Column names from the training data (for BRF/alignment).
        findf (pd.DataFrame): The original processed DataFrame for Coverage calculation.
        profile_names (list): List of class/label names.
        evaluation_type (str): 'softweights' for detailed multi-class SHAP 
                               or 'inference' for Permutation + Weighted SHAP/Permutation score.
    """
    
    # Use the training columns for alignment during importance calculation
    X_data = X_test.reindex(columns=X_data_train_cols, fill_value=0) 
    
    # 1. SOFTWEIGHTS (Multi-class Detailed SHAP)
    if evaluation_type == "softweights":
        print("Calculating Softweights Feature Importance (Detailed SHAP)...")
        if profile_names is None:
            # Assume profile_names is derived from y_train unique classes or config
            profile_names = [f'Class {i}' for i in range(len(np.unique(y_test)))] 

        # Explainer initialization (using X_data which is aligned)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_data) 
        
        # --- Multi-class SHAP Pre-processing (from your original code) ---
        shap_per_observation = []
        if isinstance(shap_values, list):
            # Handles LGBM output (list of 2D arrays)
            if len(shap_values) == len(profile_names):
                shap_values_3d = np.stack(shap_values, axis=2) 
            else: 
                # Binary case
                shap_values_3d = np.stack(shap_values, axis=2)
                profile_names = [f'Class {i}' for i in range(shap_values_3d.shape[2])]
        elif shap_values.ndim == 3:
            # Handles CatBoost output (3D array)
            shap_values_3d = shap_values
        elif shap_values.ndim == 2:
            # Handles Binary case
            shap_values_3d = shap_values[..., np.newaxis]
            profile_names = [profile_names[0]] 

        # Process per-observation SHAP for all classes
        for class_idx in range(shap_values_3d.shape[2]):
            shap_per_observation.append(
                pd.DataFrame(
                    shap_values_3d[:, :, class_idx], 
                    columns=X_data.columns, 
                    index=X_data.index
                )
            )
        
        # Aggregate SHAP values across observations and classes
        shap_values_mean_per_profile = np.mean(np.abs(shap_values_3d), axis=0) 
        shap_values_mean_across_profiles = np.mean(shap_values_mean_per_profile, axis=1)

        # Compile SHAP values into a DataFrame
        feature_importance_data = {'Feature': X_data.columns}
        for i, profile in enumerate(profile_names):
            feature_importance_data[f'SHAP Importance ({profile})'] = shap_values_mean_per_profile[:, i]

        feature_importance_data['SHAP Importance (Mean Across Profiles)'] = shap_values_mean_across_profiles
        shap_summary = pd.DataFrame(feature_importance_data).sort_values(
            by='SHAP Importance (Mean Across Profiles)', ascending=False
        )

        return shap_summary, shap_per_observation, shap_values_3d
        
    # 2. INFERENCE 
    elif evaluation_type == "inference":
        print("Calculating Inference Feature Importance...")
        # Ensure model is compatible with permutation_importance (classifiers usually are)
        
        # --- Permutation Importance ---
        perm_importance = permutation_importance(
            model, 
            X_data, # Use aligned X_data
            y_test, 
            n_repeats=30, 
            random_state=42, 
            scoring="accuracy" # Requires hard labels (y_test)
        )
        perm_importance_df = pd.DataFrame({
            "Feature": X_data.columns,
            "Permutation Importance Mean": perm_importance.importances_mean,
            "Permutation Importance Std": perm_importance.importances_std
        }).sort_values(by="Permutation Importance Mean", ascending=False)
        
         # Determine if the model is an XGBoost model
        is_xgb_model = hasattr(model, 'get_booster')
        
        # --- SHAP Preparation: Explainer Model and Data ---
        if is_xgb_model:
            explainer = shap.TreeExplainer(model.get_booster())
            # FIX 2: Define SHAP data as DMatrix to avoid base_score error and ensure categorical compatibility
            X_data_for_shap = xgb.DMatrix(X_data, 
                          enable_categorical=True, 
                          feature_names=X_test.columns.tolist())
            shap_values = explainer.shap_values(X_data_for_shap)
        else:
            explainer = shap.TreeExplainer(model)
            X_data_for_shap = X_data


        # --- SHAP Importance (Mean Absolute) ---
#        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values using the correctly formatted data
        shap_values = explainer.shap_values(X_data_for_shap)
        print("ndim",shap_values.ndim)
        # Handle different SHAP output formats (List or Array)
        if isinstance(shap_values, list):
            # LGBM/XGB multi-class: List of (N_samples, N_features)
            # Stack and take mean across observations and classes (axis 0 and 2)
            shap_values_3d = np.stack(shap_values, axis=2)
            shap_mean_abs_importance = np.mean(np.abs(shap_values_3d), axis=(0, 2))
        elif shap_values.ndim == 3:
            # CatBoost multi-class: (N_samples, N_features, N_classes)
            shap_mean_abs_importance = np.mean(np.abs(shap_values), axis=(0, 2))
        elif shap_values.ndim == 2:
            # Binary/Regression: (N_samples, N_features)
            shap_mean_abs_importance = np.mean(np.abs(shap_values), axis=0)
        else:
            print("Warning: Unknown SHAP output format. SHAP importance skipped.")
            shap_mean_abs_importance = np.zeros(len(X_data.columns))


        perm_importance_df["SHAP Importance (Mean Abs)"] = shap_mean_abs_importance

        # ... (Rest of Weighted Score/Normalization/Coverage logic) ...
        
        # Return the simplified summary, the raw shap values, and X_test
        return perm_importance_df, shap_values, X_test
        
    else:
        raise ValueError(f"Unknown evaluation_type: {evaluation_type}. Must be 'softweights' or 'inference'.")

        
def plot_inference_featureimportance_for_champion(
    shap_values, 
    X_test, 
    label_encoder, 
    save_path: str
):
    """
    Generates, displays, and saves SHAP summary bar plots for the champion model.
    
    Arguments:
        shap_values: The raw SHAP value array (or list of arrays for multiclass)
        X_test: The DataFrame of test features used for the SHAP calculation.
        label_encoder: The fitted LabelEncoder with class names.
        save_path: The file path where the plot will be saved.
    """
    
    # Check if shap_values is a list (multiclass) and handle it for the summary plot
    # The shap library handles the list/array distinction for the plot_type="bar" well.
    
    class_names = label_encoder.classes_

    print("\nVisualizing, saving, and closing Champion Model Global Feature Importance...")
    plt.figure(figsize=(10, 6))
    
    # 1. Generate the SHAP Summary Bar Plot
    # plot_type="bar" is the standard plot showing mean absolute SHAP value
    shap.summary_plot(
        shap_values, 
        X_test, 
        plot_type="bar", 
        show=False, 
        max_display=15, 
        class_names=class_names # Important for multiclass labels
    )
    
    plt.title(f"Champion Model Global Feature Importance (Mean Absolute SHAP Value)")
    plt.tight_layout()
    
    # 2. Save the plot
    plt.savefig(save_path)
    print(f"Plot for Global Feature Importance is saved to: {save_path}")
    
    plt.show() # Display the plot
    plt.close() # Important to close the plot to free memory

    # The rest of the logic from your original function (calculating the dataframe)
    # is redundant if you already have the final_importance_df from calculate_feature_importance,
    # but kept here for completeness if you still need this calculation.
    
    # mean_abs_shap is used to verify consistency across classes for multi-class
    if isinstance(shap_values, list):
        shap_values_3d = np.array(shap_values)
        mean_abs_shap = np.mean(np.abs(shap_values_3d), axis=(0, 1))
    
    else:
       if shap_values.ndim > 1:
          mean_abs_shap_per_class = np.mean(np.abs(shap_values), axis=0)
          if mean_abs_shap_per_class.ndim == 2:
              mean_abs_shap = np.mean(mean_abs_shap_per_class, axis=1)
          else:
              mean_abs_shap = mean_abs_shap_per_class
       else:
          mean_abs_shap = np.abs(shap_values).flatten()
        
    global_importance_df = pd.DataFrame({
        'Feature': X_test.columns,
        'SHAP Importance (Mean Abs)': mean_abs_shap
    }).sort_values(by='SHAP Importance (Mean Abs)', ascending=False)

    print("\nGlobal Feature Importance (averaged across all classes):")
    print(global_importance_df)

    return global_importance_df

  
  
## ======================================================================
### 🚀 COHORT-SHAP-IMPORTANCE-PIPELINE
## ======================================================================

def run_cohort_pipeline(config: ModelConfig):
    """
    The main pipeline for the cohort model, including data prep, training, evaluation, and SHAP.
    """
    print(f"\n{'='*50}\nStarting Cohort Model Pipeline for {config.data_path}\n{'='*50}")
    
    # 1. Load Data
    try:
        df = pd.read_csv(config.data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {config.data_path}. Skipping cohort pipeline.")
        return

    # 2. Prepare and Split Data
    print("\nPreparing and Splitting data...")
    X_encoded, y, country_clean, processed_data = preprocess_data(
        config=config, df=df, country_column="Country", new_column_name="Country_clean", min_obs=10, 
        id_columns=['Specialty (CDV2)', 'Country']
    )
    processed_data.to_csv(config.processesed_datapath_for_shap, index=False)

    X_train, X_val, X_test, y_train, y_val, y_test, country_train, country_val, country_test, split_indicator = split_data(
        config=config, X=X_encoded, y=y, country_column=country_clean
    )
    train_data, train_labels, train_weights = convert_data_for_weighted_training(X_train, y_train)

    # 3. Model Optimization and Training
    factory = EnhancedModelFactory()
    random_state = config.random_state
    n_trials = config.n_trials
    
    # LightGBM
    best_params_lgb = optimize_lgb_cohort(objective_cls=OptunaObjectiveEnhanced, X_train=X_train, y_train=y_train, country_clean=country_clean, train_data=train_data, train_labels=train_labels, train_weights=train_weights, X_val=X_val, y_val=y_val, config=config, factory=factory,n_trials = n_trials)
    model_lgb = lgb.LGBMClassifier(**best_params_lgb, random_state=random_state)
    model_lgb.fit(train_data, train_labels, sample_weight=train_weights)

    # BalancedRandomForest
    best_params_brf = optimize_brf_cohort(objective_cls=OptunaObjectiveEnhanced, X_train=X_train, y_train=y_train, country_clean=country_clean, train_data=train_data, train_labels=train_labels, train_weights=train_weights, X_val=X_val, y_val=y_val, config=config, factory=factory,n_trials = n_trials)
    model_brf = BalancedRandomForestClassifier(**best_params_brf, random_state=random_state)
    model_brf.fit(train_data, train_labels, sample_weight=train_weights)

    # CatBoost
    best_params_catboost = optimize_catboost_cohort(objective_cls=OptunaObjectiveEnhanced, X_train=X_train, y_train=y_train, country_clean=country_clean, train_data=train_data, train_labels=train_labels, train_weights=train_weights, X_val=X_val, y_val=y_val, config=config, factory=factory,n_trials = n_trials)
    model_catboost = CatBoostClassifier(**best_params_catboost, loss_function="MultiClass", verbose=0, random_state=random_state)
    model_catboost.fit(train_data, train_labels, sample_weight=train_weights)
    
    #XgBoost
    best_params_xgboost = optimize_xgboost_cohort(objective_cls=OptunaObjectiveEnhanced, X_train=X_train, y_train=y_train, country_clean=country_clean, train_data=train_data, train_labels=train_labels, train_weights=train_weights, X_val=X_val, y_val=y_val, config=config, factory=factory,n_trials = n_trials)
    model_xgboost = xgb.XGBRFClassifier(**best_params_xgboost, loss_function="MultiClass", verbose=0, random_state =random_state)
    model_xgboost.fit(train_data, train_labels, sample_weight=train_weights)

    
    # 4. Predict
    y_pred_proba_df_lgb = pd.DataFrame(model_lgb.predict_proba(X_test), columns=y_test.columns, index=X_test.index)
    y_pred_proba_df_brf = pd.DataFrame(model_brf.predict_proba(X_test), columns=y_test.columns, index=X_test.index)
    y_pred_proba_df_catboost = pd.DataFrame(model_catboost.predict_proba(X_test), columns=y_test.columns, index=X_test.index)
    y_pred_proba_df_xgboost = pd.DataFrame(model_xgboost.predict_proba(X_test), columns=y_test.columns, index=X_test.index)
    
    
    y_pred_proba_train_df_lgb = pd.DataFrame(model_lgb.predict_proba(X_train), columns=y_train.columns, index=X_train.index)
    y_pred_proba_train_df_brf = pd.DataFrame(model_brf.predict_proba(X_train), columns=y_train.columns, index=X_train.index)
    y_pred_proba_train_df_catboost = pd.DataFrame(model_catboost.predict_proba(X_train), columns=y_train.columns, index=X_train.index)
    y_pred_proba_train_df_xgboost = pd.DataFrame(model_xgboost.predict_proba(X_train), columns=y_train.columns, index=X_train.index)
    
    y_pred_proba_val_df_lgb = pd.DataFrame(model_lgb.predict_proba(X_val), columns=y_val.columns, index=X_val.index)
    y_pred_proba_val_df_brf = pd.DataFrame(model_brf.predict_proba(X_val), columns=y_val.columns, index=X_val.index)
    y_pred_proba_val_df_catboost = pd.DataFrame(model_catboost.predict_proba(X_val), columns=y_val.columns, index=X_val.index)
    y_pred_proba_val_df_xgboost = pd.DataFrame(model_xgboost.predict_proba(X_val), columns=y_val.columns, index=X_val.index)
    
    # 5. Evaluate and Generate Tables
    y_preds_test = {"LightGBM": y_pred_proba_df_lgb, "BalancedRandomForest": y_pred_proba_df_brf, "CatBoost": y_pred_proba_df_catboost, "XGBoost":y_pred_proba_df_xgboost}
    evaluation_results_test = evaluate_models(y_test, y_preds_test)
    y_preds_train = {"LightGBM": y_pred_proba_train_df_lgb, "BalancedRandomForest": y_pred_proba_train_df_brf, "CatBoost": y_pred_proba_train_df_catboost , "XGBoost":y_pred_proba_train_df_xgboost}
    evaluation_results_train = evaluate_models(y_train, y_preds_train)

    evaluation_results_test['dataset'] = 'test'
    evaluation_results_train['dataset'] = 'train'
    evaluation_results = pd.concat([evaluation_results_train, evaluation_results_test], ignore_index=True)
    
    evaluation_results.to_csv(config.initial_models_evaluation, index=False)
    print("\nEvaluation results saved to 'outputs/evaluation_results.csv'.")
    
    results_table_lgb = generate_results_table(y_true_train=y_train, y_pred_train={"LightGBM": y_pred_proba_train_df_lgb}, country_train=X_train, y_true_val=y_val, y_pred_val={"LightGBM": y_pred_proba_val_df_lgb}, country_val=X_val, y_true_test=y_test, y_pred_test={"LightGBM": y_pred_proba_df_lgb}, country_test=X_test)
    results_table_brf = generate_results_table(y_true_train=y_train, y_pred_train={"BalancedRandomForest": y_pred_proba_train_df_brf}, country_train=X_train, y_true_val=y_val, y_pred_val={"BalancedRandomForest": y_pred_proba_val_df_brf}, country_val=X_val, y_true_test=y_test, y_pred_test={"BalancedRandomForest": y_pred_proba_df_brf}, country_test=X_test)
    results_table_catboost = generate_results_table(y_true_train=y_train, y_pred_train={"Catboost": y_pred_proba_train_df_catboost}, country_train=X_train, y_true_val=y_val, y_pred_val={"Catboost": y_pred_proba_val_df_catboost}, country_val=X_val, y_true_test=y_test, y_pred_test={"Catboost": y_pred_proba_df_catboost}, country_test=X_test)
    results_table_xgboost = generate_results_table(y_true_train=y_train, y_pred_train={"XGBoost": y_pred_proba_train_df_xgboost}, country_train=X_train, y_true_val=y_val, y_pred_val={"XGBoost": y_pred_proba_val_df_xgboost}, country_val=X_val, y_true_test=y_test, y_pred_test={"XGBoost": y_pred_proba_df_xgboost}, country_test=X_test)
    
    # 6. Calculate SHAP
    shap_summary_lgb, shap_per_observation_lgb, shap_values_lgb = calculate_feature_importance_unified(model_lgb, X_test, profile_names=config.target_col, findf = None,X_data_train_cols = None, y_test=None)
    shap_summary_brf, shap_per_observation_brf, shap_values_brf = calculate_feature_importance_unified(model_brf, X_test, profile_names=config.target_col,findf = None,X_data_train_cols = None, y_test=None)
    shap_summary_catboost, shap_per_observation_catboost, shap_values_catboost = calculate_feature_importance_unified(model_catboost, X_test, profile_names=config.target_col,findf = None,X_data_train_cols = None, y_test=None)
    shap_summary_xgboost, shap_per_observation_xgboost, shap_values_xgboost = calculate_feature_importance_unified(model_xgboost, X_test, profile_names=config.target_col,findf = None,X_data_train_cols = None, y_test=None)
    
    
    # 7.1 Filter for Test set performance (LogLoss is the chosen metric)
    test_results = evaluation_results[evaluation_results['dataset'] == 'test']
    print(test_results.head())
    min_log_loss_index = test_results['Log Loss_all'].idxmin()
    best_model_row = test_results.loc[min_log_loss_index]
    best_model_name = best_model_row['Model']
    best_log_loss = best_model_row['Log Loss_all']

    print(f"\n✨ Champion Model Selected: **{best_model_name}** (Test Log Loss: {best_log_loss:.4f})")

    # 7.2 Map the Champion Model name to its corresponding data
    if best_model_name == "LightGBM":
        champion_model = model_lgb
        champion_shap_summary = shap_summary_lgb
        #champion_shap_values = shap_values_lgb
        champion_shap_per_observation = shap_per_observation_lgb
    elif best_model_name == "BalancedRandomForest":
        champion_model = model_brf
        champion_shap_summary = shap_summary_brf
        #champion_shap_values = shap_values_brf
        champion_shap_per_observation = shap_per_observation_brf
    elif best_model_name == "XgBoost":
        champion_model = model_xgboost
        champion_shap_summary = shap_summary_xgboost
        #champion_shap_values = shap_values_xgboost
        champion_shap_per_observation = shap_per_observation_xgboost
    elif best_model_name == "CatBoost":
        champion_model = model_catboost
        champion_shap_summary = shap_summary_catboost
        #champion_shap_values = shap_values_catboost
        champion_shap_per_observation = shap_per_observation_catboost
    else:
        # Fallback to LGBM if something goes wrong
        champion_model = model_lgb
        champion_shap_summary = shap_summary_lgb
        #champion_shap_values = shap_values_lgb
        champion_shap_per_observation = shap_per_observation_lgb
        best_model_name = "LightGBM (Fallback)"
   
    cohort_champion_model_prefix = best_model_name.replace(' ', '_')
    print(cohort_champion_model_prefix)
    new_shap_path = config.shap_path.replace(".csv", f"_{cohort_champion_model_prefix}.csv")
    # 10. Save Champion Feature Importance CSV
    champion_shap_summary[['Feature','SHAP Importance (Mean Across Profiles)']].to_csv(new_shap_path, index=False)
    print(f"Champion ( {best_model_name} ) Mean SHAP Importance saved to {new_shap_path}")
       
    # Define a consistent path for the metadata file (e.g., in the output directory)
    metadata_dir = os.path.dirname(config.shap_path)
    metadata_file_path = os.path.join(metadata_dir, "champion_metadata.json")
    metadata = {
    "champion_model_name": best_model_name,
    "champion_model_prefix": cohort_champion_model_prefix,
    "shap_file_path": new_shap_path 
    }
    with open(metadata_file_path, 'w') as f:
        json.dump(metadata, f, indent=4)    
    print(f"✨ Champion metadata saved to: {metadata_file_path}")
    
        # 7. Save Results
    SHAP_championmodel_1 = pd.concat([processed_data[["ID"]].reset_index(drop=True), split_indicator.reset_index(drop=True), champion_shap_per_observation[0].reset_index(drop=True)], axis = 1).rename(columns={0: 'Split_Indicator'})
    SHAP_championmodel_2 = pd.concat([processed_data[["ID"]].reset_index(drop=True), split_indicator.reset_index(drop=True), champion_shap_per_observation[1].reset_index(drop=True)], axis = 1).rename(columns={0: 'Split_Indicator'})
    SHAP_championmodel_3 = pd.concat([processed_data[["ID"]].reset_index(drop=True), split_indicator.reset_index(drop=True), champion_shap_per_observation[2].reset_index(drop=True)], axis = 1).rename(columns={0: 'Split_Indicator'})
    
    
    with pd.ExcelWriter(config.initial_models_evaluation, engine='openpyxl') as writer:
        evaluation_results.to_excel(writer, sheet_name="Combined Results", index=False)
        results_table_lgb.to_excel(writer, sheet_name="LGB", index=False)
        results_table_brf.to_excel(writer, sheet_name="BRF", index=False)
        results_table_catboost.to_excel(writer, sheet_name="Catboost", index=False)
        results_table_xgboost.to_excel(writer, sheet_name="XgBoost", index=False)
        shap_summary_lgb.to_excel(writer, sheet_name="SHAP_summary_lgb", index=False)
        shap_summary_brf.to_excel(writer, sheet_name="SHAP_summary_brf", index=False)
        shap_summary_catboost.to_excel(writer, sheet_name="SHAP_summary_catboost", index=False)
        shap_summary_xgboost.to_excel(writer, sheet_name="SHAP_summary_xgboost", index=False)
        SHAP_championmodel_1.to_excel(writer, sheet_name="SHAP_champion_model_class1", index=False)
        SHAP_championmodel_2.to_excel(writer, sheet_name="SHAP_champion_model_class2", index=False)
        SHAP_championmodel_3.to_excel(writer, sheet_name="SHAP_champion_model_class3", index=False)

    profile_names = config.target_col
    X_data_for_shap = X_encoded
    explainer = shap.TreeExplainer(champion_model)
    champion_shap_values = explainer.shap_values(X_data_for_shap)
    
    for class_idx in range(champion_shap_values.shape[2]):  
        print(f"Generating SHAP summary plot for class {class_idx + 1}...")
        print(profile_names[class_idx])

        # Extract SHAP values for the current class
        class_shap_values = champion_shap_values[:, :, class_idx]  # Shape: (361, 125)

        # Create a new figure for the current class
        plt.figure(figsize=(10, 6))

        # Generate SHAP summary plot for the current class
        shap.summary_plot(class_shap_values, X_encoded, max_display=20, show=False)
        plt.title(f"SHAP for {profile_names[class_idx]}")        
        
        # Save the plot for the current class
        class_plot_path = f"outputs/digital_cohort/step0_preprocessing/{cohort_champion_model_prefix}_SHAPIMPPLOT_{profile_names[class_idx]}.png"
        plt.savefig(class_plot_path, dpi=300, bbox_inches="tight")
        print(f"SHAP summary plot for class {class_idx + 1} saved to {class_plot_path}")

        # Close the figure to avoid overlapping plots
        plt.close() 

    print(f"\nCohort Model Pipeline completed.")
    return cohort_champion_model_prefix
    
## ======================================================================
### 🚀 INFERENCE-SHAP-IMPORTANCE-PIPELINE
## ======================================================================

def run_inference_pipeline (config: ModelConfig):
    """
    Identifies categorical columns, performs data preparation, model training (XGBRF),
    and calculates Permutation and SHAP feature importance metrics for inference.
    """
    
    try:
        findf = pd.read_csv(config.data_path)
        df = findf.copy()                    
    except FileNotFoundError:
        print(f"Error: Data file not found at {config.data_path}. Skipping Inference pipeline.")
        return
      
    target_column = config.target_col
    id_col = config.id_col
    test_size = config.test_size
    random_state = config.random_state
    n_trials = config.n_trials
    print("n_trials",n_trials)
    
    # --- Data Preparation ---
    X = df.drop(columns=[target_column, id_col], errors='ignore')
    y = df[target_column]
    
    
    if 'country' not in X.columns:
        print("Country column not found. Skipping feature importance calculation.")
        return None, None, None

    
    # Stratify key construction (Ensure target column is in the original data)
    stratify_key = y.astype(str) + '_' + X['country'].astype(str)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Ensure all categorical columns are set before train_test_split
    for col in categorical_cols:
         if col in X.columns and X[col].dtype.name != 'category':
            X[col] = X[col].astype('category')
    
    X_train, X_val, X_test, y_train, y_val, y_test, train_key, val_key, test_key, split_indicator = \
    split_data(
        config=config,
        X=X,
        y=y_encoded,
        country_column=stratify_key
    )

    MISSING_PLACEHOLDER = 'missing'
    for col in categorical_cols:
        if col in X_train.columns:

            # 1. Ensure X_train is type 'category' and add the placeholder
            X_train[col] = X_train[col].astype(str).astype('category')

            # Add 'missing' category to X_train if not present, and fill NaNs
            X_train[col] = X_train[col].cat.add_categories(MISSING_PLACEHOLDER)
            X_train[col] = X_train[col].fillna(MISSING_PLACEHOLDER)

            # Get the definitive categorical type object from X_train
            train_cat_dtype = X_train[col].dtype

            # 2. Apply X_train's definitive categories to X_val and X_test
            for df_set in [X_val, X_test]:
                if col in df_set.columns:

                    # Convert to string first to handle non-categorical data, then apply the definitive dtype
                    df_set[col] = df_set[col].astype(str)
                    df_set[col] = df_set[col].astype(train_cat_dtype) 

                    # Fill any remaining NaNs/values not in categories with the placeholder
                    df_set[col] = df_set[col].fillna(MISSING_PLACEHOLDER)
    
    native_train_cols = X_train.columns
    is_multiclass = len(y.unique()) > 2
                    
    # 3. Model Optimization and Training (Inference Mode - Use Hard Labels)
    factory = EnhancedModelFactory()
    RANDOM_STATE = config.random_state
    n_trials = config.n_trials
    
    # --- LightGBM (LGBM) ---
    best_params_lgb = optimize_lgb_inference(
        objective_cls=OptunaObjectiveEnhanced, 
        X_train=X_train, y_train=y_train, # Hard labels/features
        X_val=X_val, y_val=y_val, # Hard labels for validation
        config=config, factory=factory, n_trials=n_trials,
        country_clean=None,      
        train_data=X_train,        
        train_labels=y_train,    
        train_weights=None
    )
    model_lgb = lgb.LGBMClassifier(**best_params_lgb, random_state=RANDOM_STATE)
    model_lgb.fit(X_train, y_train)
    
    brf_encoder = TargetEncoder(cols=categorical_cols, handle_missing='value', smoothing=0.2)
    X_train_brf_encoded = brf_encoder.fit_transform(X_train, y_train)
    X_val_brf_encoded = brf_encoder.transform(X_val)
    brf_encoded_train_cols = X_train_brf_encoded.columns

    best_params_brf = optimize_brf_inference(
        objective_cls=OptunaObjectiveEnhanced,
        X_train=X_train_brf_encoded, y_train=y_train, 
        X_val=X_val_brf_encoded, y_val=y_val, 
        config=config, factory=factory, n_trials=n_trials,
        country_clean=None,
        train_data=X_train_brf_encoded, train_labels=y_train, train_weights=None
    )
    
    # Final Model: Create a pipeline for clean prediction/SHAP handling
    model_brf_base = BalancedRandomForestClassifier(**best_params_brf, random_state=RANDOM_STATE, n_jobs=-1)
    
    # The final model is the pipeline that includes the TargetEncoder
    model_brf = Pipeline(steps=[
        ('encoder', brf_encoder), # Use the fitted encoder
        ('classifier', model_brf_base)
    ])
    model_brf.fit(X_train, y_train)
    
    
    best_params_catboost = optimize_catboost_inference(
        objective_cls=OptunaObjectiveEnhanced, 
        X_train=X_train, y_train=y_train, 
        X_val=X_val, y_val=y_val, 
        config=config, factory=factory, n_trials=n_trials,
        country_clean=None,      
        train_data=X_train,        
        train_labels=y_train,    
        train_weights=None
    )
    model_catboost = CatBoostClassifier(
        **best_params_catboost, 
        loss_function="MultiClass" if is_multiclass else "MultiClassOneVsAll", 
        verbose=0, random_seed=RANDOM_STATE
    )
    model_catboost.fit(X_train, y_train,cat_features=categorical_cols)
    
    # --- XGBRF (Add missing model) ---
    best_params_xgboost = optimize_xgboost_inference(
        objective_cls=OptunaObjectiveEnhanced, 
        X_train=X_train, y_train=y_train, 
        X_val=X_val, y_val=y_val, 
        config=config, factory=factory, n_trials=n_trials,
        country_clean=None,      
        train_data=X_train,        
        train_labels=y_train,    
        train_weights=None
    )
    model_xgboost = xgb.XGBRFClassifier(
        **best_params_xgboost, 
        objective="multi:softmax" if is_multiclass else "binary:logistic",
        num_class=len(label_encoder.classes_) if is_multiclass else 0,
        enable_categorical=True, 
        random_state=random_state
    )
    model_xgboost.fit(X_train, y_train) # Training XGBRF

    print(f"Best CatBoost Params (Inference): {best_params_catboost}")
    
    # Evaluate models on test data
    print("\nEvaluating models on test data...")
    y_pred_lgb = model_lgb.predict(X_test)
    
    X_test_brf = X_test.reindex(columns= X_train_brf_encoded.columns, fill_value=0)
    y_pred_brf = model_brf.predict(X_test_brf)

    y_pred_catboost = model_catboost.predict(X_test)
    y_pred_xgboost = model_xgboost.predict(X_test)

    y_pred_df_lgb = pd.DataFrame(y_pred_lgb,  index=X_test.index)
    y_pred_df_brf = pd.DataFrame(y_pred_brf, index=X_test_brf.index)
    y_pred_df_catboost = pd.DataFrame(y_pred_catboost, index=X_test.index)
    y_pred_df_xgboost = pd.DataFrame( y_pred_xgboost,index=X_test.index)

     #  --- TRAIN Predictions ---
    print("Generating TRAIN predictions...")
    y_pred_train_lgb = model_lgb.predict(X_train)
    y_pred_train_brf = model_brf.predict(X_train)
    y_pred_train_catboost = model_catboost.predict(X_train)
    y_pred_train_xgboost = model_xgboost.predict(X_train)
    
    # Convert TRAIN predictions to DataFrames
    y_pred_train_df_lgb = pd.DataFrame(y_pred_train_lgb,  index=X_train.index)
    y_pred_train_df_brf = pd.DataFrame(y_pred_train_brf,  index=X_train.index)
    y_pred_train_df_catboost = pd.DataFrame(y_pred_train_catboost,  index=X_train.index)
    y_pred_train_df_xgboost = pd.DataFrame(y_pred_train_xgboost,  index=X_train.index)

    print("Generating VALIDATION predictions...")
    y_pred_val_lgb = model_lgb.predict(X_val)
    y_pred_val_brf = model_brf.predict(X_val)
    y_pred_val_catboost = model_catboost.predict(X_val)
    y_pred_val_xgboost = model_xgboost.predict(X_val)
    
    # Convert VALIDATION predictions to DataFrames
    y_pred_val_df_lgb = pd.DataFrame(y_pred_val_lgb, index=X_val.index)
    y_pred_val_df_brf = pd.DataFrame(y_pred_val_brf,  index=X_val.index)
    y_pred_val_df_catboost = pd.DataFrame(y_pred_val_catboost,  index=X_val.index)
    y_pred_val_df_xgboost = pd.DataFrame(y_pred_val_xgboost,  index=X_val.index)
    
    # Evaluate models on test data
    inf_evaluation_results_train = []
    inf_evaluation_results_test = []
    
    print("\nEvaluating models on test data...")
    y_preds_test = {
        "LightGBM": y_pred_df_lgb,
#        "BalancedRandomForest": y_pred_df_brf,
        "CatBoost": y_pred_df_catboost,
        "XgBoost" : y_pred_df_xgboost
    }
    for model_name, y_pred_df in y_preds_test.items():
        shap_test_model_metrics = calculate_fold_metrics(
        y_true=y_test, 
        y_pred=y_pred_df, 
        dataset_name= "Test Set" ,
        model_type="classification" # Use classification for hard labels
    )
    shap_test_model_metrics["Model"] = model_name  
    inf_evaluation_results_test.append(shap_test_model_metrics)
    
    # Evaluate models on train data
    print("\nEvaluating models on train data...")
    y_preds_train = {
        "LightGBM": y_pred_train_df_lgb,
#        "BalancedRandomForest": y_pred_train_df_brf,
        "CatBoost": y_pred_train_df_catboost,
        "XgBoost" : y_pred_train_df_xgboost
    }
    for model_name, y_pred_df in y_preds_train.items():
        shap_train_model_metrics = calculate_fold_metrics(
        y_true=y_train, 
        y_pred=y_pred_df, 
        dataset_name= "Train Set", 
        model_type="classification" # Use classification for hard labels
    )
    shap_train_model_metrics["Model"] = model_name  
    inf_evaluation_results_train.append(shap_train_model_metrics)
   
    # Combine train and test evaluation results into a single DataFrame
    inf_evaluation_results = pd.concat([pd.DataFrame(inf_evaluation_results_train), pd.DataFrame(inf_evaluation_results_test)], ignore_index=True)
    

    # Save evaluation results
    inf_evaluation_results.to_csv(config.initial_models_evaluation, index=False)
    print("\nEvaluation results saved to 'outputs/evaluation_results.csv'.")
    
    native_train_cols = X_train.columns
    # --- LightGBM ---
    print("\nCalculating SHAP feature importance for LightGBM...")
    shap_summary_lgb, shap_values_lgb_raw, X_test_lgb_used = calculate_feature_importance_unified(
        model=model_lgb, 
        X_test=X_test, 
        y_test=y_test, 
        X_data_train_cols=native_train_cols, 
        findf=findf, 
        evaluation_type="inference" # Pass the string literal
    )
    print("\nSHAP/Permutation Feature Importance (LightGBM):")
    print(shap_summary_lgb)
    
    print("\nCalculating SHAP feature importance for BRF (Target Encoded)...")
    # 1. Get the trained BRF classifier from the pipeline
    model_brf_classifier = model_brf['classifier']
    
    # 2. Get the Target Encoded test data used for SHAP
    X_test_brf_encoded = model_brf['encoder'].transform(X_test)
    
    # 3. Calculate SHAP on the encoded data using the classifier
    shap_summary_brf, shap_values_brf_raw, X_test_brf_used = calculate_feature_importance_unified(
        model=model_brf_classifier, 
        X_test=X_test_brf_encoded, # Use the encoded data
        y_test=y_test, 
        X_data_train_cols=brf_encoded_train_cols, # Use the encoded column names for alignment
        findf=findf, 
        evaluation_type="inference"
    )
    
    # --- CatBoost ---
    print("\nCalculating SHAP feature importance for Catboost...")
    shap_summary_catboost, shap_values_catboost_raw, X_test_cat_used = calculate_feature_importance_unified(
        model=model_catboost, 
        X_test=X_test, 
        y_test=y_test, 
        X_data_train_cols=native_train_cols, # Use the original feature set train columns
        findf=findf, 
        evaluation_type="inference"
    )
    
    # --- XGBRF ---
    print("\nCalculating SHAP feature importance for XGBRF...")
    shap_summary_xgboost, shap_values_xgboost_raw, X_test_xgboost_used = calculate_feature_importance_unified(
        model=model_xgboost, 
        X_test=X_test, 
        y_test=y_test, 
        X_data_train_cols=native_train_cols, # Use the original feature set train columns
        findf=findf, 
        evaluation_type="inference"
    )

    # 7. CHAMPION MODEL SELECTION 🏆 (New Logic)
    print("\n--- Selecting Champion Model ---")
    
    
    # 7.1 Filter for Test set performance (LogLoss is the chosen metric)
    test_results = inf_evaluation_results[inf_evaluation_results['Dataset'] == 'Test Set']
    
    max_accuracy_index = test_results['Accuracy'].idxmax()
    best_model_row = test_results.loc[max_accuracy_index]
    best_model_name = best_model_row['Model']
    best_accuracy = best_model_row['Accuracy']

    print(f"\n✨ Champion Model Selected: **{best_model_name}** (Test Log Loss: {best_accuracy:.4f})")

    # 7.2 Map the Champion Model name to its corresponding data
    if best_model_name == "LightGBM":
        champion_model = model_lgb
        champion_shap_summary = shap_summary_lgb

    elif best_model_name == "BalancedRandomForest":
        champion_model = model_brf
        champion_shap_summary = shap_summary_brf

    elif best_model_name == "CatBoost":
        champion_model = model_catboost
        champion_shap_summary = shap_summary_catboost

    elif best_model_name == "XgBoost":
        champion_model = model_xgboost
        champion_shap_summary = shap_summary_xgboost

    else:
        # Fallback to LGBM if something goes wrong
        champion_model = model_lgb
        champion_shap_summary = shap_summary_lgb
        best_model_name = "LightGBM (Fallback)"
        
    inference_champion_model_prefix = best_model_name.replace(' ', '_')
    print(inference_champion_model_prefix)
    new_shap_path = config.shap_path.replace(".csv", f"_{inference_champion_model_prefix}.csv")
    print("champion_shap_summary",champion_shap_summary.columns)
    print(new_shap_path)
    output_dir = os.path.dirname(new_shap_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    # 10. Save Champion Feature Importance CSV
    champion_shap_summary[['Feature','SHAP Importance (Mean Abs)']].to_csv(new_shap_path, index=False)
    print(f"Champion ( {best_model_name} ) Mean SHAP Importance saved to {new_shap_path}")
       
    # Define a consistent path for the metadata file (e.g., in the output directory)
    metadata_dir = os.path.dirname(config.shap_path)
    metadata_file_path = os.path.join(metadata_dir, "champion_metadata.json")
    metadata = {
    "champion_model_name": best_model_name,
    "champion_model_prefix": inference_champion_model_prefix,
    "shap_file_path": new_shap_path 
    }
    with open(metadata_file_path, 'w') as f:
        json.dump(metadata, f, indent=4)    
    print(f"✨ Champion metadata saved to: {metadata_file_path}")
  
    # 11. Generate SHAP Summary Plots (Using Champion Model)
    print("\n--- Generating SHAP Summary Plots (Champion Model) ---")
    
    profile_names = config.target_col
    explainer = shap.TreeExplainer(champion_model)
    champion_shap_values = explainer.shap_values(X_test)
    plot_inference_featureimportance_for_champion( shap_values=champion_shap_values,
                                                  X_test=X_test,
                                                  label_encoder=label_encoder,
                                                  save_path=config.shap_plot_path)
    # --- START OF KEPT EXCEL BLOCK ---
    with pd.ExcelWriter(config.initial_models_evaluation, engine='openpyxl') as writer:
            inf_evaluation_results.to_excel(writer, sheet_name="Combined Results", index=False)
            shap_summary_lgb.to_excel(writer, sheet_name="SHAP_summary_LGBM", index=False)
            shap_summary_brf.to_excel(writer, sheet_name="SHAP_summary_BRF", index=False)
            shap_summary_catboost.to_excel(writer, sheet_name="SHAP_summary_CAT", index=False)
            shap_summary_xgboost.to_excel(writer, sheet_name="SHAP_summary_XgBoost", index=False)

         
    print("All Inference Tasks Completed!")
    return inference_champion_model_prefix
  
if __name__ == "__main__":
        
        inference_champion_model_prefix = run_inference_pipeline(config = inference_model_config)        
        cohort_champion_model_prefix = run_cohort_pipeline(config = cohort_model_config)
        print("Both Inference and Cohort SHAP IMPORTANCE Completed!")
      