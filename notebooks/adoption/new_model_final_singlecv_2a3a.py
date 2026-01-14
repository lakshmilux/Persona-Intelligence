# Standard libraries
import os
import sys
import time
import json
import uuid
import warnings
import random  
from collections import Counter
import cloudpickle
# Data manipulation
import pandas as pd
import numpy as np
import cloudpickle
import shutil

# Scikit-learn
import sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, precision_score, recall_score, make_scorer,
                             accuracy_score, classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
sklearn.set_config(enable_metadata_routing=True)

# Imbalanced-learn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.ensemble import BalancedRandomForestClassifier

# Tree models
import lightgbm as lgb
from catboost import CatBoostClassifier

# Stacking
from sklearn.ensemble import StackingClassifier

# Hyperparameter tuning
import optuna

# Statistics
from scipy.stats import gmean

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to sys.path first (before importing from src)
#sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Setup project paths (must be before any project imports)
# This ensures imports work and relative paths in config files are correct
from src.common.utils import setup_project_paths
setup_project_paths()

sys.path.insert(0,os.path.abspath(os.path.join(setup_project_paths(), "src")))


# Import config module and create ModelConfig
from src.adoption_cohort import config as cohort_config
from src.adoption import config as inference_config
from src.common.config import ModelConfig
from src.common.pipeline_factory_enhanced import EnhancedPipelineFactory, build_final_catboost_softweights_model
from src.common.model_factory_enhanced import EnhancedModelFactory
from src.common.utils import set_global_seeds
from src.common.metrics import geometric_mean_f1_scorer, calculate_fold_metrics, calculate_metrics_by_country, custom_log_loss
from src.common.stratification_tools import create_stratification_df

from src.common.data_preprocessing import load_and_split_data, convert_data_for_weighted_training
from src.common.new_optunaenhanced import OptunaObjectiveEnhanced
#from src.common.ok_predictions import preprocess_data, preprocess_and_predict_on_ok_data

def singlecv(config_module):

    config = ModelConfig.from_module(config_module)
    print("TARGET_COL",config.target_col)
    set_global_seeds(config.random_state)
    
    #Determine Model Type and Optimization Direction
    is_softweights_cohort = isinstance(config.target_col, list)
    model_type_str = "softweights" if is_softweights_cohort else "classification"
    optuna_direction = "minimize" if is_softweights_cohort else "maximize" 
    
    print(f"\n--- Running Pipeline for Model Type: **{model_type_str}** ---")
    print(f"Target Column(s): {config.target_col}")
    
    gmean_f1_scorer_obj = make_scorer(geometric_mean_f1_scorer, greater_is_better=True)
    
  
    # -------------------- DATA --------------------------------------
    X_with_ids, X_train_full, y_train_full, X_test_final, y_test_final, numerical_cols, categorical_cols, feature_ranking, stratify_key = load_and_split_data(
        config=config,
        stratify_key=config.stratify_key
    )
    print(f"Stratification Key: {stratify_key}")
    print("\n--- Data Loaded and Split ---")

    # Initialize pipeline factory for the selected approach
    pipeline_factory = EnhancedPipelineFactory(numerical_cols, categorical_cols, config=config)


    # -------------------- HYPERPARAMETER OPTIMIZATION ----------------------
    print("\n--- Starting Hyperparameter Optimization ---")
    print(f"N Trials: {config.n_trials}")

    # Pass the new output directory and a flag to enable saving nested CV models
    objective = OptunaObjectiveEnhanced(
        X_train_full, 
        y_train_full, 
        stratify_key, 
        feature_ranking, 
        numerical_cols, 
        categorical_cols,
        config=config,
        train_data = None,
        train_labels = None,
        SHAP_model_name=None,
        model_factory = pipeline_factory,
        save_models=True
    )

    study = optuna.create_study(
        direction=optuna_direction, 
        sampler=optuna.samplers.TPESampler(seed=config.random_state),
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(objective, n_trials=config.n_trials, n_jobs=1, show_progress_bar=False)

    # Extract the best hyperparameters
    best_params = study.best_params
    best_params_df = pd.DataFrame(best_params.items(), columns=["Parameter", "Value"])
    # Add N_trials as a new row
    n_trials_row = pd.DataFrame({"Parameter": ["N_trials"], "Value": [config.n_trials]})
    best_params_df = pd.concat([best_params_df, n_trials_row], ignore_index=True)

    print("\n--- Best Hyperparameters Found ---")
    print(best_params_df)

    # Extract the best features
    best_num_features = best_params.get("num_features", len(feature_ranking))
    best_selected_features = feature_ranking[:best_num_features]

    print(f"\n--- Fold results for all folds are saved to `{config.step2a_dir}` ---")

    # -------------------- ESTIMATE FINAL MODEL ON FULL TRAIN SET ----------------------
    print("\n--- Training Final Model on Full Training Set ---")

    # Convert data to soft weights for cohorts
    if is_softweights_cohort:
        train_data, train_labels, train_weights = convert_data_for_weighted_training(X_train_full[best_selected_features], y_train_full)

        # Create the final model using the best hyperparameters and features
        if best_params.get("use_catboost_softweights", False) is True and not best_params.get("use_lgbm_softweights", False) and not best_params.get("use_brf_softweights", False):
          final_model = build_final_catboost_softweights_model(pipeline_factory, best_params, best_selected_features, train_data=train_data, train_labels=train_labels, train_weights=train_weights)
        else:
          final_model= pipeline_factory.build_final_model(best_params, best_selected_features, use_softweights=True)
          final_model.fit(train_data, train_labels, sample_weight=train_weights)
        
        prediction_cols = y_train_full.columns
        
    else:
        final_model = pipeline_factory.build_final_model(best_params, best_selected_features, use_softweights=False) 
        final_model.fit(X_train_full[best_selected_features], y_train_full)   
        prediction_cols = [config.target_col]
        
    # Save the final model
    os.makedirs(os.path.dirname(config.cloudpickle_filename_step3), exist_ok=True)
    with open(config.cloudpickle_filename_step3, "wb") as f:
        cloudpickle.dump(final_model, f)
    print(f"Final model saved to {config.cloudpickle_filename_step3}")


    if is_softweights_cohort:
        y_pred_train = final_model.predict_proba(X_train_full[best_selected_features])
        y_pred_test = final_model.predict_proba(X_test_final[best_selected_features])
    else:
        y_pred_train = final_model.predict(X_train_full[best_selected_features])
        y_pred_test = final_model.predict(X_test_final[best_selected_features])
    
    y_pred_train_df = pd.DataFrame(y_pred_train, columns=prediction_cols, index=X_train_full.index)
    y_pred_test_df = pd.DataFrame(y_pred_test, columns=prediction_cols, index=X_test_final.index)
    print(y_pred_train_df.head())
    print(X_train_full.columns) 
    # Calculate metrics for the train dataset
    print("\n--- Evaluating Train Metrics ---")
    train_metrics = calculate_metrics_by_country(
        X=X_train_full,
        y_real=y_train_full,
        y_pred=y_pred_train_df,
        dataset_name="Train full Data",
        model_type=model_type_str
    )

    train_metrics_df = pd.DataFrame(train_metrics)

    
    # -------------------- CALCUALTE MODEL METRICS ----------------------
    # Calculate metrics for the test dataset
    print("\n--- Evaluating Test Metrics ---")
    test_metrics = calculate_metrics_by_country(
        X=X_test_final,
        y_real=y_test_final,
        y_pred=y_pred_test_df,
        dataset_name="Test Data",
        model_type=model_type_str
    )

    test_metrics_df = pd.DataFrame(test_metrics)

    print("Model Metrics:")
    print(test_metrics_df)
    
    predictions_df = X_test_final.copy()
    if isinstance(y_test_final, pd.Series):
         y_test_final = y_test_final.to_frame(name=config.target_col)
    y_true_df = y_test_final.rename(columns=lambda x: f"y_true_{x}")
    y_pred_df = y_pred_test_df.rename(columns=lambda x: f"y_pred_{x}")
    predictions_df = pd.concat([predictions_df, y_true_df, y_pred_df], axis=1)
    
    # Save predictions
    predictions_df.to_excel(config.predictions_filename_step3, index=False, engine="openpyxl")
    print(f"Test predictions saved to {config.predictions_filename_step3}")
    
    classification_report_df = pd.DataFrame()
    cm_df = pd.DataFrame()
    N_TRIALS = config.n_trials
    
    if not is_softweights_cohort:
      
        print("\n--- Generating Hard-Label Metrics (Classification Report & Confusion Matrix) ---")
        
        # 1. Get the class names from your original data
        class_names = sorted(list(y_train_full.unique()))
        
        print(class_names)
        # Add confusion matrix for saving
        cm = confusion_matrix(y_test_final, y_pred_df, labels=class_names)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        print("cm",cm_df)

        # Prepare classification report and CV results
        classification_report_dict = classification_report(y_test_final, y_pred_df, output_dict=True)
        classification_report_df = pd.DataFrame(classification_report_dict).transpose()
        
        # Add number of trials and Geometric Mean F1 Score to the Classification Report tab
        additional_info = {
            "Number of Trials": N_TRIALS,
            "Geometric Mean F1 Score": geometric_mean_f1_scorer(y_test_final, y_pred_df)
        }
        additional_info_df = pd.DataFrame(additional_info, index=["Additional Info"])

        # Append additional info to the Classification Report DataFrame
        classification_report_df = pd.concat([classification_report_df, additional_info_df], axis=0)

    else:
         print("\n--- Skipping Classification Report & Confusion Matrix (Soft-weights Cohort Model) ---")  
    
    with pd.ExcelWriter(config.metrics_filename_step3) as writer:
        # Model parameters
        best_params_df.to_excel(writer, sheet_name="Best_params", index=True)
        # Save test metrics by country
        test_metrics_df.to_excel(writer, sheet_name="Country test metrics", index=True)
        # Save train metrics by country
        train_metrics_df.to_excel(writer, sheet_name="Country train metrics", index=True)
         # Save feature names
        pd.DataFrame({"Feature Names": best_selected_features}).to_excel(writer, sheet_name="Final_Features", index=False)  
        if not classification_report_df.empty:
             classification_report_df.to_excel(writer,sheet_name='Classification Report',index=False)
        if not cm_df.empty:
             cm_df.to_excel(writer, sheet_name="Confusion Matrix", index=True)  
            
    print(f"Results saved successfully to {config.metrics_filename_step3}")

if __name__ == "__main__":
  
  singlecv(inference_config)
  print("COHORT Singlcecv Starts")
  #singlecv(cohort_config)
  print('ALL SINGLECV Tasks Completed Successfully!')
