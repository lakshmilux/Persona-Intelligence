import os
import sys
import time
import json
import uuid
import warnings
import random
from collections import Counter

# Data manipulation
import pandas as pd
import numpy as np
import cloudpickle
import shutil

# Scikit-learn
import sklearn
print(sklearn.__version__)
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
import xgboost as xgb
from catboost import CatBoostClassifier

# Stacking
from sklearn.ensemble import StackingClassifier

# Hyperparameter tuning
import optuna

# Statistics
from scipy.stats import gmean

# Suppress warnings
warnings.filterwarnings('ignore')

from src.common.utils import setup_project_paths
setup_project_paths()

sys.path.insert(0,os.path.abspath(os.path.join(setup_project_paths(), "src")))

# ---------------------------------------------------------------------
# LOAD FUNCTIONS FROM MODULES
# ---------------------------------------------------------------------
from digital_cohort import config as cohort_config
from digital import config as inference_config
from digital_cohort import config as cohort_config
from common.config import ModelConfig
from common.pipeline_factory_enhanced import EnhancedPipelineFactory, build_final_catboost_softweights_model
from common.model_factory_enhanced import EnhancedModelFactory
from common.stratification_tools import create_stratification_df
from common.utils import set_global_seeds
from common.metrics import geometric_mean_f1_scorer, calculate_fold_metrics, calculate_metrics_by_country, custom_log_loss
from common.data_preprocessing import load_and_split_data, convert_data_for_weighted_training
from common.new_optunaenhanced import OptunaObjectiveEnhanced
from typing import List, Dict, Any, Tuple, Optional
inference_model_config = ModelConfig.from_module(inference_config)
cohort_model_config = ModelConfig.from_module(cohort_config)

def run_nestedcv(config: ModelConfig):
  
    #config = ModelConfig.from_module(config_module)
    total_tuning_start_time = time.time()
    print("TARGET_COL",config.target_col)
    set_global_seeds(config.random_state)

    #Determine Model Type and Optimization Direction
    is_softweights_cohort = isinstance(config.target_col, list)
    is_inference = isinstance(config.target_col, str)
    model_type_str = "softweights" if is_softweights_cohort else "classification"
    optuna_direction = "minimize" if is_softweights_cohort else "maximize"

    print(f"\n--- Running Pipeline for Model Type: **{model_type_str}** ---")
    print(f"Target Column(s): {config.target_col}")


    X_with_ids, X_train_full, y_train_full, X_test_final, y_test_final, numerical_cols, categorical_cols, feature_ranking, stratify_key = load_and_split_data(
      config = config,
      stratify_key = config.stratify_key)

    print(f"Stratification Key: {stratify_key}")
    print("\n--- Data Loaded and Split ---")

    OUTER_CV_SPLITS = config.outer_cv_splits
    RANDOM_STATE = config.random_state    
    N_TRIALS = config.n_trials
    OUTPUT_FOLD_RESULTS_FILE_PATH = config.output_fold_results_file_path
    ALL_OUTER_FOLD_MODELS_CLOUDPICKLE = config.all_outer_fold_models_cloudpickle # Only used by inference
    FEATURES_FILEPATH = config.features_filepath

    outer_skf = StratifiedKFold(n_splits=OUTER_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_results = pd.DataFrame()
    all_stratification_distributions = []  # To track stratification data across all folds
    pipeline_factory = EnhancedPipelineFactory(numerical_cols, categorical_cols)

    if is_softweights_cohort:
       stratify_target = y_train_full.idxmax(axis=1)
    else:
       # FIX: Use .squeeze() to ensure a Series is passed if y_train_full is a single-column DF
       stratify_target = y_train_full.squeeze() 

    if is_inference:
        os.makedirs(ALL_OUTER_FOLD_MODELS_CLOUDPICKLE, exist_ok=True)

    all_outer_fold_features = []
    outer_fold_summaries = []
    print(f"Starting nested cross-validation with {OUTER_CV_SPLITS} outer folds...")

    for fold, (train_idx, test_idx) in enumerate(outer_skf.split(X_train_full, stratify_target, stratify_key.loc[X_train_full.index])):
        print(f"Outer Fold {fold + 1}/{OUTER_CV_SPLITS}")

        # Split into train and test for this fold
        X_outer_train, y_outer_train = X_train_full.iloc[train_idx], y_train_full.iloc[train_idx]
        X_outer_test, y_outer_test = X_train_full.iloc[test_idx], y_train_full.iloc[test_idx]

        # Stratification keys for this fold
        stratify_key_outer_train = stratify_key.loc[X_outer_train.index]
        stratify_key_outer_test = stratify_key.loc[X_outer_test.index]
        train_dist_df = create_stratification_df(X_outer_train, stratify_key_outer_train, f"Outer Fold {fold + 1} Train", fold + 1)
        test_dist_df = create_stratification_df(X_outer_test, stratify_key_outer_test, f"Outer Fold {fold + 1} Test", fold + 1)

        all_stratification_distributions.append(train_dist_df)
        all_stratification_distributions.append(test_dist_df)


        objective = OptunaObjectiveEnhanced(
            X_outer_train, y_outer_train, stratify_key_outer_train,
            feature_ranking, numerical_cols, categorical_cols,
            train_data = None, train_labels = None,
            SHAP_model_name = None, model_factory = pipeline_factory
        )

        
        study = optuna.create_study(direction=optuna_direction,
                sampler=optuna.samplers.TPESampler(seed=config.random_state),
                pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=False)
        
        # Get best parameters and selected features
        best_params = study.best_params
        best_params_df = pd.DataFrame(best_params.items(), columns=["Parameter", "Value"])
        # Add N_trials as a new row
        n_trials_row = pd.DataFrame({"Parameter": ["N_trials"], "Value": [config.n_trials]})
        best_params_df = pd.concat([best_params_df, n_trials_row], ignore_index=True)

        print("\n--- Best Hyperparameters Found ---")
        print(best_params_df)

        best_num_features = best_params.get("num_features", len(feature_ranking))
        best_selected_features = feature_ranking[:best_num_features]
        all_outer_fold_features.append(best_selected_features)
        print("Best params:")
        print(best_params)
        print("features",best_selected_features)
        
        # Convert data to soft weights for cohorts
        if is_softweights_cohort:
           # Convert data
            train_data, train_labels, train_weights = convert_data_for_weighted_training(X_outer_train[best_selected_features], y_outer_train)
            if best_params.get("use_catboost_softweights", False) is True and not best_params.get("use_lgbm_softweights", False) and not best_params.get("use_brf_softweights", False):
                best_model_for_fold = build_final_catboost_softweights_model(pipeline_factory, best_params, best_selected_features, use_softweights=True, train_data=train_data, train_labels=train_labels, train_weights=train_weights)
            else:
                best_model_for_fold = pipeline_factory.build_final_model(best_params, best_selected_features, use_softweights=True)
                best_model_for_fold.fit(train_data, train_labels, sample_weight=train_weights)
        else:
            best_model_for_fold = pipeline_factory.build_final_model(best_params, best_selected_features)
            best_model_for_fold.fit(X_outer_train[best_selected_features], y_outer_train)
        
        if is_inference:
            
            model_file_path = os.path.join(ALL_OUTER_FOLD_MODELS_CLOUDPICKLE, f"model_fold_{fold}.pkl")
            try:
                with open(model_file_path, 'wb') as f:
                    cloudpickle.dump(best_model_for_fold, f)
                print(f"Model for Outer Fold {fold + 1} saved to {model_file_path}")
            except Exception as e:
                print(f"Error saving model for fold {fold + 1}: {e}")
        
        # --- 5. PREDICTION AND METRICS ---
        if is_inference:
            y_pred_outer_test = best_model_for_fold.predict(X_outer_test[best_selected_features])
            
            # FIX: Robustly get column names (Series vs DataFrame)
            if isinstance(y_outer_test, pd.Series):
                 column_names = [y_outer_test.name]
            else:
                 column_names = y_outer_test.columns
                 
            y_pred_outer_test_df = pd.DataFrame(y_pred_outer_test, columns=column_names, index=X_outer_test.index)
        else:
            y_pred_outer_test = best_model_for_fold.predict_proba(X_outer_test[best_selected_features])  # Get predicted probabilities
            y_pred_outer_test_df = pd.DataFrame(y_pred_outer_test, columns=y_outer_test.columns,index=X_outer_test.index)    
        
        fold_results = pd.concat(
            [fold_results, calculate_metrics_by_country(
                X=X_outer_test,
                y_real=y_outer_test,
                y_pred=y_pred_outer_test_df,
                dataset_name=f"Outer Fold {fold + 1}",
                model_type= model_type_str)],ignore_index=True)
                
        if is_inference:
            # Use Geometric Mean F1 for fold score
            fold_score = geometric_mean_f1_scorer(y_outer_test, y_pred_outer_test)
        else:
            fold_score = custom_log_loss(y_outer_test.values, y_pred_outer_test)

        outer_fold_summaries.append({
            'Fold': fold + 1,
            'Outer Train Rows': len(X_outer_train),
            'Outer Test Rows': len(X_outer_test),
            'Outer Fold Score': fold_score
        })

    # ---------------- 6. FINAL CONSOLIDATION & SAVE ----------------
    total_tuning_end_time = time.time()
    total_tuning_duration_seconds = total_tuning_end_time - total_tuning_start_time
    hours, rem = divmod(total_tuning_duration_seconds, 3600)
    minutes, seconds = divmod(rem, 60)

    print("\n--- Outer Cross-Validation Complete ---")

    with open(FEATURES_FILEPATH, 'w') as f:
        json.dump(all_outer_fold_features, f)
    print(f"Features for all {OUTER_CV_SPLITS} folds saved to {FEATURES_FILEPATH}")

    fold_metrics_df = pd.DataFrame(fold_results)      
    distributions_df = pd.concat(all_stratification_distributions)
    outer_fold_summaries_df = pd.DataFrame(outer_fold_summaries)

    # Export results to an Excel file with multiple sheets
    try:
        with pd.ExcelWriter(OUTPUT_FOLD_RESULTS_FILE_PATH, engine='openpyxl') as writer:
            fold_metrics_df.to_excel(writer, sheet_name="Fold Results", index=False)
            distributions_df.to_excel(writer, sheet_name='Stratification Distributions', index=False)
            outer_fold_summaries_df.to_excel(writer, sheet_name='Outerfold Distributions', index=False)
        print(f"Metrics, row counts, and distributions exported to {OUTPUT_FOLD_RESULTS_FILE_PATH}.")

    except Exception as e:
        print(f"Error exporting results to Excel: {e}")

    print(f"Total time taken for nested CV: {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s")

    return all_outer_fold_features



if __name__ == "__main__":
    run_nestedcv(inference_model_config)
    print("COHORT Nestedcv Starts")
    run_nestedcv(cohort_model_config) 
    print("\nOuter cross-validation results saved to OUTPUT_FOLD_RESULTS_FILE_PATH.")