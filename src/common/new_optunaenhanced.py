# need to change n_trails used for shap eda in cohort
"""optuna_enhanced.py
Generic, reusable Optuna hyper-parameter search harness that leverages
`EnhancedModelFactory` and `EnhancedPipelineFactory`.

The code is framework-agnostic: only a short Objective class is bound to
Optuna – the rest (search-space definition, evaluation logic, pipeline
creation) is plain Python and therefore unit-testable.

Run as a script or import its classes from elsewhere.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import os
import optuna
import numpy as np
from scipy.stats import gmean
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (f1_score, make_scorer,
                             accuracy_score, classification_report, confusion_matrix)

from sklearn.base import clone
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd 
import cloudpickle
import lightgbm as lgb
from optuna.integration import LightGBMPruningCallback
from src.common.data_preprocessing import convert_data_for_weighted_training  
from src.common.pipeline_factory_enhanced import EnhancedPipelineFactory
from src.common.model_factory_enhanced import EnhancedModelFactory
from src.common.metrics import geometric_mean_f1_scorer, custom_log_loss
from src.common.config import ModelConfig
from src.common.stratification_tools import create_stratification_df
from typing import Optional
from imblearn.ensemble import BalancedRandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBRFClassifier
from optuna.integration import XGBoostPruningCallback
#GM_F1_SCORER = make_scorer(geometric_mean_f1_scorer, greater_is_better=True)

# ---------------------------------------------------------------------------
# Search-space & helpers
# ---------------------------------------------------------------------------

class OptunaObjectiveEnhanced:
    """
    Optuna objective function for hyperparameter tuning with nested cross-validation.
    Supports both old models and new softweights models with custom loss functions.
    """

    def __init__(self, X_train, y_train, stratify_key, feature_ranking: List[str],
                 numerical_cols: List[str], categorical_cols: List[str],
                 config: Optional[ModelConfig] = None,
                 *,
                 random_state: Optional[int] = None,
                 inner_cv_splits: Optional[int] = None,
                 catboost_dir: Optional[str] = None,
                 inner_strat_output_path: Optional[str] = None, 
                 save_models: bool = False, 
                 dir_models: Optional[str] = None,
                 use_all_features: bool = False,
                 train_data: np.ndarray,            
                 train_labels: np.ndarray,          
                 train_weights: Optional[np.ndarray] = None,
                 X_val: Optional[pd.DataFrame] = None,    
                 y_val: Optional[pd.DataFrame] = None,
                 SHAP_model_name:str,
                 model_factory: EnhancedModelFactory,
                 optimize_for_inference: Optional[str] = None
                ):
        """
        Initialize Optuna objective.
        
        Args:
            X_train: Training features
            y_train: Training labels
            stratify_key: Stratification key
            feature_ranking: Ranked list of features
            numerical_cols: List of numerical column names
            categorical_cols: List of categorical column names
            config: ModelConfig object (preferred). If provided, all config values are used
                unless explicitly overridden by individual parameters.
            random_state: Random seed (overrides config.random_state if provided)
            inner_cv_splits: Number of inner CV splits (overrides config.inner_cv_splits if provided)
            catboost_dir: CatBoost directory (overrides config.catboost_dir if provided)
            inner_strat_output_path: Path to save inner CV stratification 
                (overrides config.inner_strat_output_path if provided)
            save_models: Whether to save fold models
            dir_models: Directory for saved models (overrides config.step2a_dir if provided)
            use_all_features: If True, use all features instead of feature selection (default: False)
        """
        # Initialize data and utilities
        self.factory = model_factory
        self.X = X_train  # Training features
        self.y = y_train  # Training labels (multi-class probabilities for softweights models)
        self.stratify_key = stratify_key  # Stratification key (e.g., combining class and country)
        self.feature_ranking = feature_ranking  # Ranked list of features for selection
        self.numerical_cols = numerical_cols  # List of numerical features
        self.categorical_cols = categorical_cols  # List of categorical features
        self.train_data = train_data
        self.train_labels = train_labels
        self.train_weights = train_weights
        self.X_val = X_val
        self.y_val = y_val
        self.SHAP_model_name = SHAP_model_name
        self.optimize_for_inference = optimize_for_inference
        
        # Store config object for future access
        self.config = config
        
        # Extract config values with override support (consistent pattern for all parameters)
        if config is not None:
            # Use explicit parameter if provided, otherwise use config value (allows override)
            self.random_state = random_state if random_state is not None else config.random_state
            self.inner_cv_splits = inner_cv_splits if inner_cv_splits is not None else config.inner_cv_splits
            self.catboost_dir = catboost_dir if catboost_dir is not None else config.catboost_dir
            inner_strat_output_path = inner_strat_output_path if inner_strat_output_path is not None else config.inner_strat_output_path
            dir_models = dir_models if dir_models is not None else config.step2a_dir
        else:
            # Use individual parameters with defaults
            self.random_state = random_state if random_state is not None else 42
            self.inner_cv_splits = inner_cv_splits if inner_cv_splits is not None else 5
            self.catboost_dir = catboost_dir
            # inner_strat_output_path and dir_models remain as passed (or None)
        
        # Detect structure of `y`: single-class or multi-class probabilities
        self.is_softweights = isinstance(self.y, pd.DataFrame) and self.y.shape[1] > 1

        # Dynamically configure model flags based on `y`
        if self.is_softweights:
            self.model_flags = {
                "use_lgbm_softweights": "lgbm_softweights",
                "use_brf_softweights": "brf_softweights",
                "use_catboost_softweights": "catboost_softweights",
            }
        else:
            self.model_flags = {
                "use_lgbm_base": "lgbm_base",
                "use_brf_base": "brf",
                "use_catboost_base": "catboost_base",
                "use_lgbm_wts" :"lgbm_wts",
                 "use_catboost_wts":"catboost_wts"
            }
        
        
        self.use_all_features = use_all_features
        self.space = SearchSpaceBuilder(feature_ranking, self.model_flags, random_state=self.random_state, use_all_features=use_all_features)  # Handles sampling hyperparameters and features
        # Pass config object - factory will extract what it needs
        self.pipeline_factory = EnhancedPipelineFactory(numerical_cols, categorical_cols, config=config)
        self.evaluator = NestedCVEvaluator(  # Handles inner cross-validation and evaluation
            inner_splits=self.inner_cv_splits,
            random_state=self.random_state,
            output_path=inner_strat_output_path,
            save_fold_models=save_models,
            dir_models=dir_models
        )

    def _SHAP_cohort_models(self, trial: optuna.Trial, SHAP_model_name: str, train_data: pd.DataFrame, train_labels: pd.DataFrame, 
                                 train_weights: np.ndarray, X_val: pd.DataFrame, y_val: pd.DataFrame, 
                                 class_mapping: list = None) -> float:
        """
        Objective function for tuning a single model (LGBM, BRF, or CatBoost) 
        using the dedicated sampler logic from SearchSpaceBuilder and a FIXED train/val split.
        """
        print(f"Running single-model SHAP objective for: {SHAP_model_name}")

        class_mapping_source = self.config.target_col
        X_train = self.train_data
        y_train = self.train_labels
        train_weights = self.train_weights
        X_val = self.X_val
        y_val = self.y_val # This is the soft-weights DataFrame

        # 1. Prepare Validation Data (Common to all models)
        X_val.columns = X_val.columns.str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)

        # Use the utility to get validation soft-weights DataFrame (for loss calculation) 
        # and integer labels (for model evaluation metrics/callbacks)
        y_val_df_for_loss, y_val_labels = self.factory.prepare_validation_data(
            y_val=y_val, # y_val contains the soft-weights DataFrame
            profile_names=class_mapping_source
        )

    # 2. Define Model-Specific Hyperparameters and Initialization

        if SHAP_model_name == "LGBM_Model":
            param_grid = {
                "objective": "multiclass",
                "num_class": len(np.unique(y_train)),
                "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"]),
                "num_leaves": trial.suggest_int("num_leaves", 10, 50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 30),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            }
            model = lgb.LGBMClassifier(**param_grid, random_state=self.config.random_state,verbosity=-1)

            # Fit with LightGBM-specific arguments (eval_set, callbacks)
            model.fit(
                X_train, y_train, sample_weight=train_weights,
                eval_set=[(X_val, y_val_labels)], # Use integer labels for eval_set
                eval_metric="multi_logloss",
                callbacks=[LightGBMPruningCallback(trial, "multi_logloss")]
            )

    
        elif SHAP_model_name == "BRF_Model":
            param_grid = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 10, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }
            model = BalancedRandomForestClassifier(
                **param_grid, 
                random_state=self.config.random_state )

            # Fit with standard scikit-learn fit
            model.fit(X_train, y_train, sample_weight=train_weights)

        # Correctly align elif with if
        elif SHAP_model_name == "CatBoost_Model":
            param_grid = {
                "iterations": trial.suggest_int("iterations", 100, 300),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                "border_count": trial.suggest_int("border_count", 10, 100),
            }
            #categorical_features = self.categorical_cols
            model = CatBoostClassifier(
                **param_grid,  
                loss_function="MultiClass",  
                verbose=0,  
                random_seed=self.config.random_state
            )
            model.fit(X_train, y_train, sample_weight=train_weights)
            
        elif SHAP_model_name == "XgBoost":
            print("Configuring XGBRFClassifier hyperparameters.")
            param_grid = {
                "objective": "multi:softprob",
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True), # L2 regularization
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "colsample_bynode": trial.suggest_float("colsample_bynode", 0.6, 1.0),
                 "num_parallel_tree" : trial.suggest_int("num_parallel_tree", 50, 300)

            }
 
            model = XGBRFClassifier(**param_grid,
                eval_metric="mlogloss",      
                num_class=len(class_mapping_source), 
                enable_categorical=True,
                n_jobs=-1,
                random_state=self.config.random_state
            )

            # Fit with standard 
            model.fit(X_train, y_train, sample_weight=train_weights,eval_set=[(X_val, y_val_labels)])

        else:
            raise ValueError(f"Unknown SHAP_model_name: {SHAP_model_name}")

        # 3. Evaluate the score (Common to all models)
        y_pred_proba = model.predict_proba(X_val)

        # Convert predictions to DataFrame using the columns/index from the formatted validation data
        y_pred_df = pd.DataFrame(y_pred_proba, columns=y_val_df_for_loss.columns, index=X_val.index)

        # Calculate custom log loss for evaluation using the formatted soft-weights DataFrame (y_val_df_for_loss)
        log_loss_value = custom_log_loss(y_val_df_for_loss.values, y_pred_df.values)

        return log_loss_value
      

    def _SHAP_inference_models(self, trial: optuna.Trial, SHAP_model_name: str, 
                             X_val: pd.DataFrame, y_val: np.ndarray) -> float:
        """
        Objective function for tuning models (LGBM, BRF, CatBoost, XGBRF) for 
        INFERENCE performance (Accuracy maximization).

        The function minimizes the Error Rate (1 - Accuracy).
        Assumes no sample weights (train_weights) are available.
        """
        print(f"Running single-model INFERENCE objective for: {SHAP_model_name}")

        class_mapping_source = self.config.target_col
        X_train = self.train_data # Assumed hard labeled data from self
        y_train = self.train_labels # Assumed hard labeled data from self
        X_val = self.X_val
        y_val = self.y_val 

        # Validation data preparation (Use hard labels directly)
        X_val.columns = X_val.columns.str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)
        y_val_labels_int = y_val

        # --- Prepare Base Fit Parameters ---
        # No sample_weight or train_weights used in this inference objective
        fit_params = {}
        # -----------------------------------

        # 2. Define Model-Specific Hyperparameters and Initialization

        if SHAP_model_name == "LGBM_Model":
            param_grid = {
                "objective": "multiclass",
                "num_class": len(np.unique(y_train)),
                "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"]),
                "num_leaves": trial.suggest_int("num_leaves", 10, 50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 30),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            }
            model = lgb.LGBMClassifier(**param_grid, random_state=self.config.random_state)

            # Add LGBM specific fit parameters
            fit_params.update({
                "eval_set": [(X_val, y_val_labels_int)], # Use integer labels
                "eval_metric": "multi_error", # Use error rate to optimize accuracy
                "callbacks": [LightGBMPruningCallback(trial, "multi_error")]
            })

            model.fit(X_train, y_train, **fit_params)

        elif SHAP_model_name == "BRF_Model":
            param_grid = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 10, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }
            model = BalancedRandomForestClassifier(
                **param_grid,
                random_state=self.config.random_state,
                sampling_strategy='all',
                replacement=True,
                n_jobs=-1
            )

            # Fit with standard scikit-learn fit (no eval_set/early_stopping)
            model.fit(X_train, y_train, **fit_params)

        elif SHAP_model_name == "CatBoost_Model":
            categorical_features_indices = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
            X_train_cb = X_train.copy()
            X_val_cb = X_val.copy()
            for col in categorical_features_indices:
                X_train_cb[col] = X_train_cb[col].astype(str).fillna("Missing")
                X_val_cb[col] = X_val_cb[col].astype(str).fillna("Missing")
            param_grid = {
                "iterations": trial.suggest_int("iterations", 100, 500),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                "border_count": trial.suggest_int("border_count", 10, 100),
            }
            model = CatBoostClassifier(
                **param_grid,
                loss_function="MultiClass", # CatBoost uses this for optimization
                eval_metric="Accuracy",       # Use accuracy for monitoring/pruning
                verbose=0,
                random_seed=self.config.random_state
            )

            # Add CatBoost specific fit parameters
            fit_params.update({
                "early_stopping_rounds": 20,
                "eval_set": (X_val_cb, y_val_labels_int), # CatBoost eval_set
                "cat_features": categorical_features_indices
            })

            model.fit(X_train_cb, y_train, **fit_params)

        elif SHAP_model_name == "XgBoost":
            print("Configuring XGBRFClassifier hyperparameters for MError (Inference).")
            param_grid = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "colsample_bynode": trial.suggest_float("colsample_bynode", 0.6, 1.0),
                "num_parallel_tree" : trial.suggest_int("num_parallel_tree", 50, 300)
            }

            model = XGBRFClassifier(
                **param_grid,
                objective="multi:softmax", # Outputs hard labels
                eval_metric="merror",      # MError (1-Accuracy) for internal eval/early stopping
                num_class=len(class_mapping_source),
                enable_categorical=True,
                n_jobs=-1,
                random_state=self.config.random_state
            )

            # Add XGBRF specific fit parameters
            eval_set_data = (X_val, y_val_labels_int)
            fit_params.update({
                "eval_set": [eval_set_data]
            })

            model.fit(X_train, y_train, **fit_params)

        else:
            raise ValueError(f"Unknown SHAP_model_name: {SHAP_model_name}")

        # 3. Evaluate the score (Common to all models)

        # Use model.predict() as models are optimized for hard classification
        y_pred_labels_int = model.predict(X_val)

        # Calculate the primary metric: Accuracy
        accuracy_value = accuracy_score(y_val_labels_int, y_pred_labels_int)

        # Calculate the Optuna return value: Error Rate (1 - Accuracy)
        # Optuna minimizes this, so minimizing Error Rate maximizes Accuracy
        error_rate = 1.0 - accuracy_value

        # Track Accuracy as a user attribute for viewing
        trial.set_user_attr("accuracy_score", accuracy_value)

        return error_rate

    def __call__(self, trial: optuna.trial.Trial):
        """
        Objective function for Optuna. Performs nested cross-validation and evaluates the custom loss.
        """
        # Sample hyperparameters and select features for the trial
        
        if self.SHAP_model_name:
            # Use the dedicated single-model objective for SHAP analysis
            if self.optimize_for_inference:
              return self._SHAP_inference_models(
                    trial=trial,
                    SHAP_model_name=self.SHAP_model_name,
                    X_val=self.X_val,
                    y_val=self.y_val, # CRITICAL: Assumes hard labels are stored here
                )
              
            else:
                # Call cohort/loss-focused method
                return self._SHAP_cohort_models(
                    trial=trial,
                    SHAP_model_name=self.SHAP_model_name,
                    train_data=self.train_data,
                    train_labels=self.train_labels,
                    train_weights=self.train_weights,
                    X_val=self.X_val,
                    y_val=self.y_val,
                )
        selected_features, model_map = self.space.sample(trial)
        print("model_map",model_map)
        if not model_map:
            return float("inf")  # No models selected, return a large loss

        # Build pipelines for the selected models      
        pipelines = []
        for model_name, params in model_map.items():
            #print(f"OptunaObjective Enhanced: __call__: Type of self.factory: {type(self.factory)}")  # Debug
            print(f"Building pipeline for model: {model_name}")
            # Handle special preprocessing for certain models (e.g., BRF with missing indicators)
            custom_pre = None
            if model_name in ["brf", "brf_softweights"]:  # Add missing indicators for BRF models
                custom_pre = self.factory._build_preprocessor(selected_features, add_missing_ind=True)

            # Build the pipeline for the model
            pipelines.append(self.factory.build_pipeline(
                model_name, params, selected_features, preprocessor=custom_pre
            ))

        # Delegate evaluation to NestedCVEvaluator
        meta_params = trial.params  # Contains meta-learner params for stacking (if applicable)
        score, strat_df = self.evaluator.evaluate(
            X=self.X[selected_features],  # Subset of features selected for the trial
            y=self.y,  # Multi-class probabilities or single-class labels
            stratify_key=self.stratify_key,  # Stratification key for cross-validation
            pipelines=pipelines,  # List of pipelines built for the trial
            meta_params=meta_params,  # Hyperparameters for meta-learner (if stacking is used)
            trial_number=trial.number  # Current trial number
        )

        # Save stratification data for this trial
        trial.set_user_attr("inner_cv_stratification", strat_df)

        # Return the mean score across inner folds
        return score
      
      

class NestedCVEvaluator:
    """
    Handles inner cross-validation and returns mean score and stratification data for given pipelines.
    """

    def __init__(self, inner_splits: int = 5, random_state: int = 42, output_path: str = None, save_fold_models: bool = False, dir_models: str = None):
        self.cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
        self.output_path = output_path
        self.save_fold_models = save_fold_models
        self.dir_models = dir_models
        self.random_state = random_state  # Store for meta-learner

    def evaluate(self, X, y, stratify_key: pd.Series, pipelines: List[Tuple[str, Any]],
                 meta_params: Dict[str, Any] | None, trial_number: int) -> Tuple[float, pd.DataFrame]:
        """
        Evaluate pipelines using nested cross-validation.

        Args:
            X: Features.
            y: Labels (multi-class probabilities or single-class labels).
            stratify_key: Stratification key for cross-validation.
            pipelines: List of pipelines to evaluate.
            meta_params: Hyperparameters for meta-learner (if stacking is used).
            trial_number: Current trial number for logging purposes.

        Returns:
            Tuple[float, pd.DataFrame]: Mean score across inner folds and stratification DataFrame.
        """
        scores = []  # Store scores for each fold
        stratification_data = []  # Track stratification data for debugging

        print(f"\n--- Starting CV ({self.cv.n_splits} splits) for Trial {trial_number} ---")
        
        # Reset indices of X, y, and stratify_key to ensure compatibility with StratifiedKFold
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        stratify_key = stratify_key.reset_index(drop=True)
        
        # Check if y is multi-column (softweights) or single-column (old models)
        if isinstance(y, pd.DataFrame) and y.shape[1] > 1:
            # Multi-column (softweights models): Get dominant class
            stratify_target = y.idxmax(axis=1)
        else:
            # Single-column (old models): Use y directly
            stratify_target = y
        
        # Check if we need stacking (multiple models selected)
        needs_stacking = len(pipelines) > 1
        is_softweights = any("softweights" in name for name, _ in pipelines)
        
        # If stacking is needed, evaluate on each fold
        if needs_stacking:
            # Separate CatBoost from other models
            catboost_pipelines = [(name, pipe) for name, pipe in pipelines if "catboost" in name]
            stacking_pipelines = [(name, pipe) for name, pipe in pipelines if "catboost" not in name]
            
            if catboost_pipelines and stacking_pipelines:
                # Manual stacking with CatBoost: train on each fold's training data
                # Uses nested CV: 5 outer folds × 5 inner folds = 25 model fits per trial
                for fold, (train_idx, val_idx) in enumerate(self.cv.split(X, stratify_target, stratify_key.loc[X.index])):
                    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Collect stratification data
                    train_strat_df = create_stratification_df(
                        X_train_fold, stratify_key.loc[train_idx], f"CV Train Fold {fold + 1}", fold + 1
                    )
                    val_strat_df = create_stratification_df(
                        X_val_fold, stratify_key.loc[val_idx], f"CV Validation Fold {fold + 1}", fold + 1
                    )
                    stratification_data.extend([train_strat_df, val_strat_df])
                    
                    # Convert training data for softweights if needed
                    if is_softweights:
                        train_data, train_labels, train_weights = convert_data_for_weighted_training(X_train_fold, y_train_fold)
                    else:
                        train_data, train_labels, train_weights = X_train_fold, y_train_fold, None
                    
                    # Use manual stacking (it will use CV internally for OOF predictions)
                    fold_score = self._manual_stacking_with_catboost(
                        catboost_pipelines, stacking_pipelines, 
                        train_data, train_labels, train_weights,
                        X_val_fold, y_val_fold, meta_params, is_softweights
                    )
                    
                    scores.append(fold_score)
                    print(f"  Fold {fold + 1} score: {fold_score:.6f}")
            else:
                # Standard StackingClassifier (no CatBoost): it handles CV internally
                # We still need to evaluate on each fold
                for fold, (train_idx, val_idx) in enumerate(self.cv.split(X, stratify_target, stratify_key.loc[X.index])):
                    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Collect stratification data
                    train_strat_df = create_stratification_df(
                        X_train_fold, stratify_key.loc[train_idx], f"CV Train Fold {fold + 1}", fold + 1
                    )
                    val_strat_df = create_stratification_df(
                        X_val_fold, stratify_key.loc[val_idx], f"CV Validation Fold {fold + 1}", fold + 1
                    )
                    stratification_data.extend([train_strat_df, val_strat_df])
                    
                    # Convert training data for softweights models if needed
                    if is_softweights:
                        train_data, train_labels, train_weights = convert_data_for_weighted_training(X_train_fold, y_train_fold)
                    else:
                        train_data, train_labels, train_weights = X_train_fold, y_train_fold, None
                    
                    # Evaluate models for this fold
                    fold_score = self._evaluate_models(
                        pipelines=pipelines,
                        train_data=train_data,
                        train_labels=train_labels,
                        train_weights=train_weights,
                        X_val=X_val_fold,
                        y_val=y_val_fold,
                        meta_params=meta_params,
                        is_softweights=is_softweights
                    )
                    
                    scores.append(fold_score)
        else:
            # Single model: evaluate on each fold (standard CV)
            for fold, (train_idx, val_idx) in enumerate(self.cv.split(X, stratify_target, stratify_key.loc[X.index])):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

                # Collect stratification data
                train_strat_df = create_stratification_df(
                    X_train_fold, stratify_key.loc[train_idx], f"CV Train Fold {fold + 1}", fold + 1
                )
                val_strat_df = create_stratification_df(
                    X_val_fold, stratify_key.loc[val_idx], f"CV Validation Fold {fold + 1}", fold + 1
                )
                stratification_data.extend([train_strat_df, val_strat_df])

                # Convert training data for softweights models if needed
                if is_softweights:
                    train_data, train_labels, train_weights = convert_data_for_weighted_training(X_train_fold, y_train_fold)
                else:
                    train_data, train_labels, train_weights = X_train_fold, y_train_fold, None

                # Evaluate models for this fold
                fold_score = self._evaluate_models(
                    pipelines=pipelines,
                    train_data=train_data,
                    train_labels=train_labels,
                    train_weights=train_weights,
                    X_val=X_val_fold,
                    y_val=y_val_fold,
                    meta_params=meta_params,
                    is_softweights=is_softweights
                )

                scores.append(fold_score)

        # Combine stratification data into a single DataFrame for debugging
        combined_strat_df = pd.concat(stratification_data, ignore_index=True) if stratification_data else pd.DataFrame()

        # Save stratification data if output path is provided
        if self.output_path:
            with pd.ExcelWriter(self.output_path) as writer:
                combined_strat_df.to_excel(writer, sheet_name="Inner CV Strat", index=False)

        # Return the mean score across folds and stratification DataFrame
        return np.mean(scores), combined_strat_df

    def _evaluate_models(self, pipelines, train_data, train_labels, train_weights=None, X_val=None, y_val=None, 
                         meta_params=None, is_softweights=False):
        """
        Evaluate the models for a given fold.
        
        If multiple models are selected, builds and evaluates a StackingClassifier.
        If only one model is selected, evaluates it directly.

        Args:
            pipelines (List[Tuple[str, Pipeline]]): List of model pipelines with names.
            train_data (np.ndarray): Training features.
            train_labels (np.ndarray): Training labels (single-label for old models, expanded for softweights).
            train_weights (np.ndarray, optional): Softweights for training data (used only for softweights models).
            X_val (pd.DataFrame, optional): Validation features.
            y_val (pd.DataFrame or np.ndarray, optional): Validation targets (multi-class probabilities for softweights).
            meta_params (Dict[str, Any], optional): Hyperparameters for meta-learner (if stacking is used).
            is_softweights (bool): Whether this is a softweights model (multi-target).

        Returns:
            float: The score (geometric_mean_f1_scorer for old models, custom_log_loss for softweights).
        """
        # If only one model is selected, evaluate it directly
        if len(pipelines) == 1:
            name, pipeline = pipelines[0]
            print(f"Fitting single model: {name}")

            if name == "catboost_softweights":
                # --- Special handling for CatBoostClassifier ---
                # Extract preprocessing step from the pipeline (e.g., `pre`)
                preprocessor = pipeline.named_steps["pre"]

                # Fit the preprocessing step and transform the training and validation data
                train_data_preprocessed = preprocessor.fit_transform(train_data)
                X_val_preprocessed = preprocessor.transform(X_val)

                # Extract the CatBoostClassifier from the pipeline
                catboost_model = pipeline.named_steps[name]

                # Fit the CatBoostClassifier directly with sample_weight
                catboost_model.fit(
                    train_data_preprocessed,
                    train_labels,
                    sample_weight=train_weights
                )

                # Predict probabilities for the validation set
                y_pred_proba = catboost_model.predict_proba(X_val_preprocessed)
                score = custom_log_loss(y_val.values, y_pred_proba)

            else:
                # --- For other models (e.g., scikit-learn models) ---
                if hasattr(pipeline, "fit"):
                    if "softweights" in name:  # Softweights models
                        if train_weights is None:
                            raise ValueError(f"train_weights must be provided for softweights models like {name}.")
                        pipeline.fit(train_data, train_labels, sample_weight=train_weights)
                    else:  # For old models
                        pipeline.fit(train_data, train_labels)

                # Predict probabilities for the validation set (softweights models)
                if "softweights" in name and hasattr(pipeline, "predict_proba"):
                    y_pred_proba = pipeline.predict_proba(X_val)
                    score = custom_log_loss(y_val.values, y_pred_proba)
                else:
                    # Predict single-class labels for old models
                    y_pred = pipeline.predict(X_val)
                    if is_softweights:
                      y_val_single = y_val.idxmax(axis=1)
                    else:
                        y_val_single = y_val

                    # Use geometric_mean_f1_scorer for old models
                    score = geometric_mean_f1_scorer(y_val_single, y_pred)

            return score

        # Multiple models selected: Build and evaluate StackingClassifier
        print(f"Building StackingClassifier with {len(pipelines)} base models")
        
        # Separate CatBoost models from other models
        # CatBoost cannot reliably receive sample_weight when inside Pipeline within StackingClassifier
        # (set_fit_request doesn't work reliably), so we handle it separately
        catboost_pipelines = [(name, pipe) for name, pipe in pipelines if "catboost" in name]
        stacking_pipelines = [(name, pipe) for name, pipe in pipelines if "catboost" not in name]
        
        # If CatBoost is selected with other models, use manual stacking
        # This ensures sample_weight works correctly for CatBoost while still including it in the ensemble
        if catboost_pipelines and stacking_pipelines:
            print(f"CatBoost ({[name for name, _ in catboost_pipelines]}) selected with other models.")
            print(f"  Using manual stacking to ensure sample_weight works correctly for CatBoost.")
            return self._manual_stacking_with_catboost(
                catboost_pipelines, stacking_pipelines, train_data, train_labels, 
                train_weights, X_val, y_val, meta_params, is_softweights
            )
        
        # Build and evaluate StackingClassifier with non-CatBoost models
        if len(stacking_pipelines) > 1:
            print(f"Building StackingClassifier with {len(stacking_pipelines)} scikit-learn compatible models")
            
            # Build meta-learner
            meta = self._build_meta_learner(meta_params, is_softweights, self.random_state)
            
            # Build StackingClassifier
            stacking_classifier = StackingClassifier(
                estimators=stacking_pipelines,  # List of (name, pipeline) tuples
                final_estimator=meta,
                cv=self.cv.n_splits,  # Use same CV splits as inner CV
                passthrough=False,
                n_jobs=-1,
            )
            
            # Fit the StackingClassifier
            if is_softweights and train_weights is not None:
                try:
                    stacking_classifier.fit(train_data, train_labels, sample_weight=train_weights)
                except TypeError:
                    print("Warning: StackingClassifier doesn't support sample_weight in this scikit-learn version. Fitting without weights.")
                    stacking_classifier.fit(train_data, train_labels)
            else:
                stacking_classifier.fit(train_data, train_labels)
            
            # Predict and score
            if is_softweights:
                y_pred_proba = stacking_classifier.predict_proba(X_val)
                score = custom_log_loss(y_val.values, y_pred_proba)
            else:
                y_pred = stacking_classifier.predict(X_val)
                if is_softweights:
                    y_val_single = y_val.idxmax(axis=1)
                else:
                    y_val_single = y_val
                score = geometric_mean_f1_scorer(y_val_single, y_pred)
            
            print(f"StackingClassifier score (without CatBoost): {score:.6f}")
            return score
        
        elif len(stacking_pipelines) == 1:
            # Only one non-CatBoost model, evaluate it directly
            name, pipeline = stacking_pipelines[0]
            print(f"Fitting single non-CatBoost model: {name}")
            
            if hasattr(pipeline, "fit"):
                if "softweights" in name:
                    if train_weights is None:
                        raise ValueError(f"train_weights must be provided for softweights models like {name}.")
                    pipeline.fit(train_data, train_labels, sample_weight=train_weights)
                else:
                    pipeline.fit(train_data, train_labels)
            
            # Predict and score
            if "softweights" in name and hasattr(pipeline, "predict_proba"):
                y_pred_proba = pipeline.predict_proba(X_val)
                score = custom_log_loss(y_val.values, y_pred_proba)
            else:
                y_pred = pipeline.predict(X_val)
                if is_softweights:
                    y_val_single = y_val.idxmax(axis=1)
                else:
                    y_val_single = y_val
                score = geometric_mean_f1_scorer(y_val_single, y_pred)
            
            print(f"Single model {name} score: {score:.6f}")
            return score
        
        else:
            # Only CatBoost models selected (no stacking possible)
            # This case should have been handled earlier (single model case), but handle it here for safety
            if catboost_pipelines:
                name, pipeline = catboost_pipelines[0]
                print(f"Fitting CatBoost model: {name}")
                
                # Extract preprocessing step
                preprocessor = pipeline.named_steps["pre"]
                
                # Fit preprocessing and transform data
                train_data_preprocessed = preprocessor.fit_transform(train_data)
                X_val_preprocessed = preprocessor.transform(X_val)
                
                # Extract CatBoost model
                catboost_model = pipeline.named_steps[name]
                
                # Fit CatBoost with sample_weight
                catboost_model.fit(
                    train_data_preprocessed,
                    train_labels,
                    sample_weight=train_weights
                )
                
                # Predict and score
                y_pred_proba = catboost_model.predict_proba(X_val_preprocessed)
                score = custom_log_loss(y_val.values, y_pred_proba)
                print(f"CatBoost model {name} score: {score:.6f}")
                return score
            
            raise ValueError("No models could be evaluated. Check model selection.")
    
    def _manual_stacking_with_catboost(self, catboost_pipelines, stacking_pipelines, train_data, train_labels,
                                       train_weights, X_val, y_val, meta_params, is_softweights):
        """
        Manually build stacking ensemble when CatBoost is included.
        
        This method correctly uses out-of-fold predictions for meta-learner training:
        1. Generates out-of-fold predictions from training data using 5-fold CV
        2. Trains meta-learner on out-of-fold predictions (prevents data leakage)
        3. Fits base models on full training data
        4. Gets base model predictions on validation set
        5. Evaluates ensemble on validation set
        
        Args:
            catboost_pipelines: List of (name, pipeline) tuples for CatBoost models
            stacking_pipelines: List of (name, pipeline) tuples for other models
            train_data: Training features
            train_labels: Training labels
            train_weights: Sample weights for training
            X_val: Validation features
            y_val: Validation targets
            meta_params: Hyperparameters for meta-learner
            is_softweights: Whether this is a softweights model
            
        Returns:
            float: The ensemble score (NOT averaged across models - this is the ensemble score)
        """
        from sklearn.model_selection import StratifiedKFold
        
        print(f"Manual stacking: {len(catboost_pipelines)} CatBoost model(s) + {len(stacking_pipelines)} other model(s)")
        
        # Reset indices to ensure compatibility with StratifiedKFold
        train_data_reset = train_data.reset_index(drop=True) if isinstance(train_data, pd.DataFrame) else train_data
        train_labels_reset = train_labels.reset_index(drop=True) if hasattr(train_labels, 'reset_index') else train_labels
        train_weights_reset = train_weights.reset_index(drop=True) if hasattr(train_weights, 'reset_index') else train_weights
        
        # Determine number of classes and stratification target
        if isinstance(train_labels_reset, pd.DataFrame):
            n_classes = train_labels_reset.shape[1]
            stratify_target = train_labels_reset.idxmax(axis=1)
        else:
            n_classes = len(np.unique(train_labels_reset))
            stratify_target = train_labels_reset
        
        n_samples = len(train_data_reset)
        
        # Initialize arrays for out-of-fold predictions
        # We reuse the same 5-fold CV structure for OOF predictions
        catboost_oof_predictions = [np.zeros((n_samples, n_classes)) for _ in catboost_pipelines]
        other_oof_predictions = [np.zeros((n_samples, n_classes))]
        
        # Use the same CV object (same n_splits, random_state) as the outer CV
        # This ensures consistency and allows us to reuse the same folds
        oof_cv = StratifiedKFold(
            n_splits=self.cv.n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        
        print(f"  Generating out-of-fold predictions using {oof_cv.n_splits}-fold CV")
        
        # Step 1: Generate out-of-fold predictions using CV on training data
        for fold, (train_idx, val_idx) in enumerate(oof_cv.split(train_data_reset, stratify_target)):
            print(f"    Processing fold {fold + 1}/{oof_cv.n_splits} for OOF predictions")
            
            # Split training data for this fold
            X_train_fold = train_data_reset.iloc[train_idx] if isinstance(train_data_reset, pd.DataFrame) else train_data_reset[train_idx]
            X_val_fold = train_data_reset.iloc[val_idx] if isinstance(train_data_reset, pd.DataFrame) else train_data_reset[val_idx]
            y_train_fold = train_labels_reset.iloc[train_idx] if hasattr(train_labels_reset, 'iloc') else train_labels_reset[train_idx]
            y_val_fold = train_labels_reset.iloc[val_idx] if hasattr(train_labels_reset, 'iloc') else train_labels_reset[val_idx]
            w_train_fold = None
            if train_weights_reset is not None:
               w_train_fold = train_weights_reset.iloc[train_idx] if hasattr(train_weights_reset, 'iloc') else train_weights_reset[train_idx]
            
            # Get CatBoost out-of-fold predictions (trained on fold train, predicted on fold val)
            for idx, (name, pipeline) in enumerate(catboost_pipelines):
                preprocessor = pipeline.named_steps["pre"]
                X_train_fold_preprocessed = preprocessor.fit_transform(X_train_fold)
                X_val_fold_preprocessed = preprocessor.transform(X_val_fold)
                
                catboost_model = pipeline.named_steps[name]
                catboost_model.fit(
                    X_train_fold_preprocessed,
                    y_train_fold,
                    sample_weight=w_train_fold
                )
                
                val_pred = catboost_model.predict_proba(X_val_fold_preprocessed)
                catboost_oof_predictions[idx][val_idx] = val_pred
            
            # Get other models' out-of-fold predictions
            if len(stacking_pipelines) > 1:
                # For other models, use StackingClassifier's internal logic
                # StackingClassifier would use cv internally, but we're already in a CV loop
                # So we fit directly on the fold training data and predict on fold validation data
                sub_meta = LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    multi_class="multinomial" if is_softweights else "ovr",
                    solver="lbfgs" if is_softweights else "liblinear",
                    class_weight="balanced",
                    random_state=self.random_state,
                )
                
                # Build a temporary StackingClassifier for this fold
                # Note: We use cv=2 to avoid nested CV, but this is just for the sub-ensemble
                # The main meta-learner will be trained on all OOF predictions
                sub_stacking = StackingClassifier(
                    estimators=stacking_pipelines,
                    final_estimator=sub_meta,
                    cv=2,  # Minimal CV to avoid excessive computation
                    passthrough=False,
                    n_jobs=-1,
                )
                
                if is_softweights and w_train_fold is not None:
                    try:
                        sub_stacking.fit(X_train_fold, y_train_fold, sample_weight=w_train_fold)
                    except TypeError:
                        sub_stacking.fit(X_train_fold, y_train_fold)
                else:
                    sub_stacking.fit(X_train_fold, y_train_fold)
                
                val_pred = sub_stacking.predict_proba(X_val_fold)
                other_oof_predictions[0][val_idx] = val_pred
                
            elif len(stacking_pipelines) == 1:
                name, pipeline = stacking_pipelines[0]
                if hasattr(pipeline, "fit"):
                    if "softweights" in name:
                        pipeline.fit(X_train_fold, y_train_fold, sample_weight=w_train_fold)
                    else:
                        pipeline.fit(X_train_fold, y_train_fold)
                
                if hasattr(pipeline, "predict_proba"):
                    val_pred = pipeline.predict_proba(X_val_fold)
                else:
                    y_pred = pipeline.predict(X_val_fold)
                    val_pred = np.zeros((len(y_pred), n_classes))
                    for i, pred in enumerate(y_pred):
                        val_pred[i, pred] = 1.0
                
                other_oof_predictions[0][val_idx] = val_pred
        
        # Step 2: Combine all out-of-fold predictions
        all_oof_predictions = np.hstack([np.hstack(catboost_oof_predictions), np.hstack(other_oof_predictions)])
        
        # Step 3: Train meta-learner on out-of-fold predictions
        print(f"  Training meta-learner on out-of-fold predictions")
        meta = self._build_meta_learner(meta_params, is_softweights, self.random_state)
        
        # Convert labels for meta-learner training
        if isinstance(train_labels_reset, pd.DataFrame):
            y_labels = train_labels_reset.idxmax(axis=1)
        else:
            y_labels = train_labels_reset
        
        meta.fit(all_oof_predictions, y_labels)
        
        # Step 4: Fit base models on full training data and get predictions on validation set
        catboost_val_predictions = []
        for name, pipeline in catboost_pipelines:
            preprocessor = pipeline.named_steps["pre"]
            train_data_preprocessed = preprocessor.fit_transform(train_data)
            X_val_preprocessed = preprocessor.transform(X_val)
            
            catboost_model = pipeline.named_steps[name]
            catboost_model.fit(
                train_data_preprocessed,
                train_labels,
                sample_weight=train_weights
            )
            
            y_pred_proba = catboost_model.predict_proba(X_val_preprocessed)
            catboost_val_predictions.append(y_pred_proba)
        
        other_val_predictions = []
        if len(stacking_pipelines) > 1:
            sub_meta = LogisticRegression(
                C=1.0,
                penalty="l2",
                multi_class="multinomial" if is_softweights else "ovr",
                solver="lbfgs" if is_softweights else "liblinear",
                class_weight="balanced",
                random_state=self.random_state,
            )
            
            sub_stacking = StackingClassifier(
                estimators=stacking_pipelines,
                final_estimator=sub_meta,
                cv=self.cv.n_splits,
                passthrough=False,
                n_jobs=-1,
            )
            
            if is_softweights and train_weights is not None:
                try:
                    sub_stacking.fit(train_data, train_labels, sample_weight=train_weights)
                except TypeError:
                    sub_stacking.fit(train_data, train_labels)
            else:
                sub_stacking.fit(train_data, train_labels)
            
            y_pred_proba = sub_stacking.predict_proba(X_val)
            other_val_predictions.append(y_pred_proba)
            
        elif len(stacking_pipelines) == 1:
            name, pipeline = stacking_pipelines[0]
            if hasattr(pipeline, "fit"):
                if "softweights" in name:
                    pipeline.fit(train_data, train_labels, sample_weight=train_weights)
                else:
                    pipeline.fit(train_data, train_labels)
            
            if hasattr(pipeline, "predict_proba"):
                y_pred_proba = pipeline.predict_proba(X_val)
            else:
                y_pred = pipeline.predict(X_val)
                y_pred_proba = np.zeros((len(y_pred), n_classes))
                for i, pred in enumerate(y_pred):
                    y_pred_proba[i, pred] = 1.0
            
            other_val_predictions.append(y_pred_proba)
        
        # Step 5: Combine validation predictions and evaluate
        all_val_predictions = np.hstack(catboost_val_predictions + other_val_predictions)
        
        # Get final predictions from meta-learner
        if is_softweights:
            y_pred_proba = meta.predict_proba(all_val_predictions)
            score = custom_log_loss(y_val.values, y_pred_proba)
        else:
            y_pred = meta.predict(all_val_predictions)
            if isinstance(y_val, pd.DataFrame): 
            # This assumes y_val is a DataFrame (one-hot or soft-weights)
                 y_val_single = y_val.idxmax(axis=1) 
            elif hasattr(y_val, "idxmax"):
            # If it's a Series, call idxmax without axis=1, but only if necessary (usually not needed for a Series)
            # However, since you're in a classification setting, y_val should already be the single label Series.
                 y_val_single = y_val
            else:
                 y_val_single = y_val
   
            score = geometric_mean_f1_scorer(y_val_single, y_pred)

        print(f"Manual stacking ensemble score: {score:.6f}")
        return score
    
    def _build_meta_learner(self, meta_params: Dict[str, Any] | None, is_softweights: bool, random_state: int = 42):
        """
        Build the meta-learner for StackingClassifier based on meta_params.
        
        Args:
            meta_params: Hyperparameters for meta-learner (from Optuna trial).
            is_softweights: Whether this is a softweights model.
            random_state: Random seed for reproducibility.
            
        Returns:
            Meta-learner classifier (LogisticRegression or LightGBM).
        """
        if meta_params is None:
            # Default meta-learner if no params provided
            if is_softweights:
                return LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    multi_class="multinomial",
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=random_state,
                )
            else:
                return LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=random_state,
                )
        
        # Extract meta-learner type
        meta_learner_type = meta_params.get("meta_learner", "logistic_regression")
        
        if is_softweights:
            if meta_learner_type == "lgbm_softweights":
                # LightGBM as meta-learner for softweights
                from src.common.pipeline_factory_enhanced import _extract_params_for
                # Extract meta-specific params (prefixed with "meta_")
                meta_lgbm_params = {
                    "n_estimators": meta_params.get("meta_lgbm_n_estimators", 100),
                    "learning_rate": meta_params.get("meta_lgbm_learning_rate", 0.1),
                    "num_leaves": meta_params.get("meta_lgbm_num_leaves", 31),
                    "num_class": 3,
                    "random_state": random_state,
                    "n_jobs": 1,
                }
                return EnhancedModelFactory.create_classifier("lgbm_softweights", meta_lgbm_params)
            else:
                # Logistic Regression for softweights models
                return LogisticRegression(
                    C=meta_params.get("lr_C", 1.0),
                    penalty=meta_params.get("lr_penalty", "l2"),
                    multi_class=meta_params.get("multi_class", "multinomial"),
                    solver=meta_params.get("solver", "lbfgs"),
                    class_weight="balanced",
                    random_state=random_state,
                )
        else:
            # Logistic Regression for standard models
            return LogisticRegression(
                C=meta_params.get("lr_C", 1.0),
                penalty=meta_params.get("lr_penalty", "l2"),
                solver="liblinear",
                class_weight="balanced",
                random_state=random_state,
            )
  
  
class SearchSpaceBuilder:
    """
    Encapsulates Optuna sampling logic for feature count and model hyperparameters.
    Supports both old models and new softweights models.
    """

    def __init__(self, feature_ranking: List[str], model_flags: Dict[str, str], random_state: int = 42, use_all_features: bool = False):
        self.feature_ranking = feature_ranking  # Ranked list of features
        self.model_flags = model_flags  # Dynamic model flags passed from OptunaObjectiveEnhanced
        self.random_state = random_state  # Random state for reproducibility
        self.use_all_features = use_all_features  # If True, use all features instead of feature selection

    def sample(self, trial: optuna.trial.Trial) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """
        Sample features and hyperparameters
        Args:
            trial (optuna.trial.Trial): Current Optuna trial.

        Returns:
            Tuple[List[str], Dict[str, Dict[str, Any]]]: Selected features and hyperparameters for the trial.
        """
        # ----- Feature Subset Sampling -----
        if self.use_all_features:
            # Use all features (no feature selection)
            num_features = len(self.feature_ranking)
            selected_features = self.feature_ranking[:num_features]
            # Still suggest num_features for consistency, but it will always be max
            trial.suggest_int("num_features", len(self.feature_ranking), len(self.feature_ranking))
        else:
            # Feature selection enabled
            num_features = trial.suggest_int("num_features", 5, len(self.feature_ranking))
            selected_features = self.feature_ranking[:num_features]

        # ----- Model Hyperparameter Sampling -----
        model_param_map: Dict[str, Dict[str, Any]] = {}
        for flag, model_name in self.model_flags.items():
            if trial.suggest_categorical(flag, [True, False]):  # Check if the model is selected
                print(f"Sampling model: {model_name}")  # Debug statement
                # Delegate hyperparameter sampling for each model
                sampler = getattr(self, f"_sample_{model_name}")
                model_param_map[model_name] = sampler(trial)        
        

        # If multiple models are selected, add meta-learner hyperparameters
        if len(model_param_map) > 1:  # Stacking required if multiple models are selected
            if any("softweights" in model_name for model_name in model_param_map.keys()):  # Check if softweights models are selected
            #if "softweights" in self.model_flags.values():  # Check if softweights models are included
                # Meta-learner for softweights models
                meta_learner = trial.suggest_categorical("meta_learner", ["logistic_regression", "lgbm_softweights"])
                if meta_learner == "logistic_regression":
                    # Logistic regression configured for 3 classes
                    trial.suggest_float("lr_C", 1e-4, 10.0, log=True)
                    trial.suggest_categorical("lr_penalty", ["l2"])
                    trial.suggest_categorical("multi_class", ["multinomial"])  # Ensure multi-class probabilities
                    trial.suggest_categorical("solver", ["lbfgs"])  # Solver compatible with multi-class classification
                elif meta_learner == "lgbm_softweights":
                    # LightGBM configured as meta-learner for softweights models
                    trial.suggest_float("meta_lgbm_learning_rate", 1e-3, 0.1, log=True)
                    trial.suggest_int("meta_lgbm_n_estimators", 50, 200)
                    trial.suggest_int("meta_lgbm_num_leaves", 20, 100)
            else:
                # Meta-learner for standard models
                trial.suggest_categorical("meta_learner", ["logistic_regression"])
                trial.suggest_float("lr_C", 1e-4, 10.0, log=True)
                trial.suggest_categorical("lr_penalty", ["l1", "l2"])
        return selected_features, model_param_map

    # ------------------------------------------------------------------
    # Individual Model Samplers
    # ------------------------------------------------------------------
    def _sample_lgbm_base(self, trial):
        return {
            "n_estimators": trial.suggest_int("lgbm_base_n_estimators", 100, 300),
            "learning_rate": trial.suggest_float("lgbm_base_learning_rate", 1e-3, 0.1, log=True),
            "num_leaves": trial.suggest_int("lgbm_base_num_leaves", 20, 100),
            "random_state": self.random_state,
            "n_jobs": 1,
        }
      
    def _sample_lgbm_wts(self, trial):
        return {
            "n_estimators": trial.suggest_int("lgbm_base_n_estimators", 100, 300),
            "learning_rate": trial.suggest_float("lgbm_base_learning_rate", 1e-3, 0.1, log=True),
            "num_leaves": trial.suggest_int("lgbm_base_num_leaves", 20, 100),
            "random_state": self.random_state,
            "n_jobs": 1,
            "auto_class_weights": "Balanced",  # Class weighting for imbalanced datasets
        }
      
    def _sample_lgbm_softweights(self, trial):
        return {
            #"objective": "multiclass",  # Softweights models use multiclass objective
            "num_class": 3,  # Number of classes for softweights models (e.g., 3Y)
            "n_estimators": trial.suggest_int("lgbm_softweights_n_estimators", 100, 300),
            "learning_rate": trial.suggest_float("lgbm_softweights_learning_rate", 1e-3, 0.1, log=True),
            "num_leaves": trial.suggest_int("lgbm_softweights_num_leaves", 20, 100),
            "random_state": self.random_state,
            "n_jobs": 1,
        }

    def _sample_brf(self, trial):
        return {
            "n_estimators": trial.suggest_int("brf_n_estimators", 100, 300),
            "max_depth": trial.suggest_int("brf_max_depth", 5, 20),
            "random_state": self.random_state,
            "n_jobs": 1,
        }

    def _sample_brf_softweights(self, trial):
        return {
            #"num_class": 3,
            "n_estimators": trial.suggest_int("brf_softweights_n_estimators", 100, 300),
            "max_depth": trial.suggest_int("brf_softweights_max_depth", 5, 20),
            "random_state": self.random_state,
            "n_jobs": 1,
        }

    def _sample_catboost_base(self, trial):
        return {
            "iterations": trial.suggest_int("catboost_base_iterations", 100, 300),
            "learning_rate": trial.suggest_float("catboost_base_learning_rate", 1e-3, 0.1, log=True),
            "depth": trial.suggest_int("catboost_base_depth", 4, 10),
            "random_state": self.random_state,
        }

    def _sample_catboost_softweights(self, trial):
        return {
            "iterations": trial.suggest_int("catboost_softweights_iterations", 100, 300),
            "learning_rate": trial.suggest_float("catboost_softweights_learning_rate", 1e-3, 0.1, log=True),
            "depth": trial.suggest_int("catboost_softweights_depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("catboost_softweights_l2_leaf_reg", 1e-3, 10.0, log=True),
            #"loss_function": "MultiClass",  # Softweights models use MultiClass loss
            "random_state": self.random_state,
        }

    def _sample_catboost_wts(self, trial):
        return {
            "iterations": trial.suggest_int("catboost_weights_iterations", 100, 300),
            "learning_rate": trial.suggest_float("catboost_weights_learning_rate", 1e-3, 0.1, log=True),
            "depth": trial.suggest_int("catboost_weights_depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("catboost_weights_l2_leaf_reg", 1e-3, 10.0, log=True),
            "auto_class_weights": "Balanced",  # Class weighting for imbalanced datasets
            "random_state": self.random_state,
        }
      
def optimize_model(model_name: str, objective_cls: OptunaObjectiveEnhanced, X_train, y_train, 
                   country_clean, train_data, train_labels, train_weights, X_val, y_val, 
                   n_trials: int, config: ModelConfig, factory: Any,optimize_for_inference: bool):
        """Generic function to run Optuna optimization for a single model."""
        objective = objective_cls(
            X_train, y_train, stratify_key=country_clean, feature_ranking=X_train.columns.tolist(),
            numerical_cols=[], categorical_cols=[], config=config, train_data=train_data,
            train_labels=train_labels, train_weights=train_weights, X_val=X_val, y_val=y_val,
            SHAP_model_name=model_name, model_factory=factory,
            optimize_for_inference=optimize_for_inference
        )
        study = optuna.create_study(
              direction="minimize", 
              sampler=optuna.samplers.TPESampler(seed=config.random_state),
              pruner=optuna.pruners.MedianPruner(),
              study_name=f"{model_name}_{'Inference' if optimize_for_inference else 'Cohort'}_Study"
          )
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

# Model-specific wrapper functions (using a fixed 50 trials as per your request)
def optimize_lgb_cohort(*args, **kwargs):
       return optimize_model("LGBM_Model",optimize_for_inference=False, *args, **kwargs)

def optimize_brf_cohort(*args, **kwargs):
       return optimize_model("BRF_Model",optimize_for_inference=False, *args, **kwargs)

def optimize_catboost_cohort(*args, **kwargs):
      return optimize_model("CatBoost_Model",optimize_for_inference=False, *args, **kwargs)    
  
def optimize_xgboost_cohort(*args, **kwargs):
    return optimize_model("XgBoost", optimize_for_inference=False,*args, **kwargs)
  
  
def optimize_lgb_inference(*args, **kwargs):
       return optimize_model("LGBM_Model", optimize_for_inference=True, *args, **kwargs)

def optimize_brf_inference(*args, **kwargs):
       return optimize_model("BRF_Model", optimize_for_inference=True,*args, **kwargs)

def optimize_catboost_inference(*args, **kwargs):
      return optimize_model("CatBoost_Model",optimize_for_inference=True, *args, **kwargs)    
  
def optimize_xgboost_inference(*args, **kwargs):
    return optimize_model("XgBoost",optimize_for_inference=True, *args, **kwargs)
  
  
  