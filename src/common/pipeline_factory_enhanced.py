"""pipeline_factory_enhanced.py
Enhanced pipeline factory to mirror naming of `model_factory_enhanced.py`.

This file is identical in behaviour to the previous `pipeline_factory.py`
but provides clearer naming consistency.  A backward-compatibility alias
(`PipelineFactory`) is kept at the end to avoid widespread refactors.
"""
from __future__ import annotations

from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore

from src.common.model_factory_enhanced import EnhancedModelFactory
from src.common.data_preprocessing import convert_data_for_weighted_training
from src.common.config import ModelConfig
from typing import Optional

def _add_missing_indicator(X):
    """Return 1 for NaNs and 0 otherwise."""
    return np.isnan(X).astype(int)

class EnhancedPipelineFactory:
    """Build preprocessing and model pipelines (enhanced version to support softweights)."""

    def __init__(self, numerical_cols: List[str], categorical_cols: List[str], 
                 config: Optional[ModelConfig] = None,
                 *,
                 random_state: Optional[int] = None,
                 inner_cv_splits: Optional[int] = None,
                 catboost_dir: Optional[str] = None):
        """
        Initialize the pipeline factory.
        
        Args:
            numerical_cols: List of numerical column names
            categorical_cols: List of categorical column names
            config: ModelConfig object (preferred). If provided, all config values are used
                unless explicitly overridden by individual parameters.
            random_state: Random seed (overrides config.random_state if provided)
            inner_cv_splits: Number of inner CV splits (overrides config.inner_cv_splits if provided)
            catboost_dir: CatBoost directory (overrides config.catboost_dir if provided)
        """
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        
        # Store config object for future access
        self.config = config
        
        # Extract config values with override support (consistent pattern for all parameters)
        if config is not None:
            # Use explicit parameter if provided, otherwise use config value (allows override)
            self.random_state = random_state if random_state is not None else config.random_state
            self.inner_cv_splits = inner_cv_splits if inner_cv_splits is not None else config.inner_cv_splits
            self.catboost_dir = catboost_dir if catboost_dir is not None else config.catboost_dir
        else:
            # Use individual parameters with defaults
            self.random_state = random_state if random_state is not None else 42
            self.inner_cv_splits = inner_cv_splits if inner_cv_splits is not None else 5
            self.catboost_dir = catboost_dir

      # Transformers for categorical and missing indicators
        self._categorical_transformer = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
        )
        self._missing_indicator = FunctionTransformer(_add_missing_indicator, feature_names_out="one-to-one")

    # ------------------------------------------------------------------
    # Pre-processing Builder
    # ------------------------------------------------------------------
    def _build_preprocessor(self, selected_features: List[str], add_missing_ind: bool = False) -> ColumnTransformer:
        """
        Build preprocessing pipeline for numerical and categorical features.
        """
        num_cols = [c for c in selected_features if c in self.numerical_cols]
        cat_cols = [c for c in selected_features if c in self.categorical_cols]

        transformers = [
            ("num", "passthrough", num_cols),
            ("cat", self._categorical_transformer, cat_cols),
        ]
        if add_missing_ind and num_cols:
            transformers.append(("num_nan_flags", self._missing_indicator, num_cols))
            
        return ColumnTransformer(transformers=transformers, remainder="drop",
                                verbose_feature_names_out=False)

    # ------------------------------------------------------------------
    # Pipeline Builder
    # ------------------------------------------------------------------
    def build_pipeline(
        self,
        model_name: str,
        params: Dict[str, Any] | None,
        selected_features: List[str],
        *,
        preprocessor: ColumnTransformer | None = None,
        train_dir: str | None = None,
        use_softweights: bool = False  # NEW ARGUMENT
    ) -> Tuple[str, Pipeline]:
        """
        Build and return a (name, pipeline) tuple.

        Args:
            model_name (str): Identifier for the model (e.g., "lgbm_base").
            params (Dict[str, Any] | None): Model hyperparameters.
            selected_features (List[str]): List of selected features to use.
            preprocessor (ColumnTransformer | None): Custom preprocessor, if provided.
            train_dir (str | None): Directory for training artifacts (used by CatBoost).
            use_softweights (bool): Whether this pipeline is for softweights models.

        Returns:
            Tuple[str, Pipeline]: A tuple of (model_name, pipeline).
        """

        # ------------------------------------------------------------------
        # Pre-processing
        # ------------------------------------------------------------------
        if preprocessor is None:
            add_missing = model_name in ["brf", "brf_softweights"]  # Add missing indicators for BRF if needed
            preprocessor = self._build_preprocessor(selected_features, add_missing_ind=add_missing)

        # ------------------------------------------------------------------
        # Classifier
        # ------------------------------------------------------------------
        # Create the classifier using EnhancedModelFactory
        #print(f"build_pipeline: Type of EnhancedModelFactory before create_classifier: {type(EnhancedModelFactory)}")  # Debug
        #print(f"Building pipeline for model: {model_name}")
        classifier = EnhancedModelFactory.create_classifier(model_name, params, train_dir=train_dir)

        # If using softweights, explicitly request sample_weight metadata for scikit-learn models
        #if use_softweights and model_name in ["lgbm_softweights", "brf_softweights"]:
        #    classifier.set_fit_request(sample_weight=True)
        
        # ------------------------------------------------------------------
        # Build the Pipeline
        # ------------------------------------------------------------------
        pipeline = ImbPipeline([
            ("pre", preprocessor),  # Preprocessing step
            (model_name, classifier)  # Classifier step
        ])
        
        # Note: We don't use set_fit_request() because it doesn't reliably work
        # for CatBoost in Pipeline within StackingClassifier. Instead, we handle
        # CatBoost separately by extracting it from the pipeline when needed.
        
        return model_name, pipeline

    # ------------------------------------------------------------------
    # Final Model Builder
    # ------------------------------------------------------------------
    def build_final_model(self, best_params: Dict[str, Any], all_features: List[str], use_softweights: bool = False):
      """
      Build the final model pipeline using the best parameters.

      Args:
          best_params (Dict[str, Any]): Hyperparameters for the final model.
          all_features (List[str]): List of all selected features for the final model.
          use_softweights (bool): Whether this final model is for softweights training.

      Returns:
          Pipeline or StackingClassifier: The final pipeline or stacking model.
      """
      # Define the flag-to-model mapping based on use_softweights
      if use_softweights:
          flag_to_model = {
              "use_lgbm_softweights": "lgbm_softweights",
              "use_brf_softweights": "brf_softweights",
              "use_catboost_softweights": "catboost_softweights",
          }
      else:
          flag_to_model = {
              "use_lgbm_base": "lgbm_base",
              "use_lgbm_wts": "lgbm_wts",
              "use_brf_base": "brf",
              "use_catboost_base": "catboost_base",
              "use_catboost_wts": "catboost_wts",
          }

      # Determine which base models are selected based on best_params
      base_models = [model_name for flag, model_name in flag_to_model.items() if best_params.get(flag)]
      if not base_models:
          raise ValueError("No base models selected in best_params.")

      # If only one base model is selected, return its pipeline directly
      if len(base_models) == 1:
          model_name = base_models[0]
          params = _extract_params_for(model_name, best_params)

          # Build the pipeline for the single base model
          _, pipeline = self.build_pipeline(model_name, params, all_features, use_softweights=use_softweights)
          
          # Special handling for CatBoostClassifier
          if model_name == "catboost_softweights":
              raise ValueError("catboost_softweights model cannot be estimated within the function, use build_final_catboost_softweights_model function")
          
          return pipeline

      # If multiple base models are selected, build a stacking classifier
      estimators = []
      for model_name in base_models:
          params = _extract_params_for(model_name, best_params)
          estimators.append(self.build_pipeline(model_name, params, all_features, use_softweights=use_softweights))

      # Define the meta-learner based on the model type
      if use_softweights:
          # Meta-learner for softweights models: Logistic Regression or LightGBM
          if best_params.get("meta_learner") == "lgbm_softweights":
              # Use LightGBM as meta-learner for softweights models
              meta = EnhancedModelFactory.create_classifier(
                  "lgbm_softweights",
                  _extract_params_for("lgbm_softweights", best_params),
              )
          else:
              # Logistic Regression for softweights models (configured for 3 probabilities)
              meta = LogisticRegression(
                  C=best_params.get("lr_C", 1.0),
                  penalty=best_params.get("lr_penalty", "l2"),  # Ensure compatible penalty
                  multi_class=best_params.get("multi_class", "multinomial"),  # Multi-class probabilities
                  solver=best_params.get("solver", "lbfgs"),  # Compatible solver
                  class_weight="balanced",
                  random_state=self.random_state,
              )
      else:
          # Logistic Regression for standard models
          meta = LogisticRegression(
              C=best_params.get("lr_C", 1.0),
              penalty=best_params.get("lr_penalty", "l2"),
              solver="liblinear",  # Solver for binary classification or one-vs-rest
              class_weight="balanced",
              random_state=self.random_state,
          )

      # Build the stacking classifier
      stacking_pipeline = StackingClassifier(
          estimators=estimators,  # Base models built in this trial
          final_estimator=meta,  # Dynamically selected meta-learner
          cv=self.inner_cv_splits,  # Inner cross-validation for stacking (use instance variable)
          passthrough=False,
          n_jobs=-1,
      )

      # Return the stacking pipeline without fitting (fitting happens externally)
      return stacking_pipeline
        
def _extract_params_for(model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract parameters for a specific model from the full parameter dictionary.

    Args:
        model_name (str): Name of the model (e.g., 'lgbm_base').
        params (Dict[str, Any]): Full dictionary of parameters.

    Returns:
        Dict[str, Any]: Parameters specific to the given model.
    """
    prefix = f"{model_name}_"
    return {k.replace(prefix, ""): v for k, v in params.items() if k.startswith(prefix)}

def build_final_catboost_softweights_model(pipeline_factory, best_params, selected_features, train_data, 
                                           train_labels, train_weights, catboost_dir: Optional[str] = None, 
                                           use_softweights: Bool = True):
    """
    Build and train the CatBoostClassifier for softweights models.

    Args:
        pipeline_factory: EnhancedPipelineFactory instance.
        best_params (Dict[str, Any]): Hyperparameters for the CatBoost model.
        selected_features (List[str]): List of selected features.
        train_data (pd.DataFrame): Training data (can be full DataFrame or already filtered to selected_features).
        train_labels (pd.Series): Training labels.
        train_weights (pd.Series): Sample weights for training data.
        catboost_dir (str, optional): Directory for CatBoost training artifacts.
            If None, uses pipeline_factory.catboost_dir.

    Returns:
        CatBoostClassifier: The trained CatBoost model.
    """
    # Extract preprocessing step
    preprocessor = pipeline_factory._build_preprocessor(selected_features)

    # Preprocess the training data
    # train_data should be a DataFrame (either full or already filtered to selected_features)
    # If it's a DataFrame with selected_features as columns, use them directly
    # Otherwise, assume it's already the correct format
    if not use_softweights:
          print("WARNING: build_final_catboost_softweights_model called with use_softweights=False.")                                 
    if isinstance(train_data, pd.DataFrame):
        if set(selected_features).issubset(set(train_data.columns)):
            train_data_preprocessed = preprocessor.fit_transform(train_data[selected_features])
        else:
            # train_data is already filtered to selected_features (expanded format from convert_data_for_weighted_training)
            train_data_preprocessed = preprocessor.fit_transform(train_data)
    else:
        # train_data is numpy array (shouldn't happen, but handle it)
        train_data_preprocessed = preprocessor.fit_transform(train_data)

    # Use provided catboost_dir or fall back to pipeline_factory's catboost_dir
    final_catboost_dir = catboost_dir if catboost_dir is not None else pipeline_factory.catboost_dir
    
    # Create the CatBoostClassifier directly
    catboost_model = EnhancedModelFactory.create_classifier(
        "catboost_softweights",
        _extract_params_for("catboost_softweights", best_params),
        train_dir=final_catboost_dir
    )

    # Fit the CatBoost model directly with sample weights
    catboost_model.fit(
        train_data_preprocessed,
        train_labels,
        sample_weight=train_weights
    )

    # Return the trained CatBoost model
    return catboost_model


class ManualStackingEnsemble:
    """
    A custom ensemble class that manually combines CatBoost with other models.
    This is needed because CatBoost cannot reliably receive sample_weight when inside
    a StackingClassifier's Pipeline.
    """
    def __init__(self, catboost_models, other_models, meta_learner, preprocessor):
        """
        Args:
            catboost_models: List of (name, fitted_catboost_model, preprocessor) tuples
            other_models: Fitted model (StackingClassifier or single model) for non-CatBoost models
            meta_learner: Fitted meta-learner
            preprocessor: Preprocessor for transforming input data
        """
        self.catboost_models = catboost_models  # List of (name, model, preprocessor) tuples
        self.other_models = other_models  # StackingClassifier or single model
        self.meta_learner = meta_learner
        self.preprocessor = preprocessor
    
    def predict_proba(self, X):
        """
        Get predictions from all models and combine them via meta-learner.
        
        Args:
            X: Input features (pd.DataFrame)
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        # Get CatBoost predictions
        catboost_predictions = []
        for name, catboost_model, catboost_preprocessor in self.catboost_models:
            X_preprocessed = catboost_preprocessor.transform(X)
            y_pred_proba = catboost_model.predict_proba(X_preprocessed)
            catboost_predictions.append(y_pred_proba)
        
        # Get other models' predictions
        if hasattr(self.other_models, 'predict_proba'):
            other_pred = self.other_models.predict_proba(X)
            other_predictions = [other_pred]
        else:
            # Fallback: convert predictions to probabilities
            y_pred = self.other_models.predict(X)
            n_classes = catboost_predictions[0].shape[1] if catboost_predictions else 3
            other_pred_proba = np.zeros((len(y_pred), n_classes))
            for i, pred in enumerate(y_pred):
                other_pred_proba[i, pred] = 1.0
            other_predictions = [other_pred_proba]
        
        # Combine all predictions
        all_predictions = np.hstack(catboost_predictions + other_predictions)
        
        # Get final predictions from meta-learner
        final_pred = self.meta_learner.predict_proba(all_predictions)
        
        # Debug check: warn if predictions are uniform
        if len(final_pred) > 1:
            pred_std = final_pred.std(axis=0)
            if np.all(pred_std < 1e-6):  # All predictions are essentially the same
                print(f"WARNING: Meta-learner predictions appear uniform (std per class: {pred_std})")
                print(f"  Sample predictions: {final_pred[:5]}")
                print(f"  Combined predictions stats: min={all_predictions.min():.4f}, max={all_predictions.max():.4f}, std={all_predictions.std():.4f}")
        
        return final_pred


def build_final_manual_stacking_model(pipeline_factory, best_params, selected_features, train_data, train_labels, train_weights, use_softweights=True):
    """
    Build final model using manual stacking when CatBoost is selected with other models.
    
    This function correctly uses out-of-fold predictions for meta-learner training:
    1. Generates out-of-fold predictions using cross-validation
    2. Trains meta-learner on out-of-fold predictions (prevents overfitting)
    3. Refits base models on full training data for final predictions
    4. Returns a ManualStackingEnsemble object
    
    Args:
        pipeline_factory: EnhancedPipelineFactory instance
        best_params: Best hyperparameters from Optuna
        selected_features: Selected features
        train_data: Training features (pd.DataFrame)
        train_labels: Training labels
        train_weights: Sample weights
        use_softweights: Whether this is a softweights model
        
    Returns:
        ManualStackingEnsemble: The fitted ensemble model
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import StackingClassifier
    from src.common.model_factory_enhanced import EnhancedModelFactory
    
    # Determine which models are selected
    if use_softweights:
        flag_to_model = {
            "use_lgbm_softweights": "lgbm_softweights",
            "use_brf_softweights": "brf_softweights",
            "use_catboost_softweights": "catboost_softweights",
        }
    else:
        flag_to_model = {
            "use_lgbm_base": "lgbm_base",
            "use_brf_base": "brf",
            "use_catboost_base": "catboost_base",
        }
    
    base_models = [model_name for flag, model_name in flag_to_model.items() if best_params.get(flag)]
    
    # Separate CatBoost from other models
    catboost_models = [m for m in base_models if "catboost" in m]
    other_models = [m for m in base_models if "catboost" not in m]
    
    if not catboost_models:
        raise ValueError("build_final_manual_stacking_model should only be called when CatBoost is selected")
    if not other_models:
        raise ValueError("build_final_manual_stacking_model should only be called when other models are also selected")
    
    print(f"Building manual stacking ensemble: {len(catboost_models)} CatBoost model(s) + {len(other_models)} other model(s)")
    
    # Prepare data for CV
    # Reset indices to ensure compatibility with StratifiedKFold
    train_data_reset = train_data.reset_index(drop=True) if isinstance(train_data, pd.DataFrame) else train_data
    train_labels_reset = train_labels.reset_index(drop=True) if hasattr(train_labels, 'reset_index') else train_labels
    train_weights_reset = train_weights.reset_index(drop=True) if hasattr(train_weights, 'reset_index') else train_weights
    
    # Determine number of classes for prediction arrays
    if isinstance(train_labels_reset, pd.DataFrame):
        n_classes = train_labels_reset.shape[1]
        stratify_target = train_labels_reset.idxmax(axis=1)
    else:
        n_classes = len(np.unique(train_labels_reset))
        stratify_target = train_labels_reset
    
    n_samples = len(train_data_reset)
    
    # Initialize arrays for out-of-fold predictions
    catboost_oof_predictions = [np.zeros((n_samples, n_classes)) for _ in catboost_models]
    other_oof_predictions = [np.zeros((n_samples, n_classes))]
    
    # Setup cross-validation for out-of-fold predictions
    cv = StratifiedKFold(
        n_splits=pipeline_factory.inner_cv_splits,
        shuffle=True,
        random_state=pipeline_factory.random_state
    )
    
    print(f"  Generating out-of-fold predictions using {cv.n_splits}-fold CV")
    
    # Step 1: Generate out-of-fold predictions using cross-validation
    for fold, (train_idx, val_idx) in enumerate(cv.split(train_data_reset, stratify_target)):
        print(f"    Processing fold {fold + 1}/{cv.n_splits} for out-of-fold predictions")
        
        # Split data for this fold
        X_train_fold = train_data_reset.iloc[train_idx][selected_features] if isinstance(train_data_reset, pd.DataFrame) else train_data_reset[train_idx]
        X_val_fold = train_data_reset.iloc[val_idx][selected_features] if isinstance(train_data_reset, pd.DataFrame) else train_data_reset[val_idx]
        y_train_fold = train_labels_reset.iloc[train_idx] if hasattr(train_labels_reset, 'iloc') else train_labels_reset[train_idx]
        y_val_fold = train_labels_reset.iloc[val_idx] if hasattr(train_labels_reset, 'iloc') else train_labels_reset[val_idx]
        w_train_fold = train_weights_reset.iloc[train_idx] if hasattr(train_weights_reset, 'iloc') else train_weights_reset[train_idx]
        
        # Get CatBoost out-of-fold predictions
        for idx, model_name in enumerate(catboost_models):
            # Build preprocessor for this fold
            catboost_preprocessor = pipeline_factory._build_preprocessor(selected_features)
            
            # Preprocess fold data
            X_train_fold_preprocessed = catboost_preprocessor.fit_transform(X_train_fold)
            X_val_fold_preprocessed = catboost_preprocessor.transform(X_val_fold)
            
            # Create and fit CatBoost model for this fold
            fold_catboost = EnhancedModelFactory.create_classifier(
                model_name,
                _extract_params_for(model_name, best_params),
                train_dir=pipeline_factory.catboost_dir
            )
            
            fold_catboost.fit(
                X_train_fold_preprocessed,
                y_train_fold,
                sample_weight=w_train_fold
            )
            
            # Get predictions on validation fold (out-of-fold)
            val_pred = fold_catboost.predict_proba(X_val_fold_preprocessed)
            catboost_oof_predictions[idx][val_idx] = val_pred
        
        # Get other models' out-of-fold predictions
        if len(other_models) > 1:
            # Multiple other models: use StackingClassifier
            sub_meta = LogisticRegression(
                C=1.0,
                penalty="l2",
                multi_class="multinomial" if use_softweights else "ovr",
                solver="lbfgs" if use_softweights else "liblinear",
                class_weight="balanced",
                random_state=pipeline_factory.random_state,
            )
            
            estimators = []
            for model_name in other_models:
                params = _extract_params_for(model_name, best_params)
                estimators.append(pipeline_factory.build_pipeline(model_name, params, selected_features, use_softweights=use_softweights))
            
            fold_stacking = StackingClassifier(
                estimators=estimators,
                final_estimator=sub_meta,
                cv=cv.n_splits,
                passthrough=False,
                n_jobs=-1,
            )
            
            if use_softweights and w_train_fold is not None:
                try:
                    fold_stacking.fit(X_train_fold, y_train_fold, sample_weight=w_train_fold)
                except TypeError:
                    fold_stacking.fit(X_train_fold, y_train_fold)
            else:
                fold_stacking.fit(X_train_fold, y_train_fold)
            
            val_pred = fold_stacking.predict_proba(X_val_fold)
            other_oof_predictions[0][val_idx] = val_pred
            
        elif len(other_models) == 1:
            # Single other model: fit directly
            model_name = other_models[0]
            params = _extract_params_for(model_name, best_params)
            _, fold_pipeline = pipeline_factory.build_pipeline(model_name, params, selected_features, use_softweights=use_softweights)
            
            if use_softweights and w_train_fold is not None:
                fold_pipeline.fit(X_train_fold, y_train_fold, sample_weight=w_train_fold)
            else:
                fold_pipeline.fit(X_train_fold, y_train_fold)
            
            val_pred = fold_pipeline.predict_proba(X_val_fold)
            other_oof_predictions[0][val_idx] = val_pred
    
    # Step 2: Combine all out-of-fold predictions
    all_oof_predictions = np.hstack([np.hstack(catboost_oof_predictions), np.hstack(other_oof_predictions)])
    print(f"    Combined out-of-fold predictions shape: {all_oof_predictions.shape}")
    print(f"    Combined predictions stats - min: {all_oof_predictions.min():.4f}, max: {all_oof_predictions.max():.4f}, mean: {all_oof_predictions.mean():.4f}, std: {all_oof_predictions.std():.4f}")
    
    # Step 3: Train meta-learner on out-of-fold predictions
    print("  Training meta-learner on out-of-fold predictions")
    
    # Convert labels for meta-learner training
    if isinstance(train_labels_reset, pd.DataFrame):
        y_labels = train_labels_reset.idxmax(axis=1)
    else:
        y_labels = train_labels_reset
    
    print(f"    Training labels distribution: {pd.Series(y_labels).value_counts().to_dict()}")
    
    # Build meta-learner
    if use_softweights:
        if best_params.get("meta_learner") == "lgbm_softweights":
            meta = EnhancedModelFactory.create_classifier(
                "lgbm_softweights",
                _extract_params_for("lgbm_softweights", best_params),
            )
        else:
            meta = LogisticRegression(
                C=best_params.get("lr_C", 1.0),
                penalty=best_params.get("lr_penalty", "l2"),
                multi_class=best_params.get("multi_class", "multinomial"),
                solver=best_params.get("solver", "lbfgs"),
                class_weight="balanced",
                random_state=pipeline_factory.random_state,
            )
    else:
        meta = LogisticRegression(
            C=best_params.get("lr_C", 1.0),
            penalty=best_params.get("lr_penalty", "l2"),
            solver="liblinear",
            class_weight="balanced",
            random_state=pipeline_factory.random_state,
        )
    
    # Fit meta-learner on out-of-fold predictions (prevents overfitting)
    meta.fit(all_oof_predictions, y_labels)
    
    # Debug: Check meta-learner predictions on out-of-fold data
    meta_oof_pred = meta.predict_proba(all_oof_predictions)
    print(f"    Meta-learner OOF predictions shape: {meta_oof_pred.shape}")
    print(f"    Meta-learner OOF predictions mean per class: {meta_oof_pred.mean(axis=0)}")
    print(f"    Meta-learner OOF predictions std per class: {meta_oof_pred.std(axis=0)}")
    if meta_oof_pred.std(axis=0).max() < 1e-6:
        print(f"    WARNING: Meta-learner predictions are uniform! This suggests the meta-learner isn't learning.")
        print(f"    Sample predictions: {meta_oof_pred[:5]}")
    
    # Step 4: Refit base models on full training data for final predictions
    print("  Refitting base models on full training data for final ensemble")
    
    # Refit CatBoost models on full data
    catboost_fitted = []
    preprocessor = pipeline_factory._build_preprocessor(selected_features)
    
    for model_name in catboost_models:
        print(f"    Refitting CatBoost model: {model_name}")
        
        catboost_preprocessor = pipeline_factory._build_preprocessor(selected_features)
        train_data_preprocessed = catboost_preprocessor.fit_transform(train_data[selected_features])
        
        catboost_model = EnhancedModelFactory.create_classifier(
            model_name,
            _extract_params_for(model_name, best_params),
            train_dir=pipeline_factory.catboost_dir
        )
        
        catboost_model.fit(
            train_data_preprocessed,
            train_labels,
            sample_weight=train_weights
        )
        
        catboost_fitted.append((model_name, catboost_model, catboost_preprocessor))
    
    # Refit other models on full data
    if len(other_models) > 1:
        print(f"    Refitting StackingClassifier with {len(other_models)} other models")
        
        sub_meta = LogisticRegression(
            C=1.0,
            penalty="l2",
            multi_class="multinomial" if use_softweights else "ovr",
            solver="lbfgs" if use_softweights else "liblinear",
            class_weight="balanced",
            random_state=pipeline_factory.random_state,
        )
        
        estimators = []
        for model_name in other_models:
            params = _extract_params_for(model_name, best_params)
            estimators.append(pipeline_factory.build_pipeline(model_name, params, selected_features, use_softweights=use_softweights))
        
        sub_stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=sub_meta,
            cv=pipeline_factory.inner_cv_splits,
            passthrough=False,
            n_jobs=-1,
        )
        
        if use_softweights and train_weights is not None:
            try:
                sub_stacking.fit(train_data[selected_features], train_labels, sample_weight=train_weights)
            except TypeError:
                sub_stacking.fit(train_data[selected_features], train_labels)
        else:
            sub_stacking.fit(train_data[selected_features], train_labels)
        
        other_fitted = sub_stacking
        
    elif len(other_models) == 1:
        model_name = other_models[0]
        print(f"    Refitting single other model: {model_name}")
        
        params = _extract_params_for(model_name, best_params)
        _, pipeline = pipeline_factory.build_pipeline(model_name, params, selected_features, use_softweights=use_softweights)
        
        if use_softweights and train_weights is not None:
            pipeline.fit(train_data[selected_features], train_labels, sample_weight=train_weights)
        else:
            pipeline.fit(train_data[selected_features], train_labels)
        
        other_fitted = pipeline
    
    # Step 5: Return ensemble with meta-learner trained on out-of-fold predictions
    return ManualStackingEnsemble(catboost_fitted, other_fitted, meta, preprocessor)


def build_and_fit_final_model(pipeline_factory, best_params, selected_features, train_data, train_labels, train_weights, use_softweights=True):
    """
    Build and fit the final model based on best hyperparameters from Optuna.
    
    This function automatically determines which model building function to use based on
    which models are selected in best_params:
    - Only CatBoost selected → uses build_final_catboost_softweights_model
    - Only LGBM or BRF selected → uses build_final_model with StackingClassifier
    - CatBoost + other models → uses build_final_manual_stacking_model
    - LGBM + BRF (no CatBoost) → uses build_final_model with StackingClassifier
    
    Args:
        pipeline_factory: EnhancedPipelineFactory instance
        best_params: Best hyperparameters from Optuna study
        selected_features: List of selected feature names
        train_data: Training features (pd.DataFrame)
        train_labels: Training labels
        train_weights: Sample weights for training
        use_softweights: Whether this is a softweights model
        
    Returns:
        Fitted model (CatBoostClassifier, Pipeline, StackingClassifier, or ManualStackingEnsemble)
    """
    # Determine which models are selected
    if use_softweights:
        has_catboost = best_params.get("use_catboost_softweights", False)
        has_lgbm = best_params.get("use_lgbm_softweights", False)
        has_brf = best_params.get("use_brf_softweights", False)
    else:
        has_catboost = best_params.get("use_catboost_base", False) or best_params.get("use_catboost_weights", False)
        has_lgbm = best_params.get("use_lgbm_base", False) or best_params.get("use_lgbm_weights", False)
        has_brf = best_params.get("use_brf_base", False)
    
    # Count selected models
    num_selected = sum([has_catboost, has_lgbm, has_brf])
    
    if num_selected == 0:
        raise ValueError("No models selected in best_params!")
    
    elif num_selected == 1:
        # Only one model selected
        if has_catboost:
            # Only CatBoost: use special function
            return build_final_catboost_softweights_model(
                pipeline_factory, best_params, selected_features,
                train_data, train_labels, train_weights
            )
        else:
            # Only LGBM or BRF: use standard build_final_model
            final_model = pipeline_factory.build_final_model(
                best_params, selected_features, use_softweights=use_softweights
            )
            final_model.fit(train_data[selected_features], train_labels, sample_weight=train_weights)
            return final_model
    
    else:
        # Multiple models selected
        if has_catboost:
            # CatBoost + other models: use manual stacking
            return build_final_manual_stacking_model(
                pipeline_factory, best_params, selected_features,
                train_data[selected_features], train_labels, train_weights, use_softweights=use_softweights
            )
        else:
            # Only LGBM + BRF (no CatBoost): use standard StackingClassifier
            final_model = pipeline_factory.build_final_model(
                best_params, selected_features, use_softweights=use_softweights
            )
            final_model.fit(train_data[selected_features], train_labels, sample_weight=train_weights)
            return final_model

