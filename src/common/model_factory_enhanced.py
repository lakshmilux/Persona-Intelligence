import lightgbm as lgb
from imblearn.ensemble import BalancedRandomForestClassifier
from catboost import CatBoostClassifier
from typing import Dict, Any, Optional, Type, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
import pandas as pd
import numpy as np


class EnhancedModelFactory:
    """
    An enhanced factory to create different classifier models based on a given name and parameters.
    
    This implementation improves upon the basic ModelFactory by:
    1. Adding validation for required parameters
    2. Using configuration dictionaries instead of hard-coded parameters
    3. Supporting a registry pattern for extensibility
    4. Providing consistent parameter handling across model types
    5. Supporting default parameters
    """
    
    # Registry to store model creators
    _model_registry = {}
    
    # Default parameters for each model type
    _default_params = {
        #"lgbm_base": {"objective": "multiclass", "verbose": -1},
        #"lgbm_wts": {"objective": "multiclass", "class_weight": "balanced", "verbose": -1},
        "brf": {"verbose": 0},
        #"catboost_base": {"loss_function": "MultiClass", "verbose": 0},
        #"catboost_wts": {"loss_function": "MultiClass", "auto_class_weights": "Balanced", "verbose": 0}
        "lgbm_base": {"objective": "multiclassova", "verbose": -1}, 
        "lgbm_wts": {"objective": "multiclassova", "class_weight": "balanced", "verbose": -1},
        "catboost_base": {"loss_function": "MultiClassOneVsAll", "verbose": 0}, 
        "catboost_wts": {"loss_function": "MultiClassOneVsAll", "auto_class_weights": "Balanced", "verbose": 0},
        "lgbm_softweights": {"objective": "multiclass", "verbose": -1}, 
        "brf_softweights": {"verbose": 0}, 
        "catboost_softweights": {"loss_function": "MultiClass", "verbose": 0}
    }
    
    # Parameter keys to remove for each model type
    _params_to_remove = {
        "lgbm_base": ["objective", "class_weight", "verbose"],
        "lgbm_wts": ["objective", "class_weight", "verbose"],
        "brf": ["verbose"],
        "catboost_base": ["loss_function", "verbose", "train_dir"],
        "catboost_wts": ["loss_function", "auto_class_weights", "verbose", "train_dir"]
    }
    
    @classmethod
    def register_model(cls, name: str):
        """Decorator to register a new model creator function."""
        def decorator(func):
            cls._model_registry[name] = func
            return func
        return decorator
    
    def prepare_validation_data(self, y_val: pd.DataFrame, profile_names: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepares the already-split validation targets (y_val) into the two required formats:
        1. One-hot DataFrame for custom_log_loss (used as y_val in the objective).
        2. Integer-encoded array (generated for completeness, though recalculated by objective_lgb).

        Args:
            y_val (pd.DataFrame): Validation targets (soft-weights/normalized probabilities).
            profile_names (List[str]): List of all class names (columns).

        Returns:
            tuple: (y_val_df_for_loss, y_val_labels_int)

            y_val_df_for_loss (pd.DataFrame): Multi-class DataFrame (used as y_val in objective).
            y_val_labels_int (np.ndarray): Integer labels (optional/secondary).
        """

        # 1. One-hot/Soft-weights DataFrame for custom_log_loss.
        # Since your y_val is already a soft-weights DataFrame from the split, 
        # we just ensure column order and consistency. This serves as the 'y_val' input.
        y_val_df_for_loss = y_val.reindex(columns=profile_names, fill_value=0)

        # 2. Integer-encoded labels for LightGBM/CatBoost eval_set.
        # Note: This is redundant because your objective_lgb recalculates this using idxmax.
        # We generate it here based on the single-label derived from the soft-weights.
        y_val_single_label = y_val.idxmax(axis=1)
        class_to_int = {name: i for i, name in enumerate(profile_names)}
        y_val_labels_int = y_val_single_label.map(class_to_int).values

        return y_val_df_for_loss, y_val_labels_int
  
    # --- 🌟 END OF NEW METHOD 🌟 ---
    
    
    @classmethod
    def create_classifier(cls, name: str, params: Optional[Dict[str, Any]] = None, train_dir: Optional[str] = None):
        """
        Create a classifier model based on name and parameters.
        
        Args:
            name: The name of the classifier model to create
            params: Dictionary of parameters to pass to the model constructor
            train_dir: Optional directory for training artifacts (used by some models)
            
        Returns:
            An initialized classifier model
            
        Raises:
            ValueError: If the requested model name is not registered
            ValueError: If required parameters for the model are missing
        """
        

        # Use empty dict if params is None
        params = params or {}
        
        # Create a shallow copy to avoid modifying the original dictionary
        model_params = params.copy()
        
        # Check if the model exists in registry
        if name not in cls._model_registry:
            available_models = list(cls._model_registry.keys())
            raise ValueError(f"Unknown classifier: {name}. Available models: {available_models}")
        
        # Get default parameters for this model type
        default_params = cls._default_params.get(name, {})
        
        # Remove parameters that will be set explicitly
        for param in cls._params_to_remove.get(name, []):
            model_params.pop(param, None)
            
        # Log model creation
        #logger.info(f"Creating {name} classifier with parameters: {model_params}")
        
        # Call the registered model creator function
        return cls._model_registry[name](model_params, train_dir, default_params)


# Register model creators
@EnhancedModelFactory.register_model("lgbm_base")
def create_lgbm_base(params, train_dir, defaults):
    # Validate required parameters if needed
    return lgb.LGBMClassifier(**params, **defaults)


@EnhancedModelFactory.register_model("lgbm_wts")
def create_lgbm_wts(params, train_dir, defaults):
    return lgb.LGBMClassifier(**params, **defaults)
  

@EnhancedModelFactory.register_model("lgbm_softweights")
def create_lgbm_softweights(params, train_dir, defaults):
    """
    Create a LightGBM model for softweights training.
    """
    model = lgb.LGBMClassifier(**params, **defaults)
    model.set_fit_request(sample_weight=True)  # Explicitly request sample_weight metadata for pipeline compatibility
    return model  



@EnhancedModelFactory.register_model("brf")
def create_brf(params, train_dir, defaults):
    return BalancedRandomForestClassifier(**params, **defaults)
 

@EnhancedModelFactory.register_model("brf_softweights")
def create_brf_softweights(params, train_dir, defaults):
    """
    Create a BalancedRandomForest model for softweights training.
    """
    model = BalancedRandomForestClassifier(**params, **defaults)
    model.set_fit_request(sample_weight=True)  # Explicitly request sample_weight metadata for pipeline compatibility
    return model  




@EnhancedModelFactory.register_model("catboost_base")
def create_catboost_base(params, train_dir, defaults):
    #print(f"train_dir received: {train_dir}")
    # Include train_dir in the model parameters
    if train_dir:
        return CatBoostClassifier(**params, **defaults, train_dir=train_dir)
    return CatBoostClassifier(**params, **defaults)


@EnhancedModelFactory.register_model("catboost_wts")
def create_catboost_wts(params, train_dir, defaults):
    if train_dir:
        return CatBoostClassifier(**params, **defaults, train_dir=train_dir)
    return CatBoostClassifier(**params, **defaults)
  

@EnhancedModelFactory.register_model("catboost_softweights")
def create_catboost_softweights(params, train_dir, defaults):
    """
    Create CatBoostClassifier with sample_weight support.
    
    Note: CatBoostClassifier natively supports sample_weight in its fit() method.
    When used inside a Pipeline, metadata routing (set_fit_request) is handled
    at the Pipeline level, not at the individual model level.
    """
    model = CatBoostClassifier(**params, **defaults)
    # CatBoostClassifier supports sample_weight natively, no need for set_fit_request
    # Metadata routing is handled at the Pipeline level in build_pipeline()
    return model

