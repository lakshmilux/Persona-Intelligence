#--------------------COMMON CONFIG----------------
"""Unified configuration dataclass for all model types.

This module provides a ModelConfig dataclass that can be created from
any model-specific config module (digital_cohort, adoption_cohort, etc.)
"""
from dataclasses import dataclass
from typing import Optional, List, Union
import pandas as pd

@dataclass
class ModelConfig:
    """Unified configuration for all model types.
    
    This dataclass centralizes all configuration parameters needed for
    model training, evaluation, and prediction. It can be created from
    existing config modules using the `from_module()` classmethod.
    
    Attributes:
        target_col: List of target column names
        id_col: Name of the ID column
        data_path: Path to the main data file
        processesed_datapath_for_shap : path to the preprocessed data for shap
        shap_path: Path to the SHAP feature importance file
        ok_data_path: Path to the OK data file (optional)
        initial_models_evaluation : Path to the shap evaluation file
        shap_plot_path : path to the shap importance plot .png file
        random_state: Random seed for reproducibility
        test_size: Proportion of data to use for testing
        inner_cv_splits: Number of splits for inner cross-validation
        outer_cv_splits: Number of splits for outer cross-validation
        n_trials: Number of Optuna trials for hyperparameter optimization
        catboost_dir: Directory for CatBoost training artifacts (optional)
        rawtest_datapath: Path to save raw test data (optional)
        testdata_originalpred: Path to save test data with original predictions (optional)
        total_testdata_path: Path to save complete test dataset (optional)
        inner_strat_output_path: Path to save inner CV stratification data (optional)
        step2a_dir: Directory for step 2a hyperparameter optimization outputs (optional)
        cloudpickle_filename_step3: Path to save final model pickle file (optional)
        predictions_filename_step3: Path to save test predictions Excel file (optional)
        metrics_filename_step3: Path to save metrics Excel file (optional)
        ok_predictions_filename_step3: Path to save OK predictions CSV file (optional)
        stratify_key : Key for stratification
        output_fold_results_file_path: path to store fold results
        cv1_onekey_predictions: path to store OK predictions from model
        cv1_onekey_predictions_distribution : path to store OK prediction distributions from model
    """
    # Required fields
    target_col: Union[str, List[str]]
    id_col: str
    data_path: str
    processesed_datapath_for_shap:str
    shap_path: str
    
    # Optional fields with defaults
    ok_data_path: Optional[str] = None
    shap_plot_path : Optional[str] = None
    initial_models_evaluation: Optional[str] = None
    random_state: int = 42
    test_size: float = 0.2
    inner_cv_splits: int = 5
    outer_cv_splits: int = 5
    n_trials: int = 20
    stratify_key :  Optional[str] = None 
    
    # Optional paths
    catboost_dir: Optional[str] = None
    rawtest_datapath: Optional[str] = None
    testdata_originalpred: Optional[str] = None
    total_testdata_path: Optional[str] = None
    inner_strat_output_path: Optional[str] = None  # Path to save inner CV stratification data
    # Output paths (notebook-specific)
    
    step2a_dir: Optional[str] = None  # Directory for step 2a hyperparameter optimization
    cloudpickle_filename_step3: Optional[str] = None  # Path to save final model pickle
    predictions_filename_step3: Optional[str] = None  # Path to save test predictions
    metrics_filename_step3: Optional[str] = None  # Path to save metrics Excel file
    ok_predictions_filename_step3: Optional[str] = None  # Path to save OK predictions
    output_fold_results_file_path:Optional[str] = None
    all_outer_fold_models_cloudpickle:Optional[str] = None
    features_filepath:Optional[str] = None
    nestedcv_stratification:Optional[str] = None
    countrywise_perm_importance_dir:Optional[str] = None
    countrywise_perm_importance_filename:Optional[str] = None
    cv1_onekey_predictions:Optional[str] = None
    cv1_onekey_predictions_distribution:Optional[str]=None
    final_results:Optional[str]=None
     
    @classmethod
    def from_module(cls, config_module):
        """Create ModelConfig from existing config module.
        
        This method extracts configuration values from a config module
        (e.g., src.digital_cohort.config or src.adoption_cohort.config)
        and creates a ModelConfig instance.
        
        Usage:
            from src.adoption_cohort import config as adoption_config
            config = ModelConfig.from_module(adoption_config)
        
        Args:
            config_module: A config module with standard attributes like
                TARGET_COL, ID_COL, DATA_PATH, etc.
        
        Returns:
            ModelConfig: A configured ModelConfig instance
        """
        target_col_value = config_module.TARGET_COL
        if isinstance(target_col_value, str):
              final_target_col = target_col_value
        else:
              final_target_col = target_col_value    
        return cls(
            target_col = final_target_col,
            id_col=config_module.ID_COL,
            data_path=config_module.DATA_PATH,
            shap_path=config_module.SHAP_PATH,
            stratify_key=getattr(config_module, 'STRATIFY_KEY', None),
            ok_data_path=getattr(config_module, 'OK_DATA_PATH', None),
            processesed_datapath_for_shap = getattr(config_module,'PROCESSED_DATAPATH_FOR_SHAP',None),
            shap_plot_path = getattr(config_module,'SHAP_PLOT_PATH',None),
            initial_models_evaluation = getattr(config_module,'INITIAL_MODELS_EVALUATION',None),
            random_state=getattr(config_module, 'RANDOM_STATE', 42),
            test_size=getattr(config_module, 'TEST_SIZE', 0.2),
            inner_cv_splits=getattr(config_module, 'INNER_CV_SPLITS', 5),
            outer_cv_splits=getattr(config_module, 'OUTER_CV_SPLITS', 5),
            n_trials=getattr(config_module, 'N_TRIALS', 20),
            catboost_dir=getattr(config_module, 'CATBOOST_DIR', None),
            rawtest_datapath=getattr(config_module, 'RAWTEST_DATAPATH', None),
            testdata_originalpred=getattr(config_module, 'TESTDATA_ORIGINALPRED', None),
            total_testdata_path=getattr(config_module, 'TOTAL_TESTDATA', None),
            inner_strat_output_path=getattr(config_module, 'SINGLECV_STRATIFICATION', None),
            step2a_dir=getattr(config_module, 'step2a_dir', None),
            cloudpickle_filename_step3=getattr(config_module, 'CLOUDPICKLE_FILENAME_STEP3', None),
            predictions_filename_step3=getattr(config_module, 'PREDICTIONS_FILENAME_STEP3', None),
            metrics_filename_step3=getattr(config_module, 'METRICS_FILENAME_STEP3', None),
            output_fold_results_file_path = getattr(config_module, 'OUTPUT_FOLD_RESULTS_FILE_PATH', None),
            ok_predictions_filename_step3=getattr(config_module, 'OK_PREDICTIONS_FILENAME_STEP3', None),
            all_outer_fold_models_cloudpickle=getattr(config_module,'ALL_OUTER_FOLD_MODELS_CLOUDPICKLE', None),
            features_filepath=getattr(config_module, 'FEATURES_FILEPATH', None),
            nestedcv_stratification=getattr(config_module, 'NESTEDCV_STRATIFICATION', None),
            countrywise_perm_importance_dir=getattr(config_module,'COUNTRYWISE_PERM_IMPORTANCE_DIR',None),
            countrywise_perm_importance_filename = getattr(config_module,'COUNTRYWISE_PERM_IMPORTANCE_FILENAME',None),
            cv1_onekey_predictions = getattr(config_module,'CV1_ONEKEY_PREDICTIONS',None),
            cv1_onekey_predictions_distribution = getattr(config_module,'CV1_ONEKEY_PREDICTIONS_DISTRIBUTION',None), 
            final_results = getattr(config_module,'Final_Results',None)
        )

#from src.digital import config as inference_config
#from src.digital_cohort import config as cohort_config
#from src.common.config import ModelConfig
#config = ModelConfig.from_module(cohort_config)
#dp = pd.read_csv(config.ok_data_path)
#dp.columns

#config.countrywise_perm_importance_dir

