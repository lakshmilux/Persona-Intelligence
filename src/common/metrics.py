#import numpy as np  # For numerical operations
#from scipy.stats import gmean  # For geometric mean calculation
#from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score  # For classification metrics
#import pandas as pd  # For handling DataFrame operations
#from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#from typing import Tuple, Dict, Optional, List
#
### TO DO
### add function for customised log loss function
### calculate_fold_metrics -> in case of probability model report other metrics
#
#def geometric_mean_f1_scorer(y_true, y_pred, pos_label=None):
#    """
#    Calculate the geometric mean of F1 scores across all classes.
#    
#    This function computes F1 scores for each class and returns their geometric mean.
#    It handles both binary and multiclass classification scenarios, and properly
#    manages edge cases like missing classes or zero scores.
#    
#    The geometric mean of F1 scores provides a balanced measure for imbalanced datasets,
#    as it's sensitive to poor performance in any single class.
#    
#    Args:
#        y_true (array-like): Ground truth (correct) target values
#        y_pred (array-like): Estimated targets as returned by a classifier
#        pos_label (any, optional): Label of the positive class for binary classification.
#                                  If None, will be inferred from the data.
#    
#    Returns:
#        float: Geometric mean of F1 scores across all classes.
#               Returns 0.0 if any class has F1 score of 0 or if there are no valid scores.
#    
#    Notes:
#        - For binary classification, calculates F1 for both positive and negative classes
#        - For multiclass, calculates F1 for each class separately
#        - Uses zero_division=0 to handle cases with no predictions for a class
#        - Filters out NaN values before computing the geometric mean
#    """
#    y_true_np = np.asarray(y_true)
#    y_pred_np = np.asarray(y_pred)
#
#    unique_labels = np.unique(y_true_np)
#    num_unique_labels = len(unique_labels)
#
#    f1_scores_to_average = []
#
#    if num_unique_labels <= 2: # Treat as binary classification
#        effective_pos_label = pos_label
#        
#        # If pos_label is not explicitly provided, try to infer it for binary case
#        if effective_pos_label is None:
#            if num_unique_labels == 2:
#                # If there are two unique labels, pick the first one as the effective_pos_label
#                # The other will be the effective_neg_label. Order doesn't strictly matter for G-mean.
#                effective_pos_label = unique_labels[0]
#            elif num_unique_labels == 1:
#                # If only one unique label, that's the only 'positive' we can consider
#                effective_pos_label = unique_labels[0]
#            else: # Should not happen if num_unique_labels <= 2
#                print("Warning: Could not determine positive class label for binary case. Returning 0.0.")
#                return 0.0
#
#        # --- Calculate F1 for the designated positive class ---
#        if effective_pos_label not in unique_labels:
#            # This can happen if pos_label was provided but is not in the current fold's y_true
#            print(f"Warning: Designated pos_label '{effective_pos_label}' not found in y_true for this fold. "
#                  "F1 for this class will be 0.")
#            f1_for_pos = 0.0
#        else:
#            # Convert problem to binary: 'effective_pos_label' vs 'not effective_pos_label'
#            y_true_binary_pos = (y_true_np == effective_pos_label).astype(int)
#            y_pred_binary_pos = (y_pred_np == effective_pos_label).astype(int)
#            f1_for_pos = f1_score(y_true_binary_pos, y_pred_binary_pos, pos_label=1, average='binary', zero_division=0)
#        
#        f1_scores_to_average.append(f1_for_pos)
#
#        # --- Handle the "negative" (other) class for binary problems ---
#        other_labels = [label for label in unique_labels if label != effective_pos_label]
#        
#        if not other_labels:
#            return f1_for_pos
#
#        effective_neg_label = other_labels[0]
#
#        if effective_neg_label not in unique_labels: 
#             print(f"Warning: Designated neg_label '{effective_neg_label}' not found in y_true for this fold. "
#                   "F1 for this class will be 0.")
#             f1_for_neg = 0.0
#        else:
#            y_true_binary_neg = (y_true_np == effective_neg_label).astype(int)
#            y_pred_binary_neg = (y_pred_np == effective_neg_label).astype(int)
#            f1_for_neg = f1_score(y_true_binary_neg, y_pred_binary_neg, pos_label=1, average='binary', zero_division=0)
#        
#        f1_scores_to_average.append(f1_for_neg)
#
#    else: # Multiclass classification 
#        f1_per_class = f1_score(y_true_np, y_pred_np, labels=unique_labels, average=None, zero_division=0)
#        f1_scores_to_average.extend(f1_per_class)
#
#    # Filter out NaN values and ensure scores are non-negative
#    valid_f1_scores = [s for s in f1_scores_to_average if not np.isnan(s) and s >= 0]
#
#    # If any F1 score is 0 (due to zero_division=0 or actual 0 performance),
#    # or if there are no valid scores, the geometric mean should be 0.
#    if not valid_f1_scores or any(s == 0 for s in valid_f1_scores):
#        return 0.0
#    else:
#        return gmean(valid_f1_scores)
#      
#def calculate_fold_metrics(y_true, y_pred, dataset_name, model_type="classification"):
#    """
#    Calculate performance metrics for a dataset and return them as a dictionary.
#
#    Args:
#        y_true (pd.DataFrame or array-like): True labels or probabilities.
#        y_pred (pd.DataFrame or array-like): Predicted labels or probabilities.
#        dataset_name (str): Name or description of the dataset.
#        model_type (str): Type of model ("classification" or "softweights").
#
#    Returns:
#        dict: A dictionary containing dataset metrics.
#    """
#    if model_type == "classification":
#        # Handle old models (classification outputs)
#        y_true_np = np.asarray(y_true)
#        y_pred_np = np.asarray(y_pred)
#
#        # Calculate metrics for classification models
#        accuracy = accuracy_score(y_true_np, y_pred_np)
#        recall = recall_score(y_true_np, y_pred_np, average="weighted", zero_division=0)
#        precision = precision_score(y_true_np, y_pred_np, average="weighted", zero_division=0)
#        f1 = f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)
#        geom_avg_f1 = geometric_mean_f1_scorer(y_true_np, y_pred_np)
#        sample_size = len(y_true_np)
#
#        return {
#            "Dataset": dataset_name,
#            "Sample Size": sample_size,
#            "Accuracy": accuracy,
#            "Recall": recall,
#            "Precision": precision,
#            "F1-Score": f1,
#            "Geometric Mean F1": geom_avg_f1,
#        }
#
#    elif model_type == "softweights":
#        # Handle new models (softweights outputs with multi-class probabilities)
#        results = []
#
#        # Initialize accumulators for overall metrics
#        total_mae, total_mse, total_r2 = [], [], []
#
#        # Loop through profiles (columns in `y_true` and `y_pred`)
#        for profile in y_true.columns:
#            y_true_profile = y_true[profile]
#            y_pred_profile = y_pred[profile]
#
#            # Calculate metrics for the current profile
#            mae = mean_absolute_error(y_true_profile, y_pred_profile)
#            mse = mean_squared_error(y_true_profile, y_pred_profile)
#            r2 = r2_score(y_true_profile, y_pred_profile)
#            #log_loss_custom_value = custom_log_loss(y_true_profile.values, y_pred_profile.values)
#            sample_size= len(y_true[profile])
#            
#            # Append results for the current profile
#            results.append({
#                "Dataset": dataset_name,
#                "Profile": profile,
#                "MAE": mae,
#                "MSE": mse,
#                "R2": r2,
#                "Log Loss": pd.NA,
#                "sample_size": sample_size,
#            })
#
#            # Accumulate metrics for overall calculation
#            total_mae.append(mae)
#            total_mse.append(mse)
#            total_r2.append(r2)
#            
#        log_loss_custom_value = custom_log_loss(y_true, y_pred)
#        
#        # Add overall metrics across all profiles
#        results.append({
#            "Dataset": dataset_name,
#            "Profile": "All Profiles",
#            "MAE": np.mean(total_mae),
#            "MSE": np.mean(total_mse),
#            "R2": np.mean(total_r2),
#            "Log Loss": log_loss_custom_value,
#            "sample_size": sample_size,
#        })
#        #print("results:")
#        #print(results)
#
#        return pd.DataFrame(results)
#      
## In ~/src/common/metrics.py
#
#def calculate_metrics_by_country(X, y_real, y_pred, dataset_name, model_type="classification"):
#    """
#    Calculate metrics for the entire dataset and for each country separately.
#
#    Args:
#        X (pd.DataFrame): Dataset containing features. Should contain either OHE 'Country_'
#                          columns (for softweights) or a single 'country' column (for classification).
#        y_real (pd.DataFrame or array-like): True labels or probabilities for the dataset.
#        y_pred (pd.DataFrame or array-like): Predicted labels or probabilities for the dataset.
#        dataset_name (str): Name or description of the dataset.
#        model_type (str): Type of model ("classification" or "softweights").
#
#    Returns:
#        pd.DataFrame: A DataFrame containing metrics for the entire dataset and each country.
#    """
#    results = pd.DataFrame()
#    country_col_name = None 
#
#    # Handle country extraction based on model type
#    if model_type == "softweights":
#        # Dynamically derive the country column from one-hot encoded columns
#        country_columns = X.filter(like="Country_").columns
#        if country_columns.empty:
#            raise ValueError("DataFrame X must contain 'Country_' OHE columns for softweights models.")
#            
#        country = X[country_columns].idxmax(axis=1)  # Get the column name with the highest value
#        country = country.str.replace("Country_", "", regex=False)  # Remove "Country_" prefix
#        country_col_name = 'Country' # <--- NEW/FIX: Set a consistent name for the metrics column
#    else:
#        # For old models, use the existing 'Country' column (case-insensitive)
#        matching_cols = [col for col in X.columns if col.lower() == 'country'] # <--- REVISED: Case-insensitive match
#        
#        if not matching_cols:
#            raise ValueError("DataFrame X must contain a 'country' column (case-insensitive) for classification models.")
#        
#        # 2. Use the actual column name found (e.g., 'Country' or 'country')
#        country_col_name = matching_cols[0] # <--- REVISED: Store the actual column name
#        country = X[country_col_name]
#
#    # Calculate metrics for the entire dataset
#    total_metrics = calculate_fold_metrics(
#        y_true=y_real,
#        y_pred=y_pred,
#        dataset_name=f"{dataset_name}",
#        model_type=model_type  # Pass model type (classification or softweights)
#    )
#    
#    # <--- FIX: Ensure total_metrics is a DataFrame before concatenation
#    if isinstance(total_metrics, dict):
#        total_metrics = pd.DataFrame([total_metrics])
#        
#    total_metrics[country_col_name]="All"
#    results = pd.concat([results, total_metrics], ignore_index=True)
#
#    # Calculate metrics for each country separately
#    unique_countries = country.unique()
#    for country_name in unique_countries:
#        # Filter data for the current country
#        country_mask = country == country_name
#        
#        # Align country_mask with the indices of y_real and y_pred
#        country_mask = country_mask.reindex(y_real.index, fill_value=False)
#        
#        # Check if we have enough data for this country
#        if country_mask.sum() == 0:
#            print(f"Warning: No data found for country '{country_name}'. Skipping.")
#            continue
#            
#        y_country_real = y_real[country_mask]
#        y_country_pred = y_pred[country_mask]
#        
#        try:
#            # Calculate metrics for the current country
#            country_metrics = calculate_fold_metrics(
#                y_true=y_country_real,
#                y_pred=y_country_pred,
#                dataset_name=f"{dataset_name}",
#                model_type=model_type  # Pass model type
#            )
#            
#            # <--- FIX: Ensure country_metrics is a DataFrame before concatenation
#            if isinstance(country_metrics, dict):
#                country_metrics = pd.DataFrame([country_metrics])
#                
#            country_metrics[country_col_name]=country_name
#            results = pd.concat([results, country_metrics], ignore_index=True)
#        except ValueError as e:
#            print(f"Error calculating metrics for country '{country_name}': {e}")
#            continue
#
#    return results 
#  
#def custom_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#    """
#    Custom implementation of log loss for probability distributions.
#
#    Args:
#        y_true (np.ndarray): True probabilities (shape: [n_samples, n_classes]).
#        y_pred (np.ndarray): Predicted probabilities (shape: [n_samples, n_classes]).
#
#    Returns:
#        float: Log loss value.
#    """
#    # Add a small constant to avoid log(0)
#    epsilon = 1e-15
#    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Ensure predictions are in range [epsilon, 1 - epsilon]
#
#    # Compute log loss for each sample
#    log_loss_per_sample = -np.sum(y_true * np.log(y_pred), axis=1)  # Sum over classes for each sample
#
#    # Average over all samples
#    return np.mean(log_loss_per_sample)  
#                                 
#
#    
#def evaluate_models(y_true: pd.DataFrame, y_preds: Dict[str, pd.DataFrame]) -> pd.DataFrame:
#    """
#    Evaluate models using MAE, MSE, R2 scores, and Log Loss.
#    Returns a results DataFrame.
#
#    Args:
#        y_true (pd.DataFrame): True class probabilities (ground truth) as a DataFrame.
#        y_preds (Dict[str, pd.DataFrame]): Dictionary of predictions where keys are model names
#                                           and values are DataFrames of predicted probabilities.
#
#    Returns:
#        pd.DataFrame: DataFrame summarizing the evaluation metrics for each model.
#    """
#    results = []
#
#    for model_name, y_pred in y_preds.items():
#        # Calculate MAE for each class
#        mae = {f"MAE_{col}": mean_absolute_error(y_true[col], y_pred[col]) for col in y_true.columns}
#        
#        # Calculate MSE for each class
#        mse = {f"MSE_{col}": mean_squared_error(y_true[col], y_pred[col]) for col in y_true.columns}
#        
#        # Calculate R2 for each class
#        r2 = {f"R2_{col}": r2_score(y_true[col], y_pred[col]) for col in y_true.columns}
#                
#        # Calculate log loss using the custom implementation
#        log_loss_custom_value = custom_log_loss(y_true.values, y_pred.values)
#        
#        # Calculate mean metrics across all classes (profiles)
#        mean_mae = np.mean(list(mae.values()))
#        mean_mse = np.mean(list(mse.values()))
#        mean_r2 = np.mean(list(r2.values()))
#        
#        # Compile results into a dictionary
#        results.append({
#            "Model": model_name,
#            "Log Loss_all": log_loss_custom_value,
#            "MAE_all": mean_mae,   # Add mean MAE
#            "MSE_all": mean_mse,
#            "R2_all": mean_r2,
#            **mae,
#            **mse,
#            **r2
#        })
#
#    # Convert results to a DataFrame
#    return pd.DataFrame(results)
#

import numpy as np # For numerical operations
from scipy.stats import gmean # For geometric mean calculation
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score # For classification metrics
import pandas as pd # For handling DataFrame operations
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Dict, Optional, List, Any


# Assuming custom_log_loss is defined elsewhere in your module:
def custom_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Custom implementation of log loss for probability distributions."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    log_loss_per_sample = -np.sum(y_true * np.log(y_pred), axis=1)
    return np.mean(log_loss_per_sample)

# Assuming geometric_mean_f1_scorer is defined as provided previously:
def geometric_mean_f1_scorer(y_true, y_pred, pos_label=None):
    # (Implementation as provided in your prompt)
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)

    unique_labels = np.unique(y_true_np)
    num_unique_labels = len(unique_labels)

    f1_scores_to_average = []

    if num_unique_labels <= 2: 
        effective_pos_label = pos_label
        
        if effective_pos_label is None:
            if num_unique_labels == 2:
                effective_pos_label = unique_labels[0]
            elif num_unique_labels == 1:
                effective_pos_label = unique_labels[0]
            else:
                return 0.0

        if effective_pos_label in unique_labels:
            y_true_binary_pos = (y_true_np == effective_pos_label).astype(int)
            y_pred_binary_pos = (y_pred_np == effective_pos_label).astype(int)
            f1_for_pos = f1_score(y_true_binary_pos, y_pred_binary_pos, pos_label=1, average='binary', zero_division=0)
            f1_scores_to_average.append(f1_for_pos)

        other_labels = [label for label in unique_labels if label != effective_pos_label]
        
        if not other_labels:
            return f1_scores_to_average[0] if f1_scores_to_average else 0.0

        effective_neg_label = other_labels[0]

        if effective_neg_label in unique_labels:
            y_true_binary_neg = (y_true_np == effective_neg_label).astype(int)
            y_pred_binary_neg = (y_pred_np == effective_neg_label).astype(int)
            f1_for_neg = f1_score(y_true_binary_neg, y_pred_binary_neg, pos_label=1, average='binary', zero_division=0)
            f1_scores_to_average.append(f1_for_neg)

    else: # Multiclass classification 
        f1_per_class = f1_score(y_true_np, y_pred_np, labels=unique_labels, average=None, zero_division=0)
        f1_scores_to_average.extend(f1_per_class)

    valid_f1_scores = [s for s in f1_scores_to_average if not np.isnan(s) and s >= 0]

    if not valid_f1_scores or any(s == 0 for s in valid_f1_scores):
        return 0.0
    else:
        return gmean(valid_f1_scores)

# --- NEW HELPER FUNCTION FOR SOFTWEIGHTS METRICS ---
def _calculate_softweights_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Helper to calculate all softweights metrics (per-profile and overall) once.
    
    Returns: 
        Tuple: (overall_metrics_dict, profile_metrics_list)
    """
    total_mae, total_mse, total_r2 = [], [], []
    profile_results = []
    
    for profile in y_true.columns:
        y_true_profile = y_true[profile]
        y_pred_profile = y_pred[profile]

        # Calculate metrics for the current profile
        mae = mean_absolute_error(y_true_profile, y_pred_profile)
        mse = mean_squared_error(y_true_profile, y_pred_profile)
        r2 = r2_score(y_true_profile, y_pred_profile)

        # Store results for the long format
        profile_results.append({
            "Profile": profile,
            "MAE": mae,
            "MSE": mse,
            "R2": r2,
            "sample_size": len(y_true_profile),
        })
        
        # Accumulate metrics for overall calculation
        total_mae.append(mae)
        total_mse.append(mse)
        total_r2.append(r2)

    # Calculate overall metrics
    log_loss_custom_value = custom_log_loss(y_true.values, y_pred.values)
    mean_mae = np.mean(total_mae)
    mean_mse = np.mean(total_mse)
    mean_r2 = np.mean(total_r2)
    sample_size = len(y_true)

    # Compile the wide format metrics (for evaluate_models)
    wide_mae = {f"MAE_{col}": res['MAE'] for res in profile_results for col in y_true.columns if res['Profile'] == col}
    wide_mse = {f"MSE_{col}": res['MSE'] for res in profile_results for col in y_true.columns if res['Profile'] == col}
    wide_r2 = {f"R2_{col}": res['R2'] for res in profile_results for col in y_true.columns if res['Profile'] == col}

    overall_metrics_wide = {
        "Log Loss_all": log_loss_custom_value,
        "MAE_all": mean_mae,
        "MSE_all": mean_mse,
        "R2_all": mean_r2,
        **wide_mae,
        **wide_mse,
        **wide_r2,
    }
    
    # Compile the long format metrics (for calculate_fold_metrics)
    overall_metrics_long = {
        "Profile": "All Profiles",
        "MAE": mean_mae,
        "MSE": mean_mse,
        "R2": mean_r2,
        "Log Loss": log_loss_custom_value,
        "sample_size": sample_size
    }
    
    return overall_metrics_wide, profile_results + [overall_metrics_long]



  
def calculate_fold_metrics(y_true, y_pred, dataset_name, model_type="classification"):
    """
    Calculate performance metrics for a dataset and return them as a dictionary (classification) 
    or a DataFrame (softweights, long format).
    """
    if model_type == "classification":
        # (Classification metrics remain unchanged)
        y_true_np = np.asarray(y_true)
        y_pred_np = np.asarray(y_pred)

        accuracy = accuracy_score(y_true_np, y_pred_np)
        recall = recall_score(y_true_np, y_pred_np, average="weighted", zero_division=0)
        precision = precision_score(y_true_np, y_pred_np, average="weighted", zero_division=0)
        f1 = f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)
        geom_avg_f1 = geometric_mean_f1_scorer(y_true_np, y_pred_np)
        sample_size = len(y_true_np)

        return {
            "Dataset": dataset_name,
            "Sample Size": sample_size,
            "Accuracy": accuracy,
            "Recall": recall,
            "Precision": precision,
            "F1-Score": f1,
            "Geometric Mean F1": geom_avg_f1,
        }

    elif model_type == "softweights":
        _, results_long = _calculate_softweights_metrics(y_true, y_pred)
        
        # Add the dataset name to all rows
        for row in results_long:
            row['Dataset'] = dataset_name
            
        return pd.DataFrame(results_long)
    
    raise ValueError(f"Unknown model_type: {model_type}")



def evaluate_models(y_true: pd.DataFrame, y_preds: Dict[str, pd.DataFrame], evaluation_type: str = "softweights") -> pd.DataFrame:
    """
    Evaluate models using either hard label (classification) via calculate_fold_metrics 
    or softweights (regression) metrics.
    """
    results = []
    if evaluation_type == "softweights":
        metric_func = _calculate_softweights_metrics
    elif evaluation_type != "classification":
         raise ValueError(f"Unknown evaluation_type: {evaluation_type}. Must be 'softweights' or 'classification'.")

    for model_name, y_pred in y_preds.items():        
        if evaluation_type == "softweights":
            overall_metrics_wide, _ = metric_func(y_true, y_pred)
            final_metrics = {
                "Model": model_name,
                **overall_metrics_wide
            }
            
        elif evaluation_type == "classification":          
            shap_metrics = calculate_fold_metrics(
                y_true=y_true, 
                y_pred=y_pred, 
                dataset_name="Test Set", # Placeholder dataset name
                model_type="classification"
            )
            final_metrics = {
                "Model": model_name,
                "Sample Size": shap_metrics["Sample Size"],
                "Accuracy": shap_metrics["Accuracy"],
                "Recall": shap_metrics["Recall"],
                "Precision": shap_metrics["Precision"],
                "F1-Score": shap_metrics["F1-Score"],
                "Geometric Mean F1": shap_metrics["Geometric Mean F1"],
            }
        
        results.append(final_metrics)

    # 3. Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    results_df['Evaluation_Type'] = evaluation_type    
    return results_df

def calculate_metrics_by_country(X, y_real, y_pred, dataset_name, model_type="classification"):
    """
    Calculate metrics for the entire dataset and for each country separately.
    (Implementation remains as provided, ensuring proper DataFrame handling from calculate_fold_metrics)
    """
    results = pd.DataFrame()
    country_col_name = None 

    # Handle country extraction based on model type
    if model_type == "softweights":
        country_columns = X.filter(like="Country_").columns
        if country_columns.empty:
            raise ValueError("DataFrame X must contain 'Country_' OHE columns for softweights models.")
            
        country = X[country_columns].idxmax(axis=1) 
        country = country.str.replace("Country_", "", regex=False) 
        country_col_name = 'Country' 
    else:
        matching_cols = [col for col in X.columns if col.lower() == 'country'] 
        
        if not matching_cols:
            raise ValueError("DataFrame X must contain a 'country' column (case-insensitive) for classification models.")
        
        country_col_name = matching_cols[0] 
        country = X[country_col_name]

    # Calculate metrics for the entire dataset
    total_metrics = calculate_fold_metrics(
        y_true=y_real,
        y_pred=y_pred,
        dataset_name=f"{dataset_name}",
        model_type=model_type
    )
    
    # Ensure total_metrics is a DataFrame before concatenation
    if isinstance(total_metrics, dict):
        total_metrics = pd.DataFrame([total_metrics])
        
    total_metrics[country_col_name]="All"
    results = pd.concat([results, total_metrics], ignore_index=True)

    # Calculate metrics for each country separately
    unique_countries = country.unique()
    for country_name in unique_countries:
        # Filter data for the current country
        country_mask = country == country_name
        
        country_mask = country_mask.reindex(y_real.index, fill_value=False)
        
        if country_mask.sum() == 0:
            continue
            
        y_country_real = y_real[country_mask]
        y_country_pred = y_pred[country_mask]
        
        try:
            # Calculate metrics for the current country
            country_metrics = calculate_fold_metrics(
                y_true=y_country_real,
                y_pred=y_country_pred,
                dataset_name=f"{dataset_name}",
                model_type=model_type
            )
            
            # Ensure country_metrics is a DataFrame before concatenation
            if isinstance(country_metrics, dict):
                country_metrics = pd.DataFrame([country_metrics])
                
            country_metrics[country_col_name]=country_name
            results = pd.concat([results, country_metrics], ignore_index=True)
        except ValueError as e:
            print(f"Error calculating metrics for country '{country_name}': {e}")
            continue

    return results 

def generate_results_table(
    y_true_train: pd.DataFrame,
    y_pred_train: Dict[str, pd.DataFrame],
    country_train: pd.DataFrame,
    y_true_val: pd.DataFrame,
    y_pred_val: Dict[str, pd.DataFrame],
    country_val: pd.DataFrame,
    y_true_test: pd.DataFrame,
    y_pred_test: Dict[str, pd.DataFrame],
    country_test: pd.DataFrame,
    model_type: str = "softweights" 
) -> pd.DataFrame:
  
    """
     Aggregates metrics from train, validation, and test 
    sets, including country-wise breakdowns, into a single final DataFrame.
    """
    
    all_results = []
    
    # Process each model and dataset
    for model_name in y_pred_train.keys():
        
        # 1. Train Set Metrics
        train_results = calculate_metrics_by_country(
            X=country_train,
            y_real=y_true_train,
            y_pred=y_pred_train[model_name],
            dataset_name="Train",
            model_type=model_type
        )
        train_results['Model'] = model_name
        all_results.append(train_results)
        
        # 2. Validation Set Metrics
        val_results = calculate_metrics_by_country(
            X=country_val,
            y_real=y_true_val,
            y_pred=y_pred_val[model_name],
            dataset_name="Validation",
            model_type=model_type
        )
        val_results['Model'] = model_name
        all_results.append(val_results)
        
        # 3. Test Set Metrics
        test_results = calculate_metrics_by_country(
            X=country_test,
            y_real=y_true_test,
            y_pred=y_pred_test[model_name],
            dataset_name="Test",
            model_type=model_type
        )
        test_results['Model'] = model_name
        all_results.append(test_results)

    # Concatenate all results into a single DataFrame
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Reorder columns for presentation
    country_cols = [col for col in final_df.columns if col.lower() in ('country', 'country_col_name')]
    country_col = country_cols[0] if country_cols else 'Country' # Default to 'Country' if not found

    if model_type == "softweights":
        column_order = ['Model', 'Dataset', country_col, 'Profile', 'MAE', 'MSE', 'R2', 'Log Loss', 'sample_size']
    else: 
        column_order = ['Model', 'Dataset', country_col, 'Sample Size', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Geometric Mean F1']

    valid_cols = [col for col in column_order if col in final_df.columns]
    
    return final_df[valid_cols]
