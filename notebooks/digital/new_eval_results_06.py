
import pandas as pd
import numpy as np
import os
import re


from src.common.utils import setup_project_paths
setup_project_paths()

sys.path.insert(0,os.path.abspath(os.path.join(setup_project_paths(), "src")))
from src.digital_cohort import config as cohort_config
from src.digital import config as inference_config
from src.common.config import ModelConfig

def load_all_excel_sheets(file_path: str) -> dict[str, pd.DataFrame]:
    """
    Loads all sheets from an Excel file into a dictionary of DataFrames.
    This function is used for dynamic loading (e.g., Permutation Importance file).

    Args:
        file_path (str): The path to the Excel file.

    Returns:
        dict[str, pd.DataFrame]: A dictionary where keys are sheet names and values are DataFrames.
    """
    try:
        # sheet_name=None loads all sheets
        return pd.read_excel(file_path, sheet_name=None)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return {}
    except Exception as e:
        print(f"An error occurred while reading file {file_path}: {e}")
        return {}


def load_excel_data(file_path: str, sheet_name: str) -> pd.DataFrame:
    """
    Loads a single Excel sheet into a pandas DataFrame.
    """
    try:
        return pd.read_excel(file_path, sheet_name=sheet_name)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while reading file {file_path}: {e}")
        return pd.DataFrame()


# The rest of the helper functions are included below for completeness:

def load_all_fold_metrics(config, base_dir: str, num_folds: int = 5) -> pd.DataFrame:
    """
    Loads and concatenates metrics from a specified number of fold subdirectories.
    """
    all_metrics_data = []
    OUTER_CV_SPLITS = config.outer_cv_splits
    print(f"\nLoading metrics from {OUTER_CV_SPLITS} folds in directory: {base_dir}")

    for i in range(1, OUTER_CV_SPLITS + 1):
        fold_folder_name = f"best_model_fold_{i}"
        metrics_filename = f"best_model_fold_{i}_metrics.xlsx"
        metrics_file_path = os.path.join(base_dir, fold_folder_name, metrics_filename)

        if os.path.exists(metrics_file_path):
            try:
                df = pd.read_excel(metrics_file_path, sheet_name='Train Fold Metrics')
                df['Fold'] = f"Fold_{i}"
                all_metrics_data.append(df)
            except Exception as e:
                print(f"❌ Could not read file {metrics_file_path}: {e}")

    if all_metrics_data:
        return pd.concat(all_metrics_data, ignore_index=True)
    else:
        return pd.DataFrame()

def process_metrics_dataframe(df: pd.DataFrame, pattern: str) -> pd.DataFrame:
    """
    Processes a metrics DataFrame to extract country/fold information and clean data.
    Modified to prioritize 'country' column if present.
    """
    if df.empty:
        return pd.DataFrame()

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    if 'country' in df.columns:
        # If 'country' column exists (like in your Fold Results sheet), use it directly
        df['Country'] = df['country']
        # Extract fold from the 'Dataset' column if needed
        extracted_fold = df['Dataset'].str.extract(r'Outer Fold (\d+)')
        if not extracted_fold.empty:
            df['Fold'] = extracted_fold[0]
        else:
            df['Fold'] = np.nan
        # Drop the original 'country' column to prevent duplicates, but keep 'Dataset'
        # df = df.drop(columns=['country']) # Optional, but generally cleaner

    
    else:
        extracted = df['Dataset'].str.extract(pattern)
        if extracted.shape[1] == 2:
            df[['Fold', 'Country']] = extracted
        else:
            df['Country'] = extracted[0]
            df['Fold'] = np.nan

    # This cleaning step is CRITICAL for merging
    # Convert to lowercase and strip leading/trailing spaces/parentheses
    df['Country'] = df['Country'].str.strip('() ').str.lower()

    return df
  
def pivot_and_merge_cv_results(df_single_cv: pd.DataFrame, df_nested_cv: pd.DataFrame) -> pd.DataFrame:
    """
    Pivots and merges the single and nested CV results.
    """
    if df_single_cv.empty or df_nested_cv.empty:
        return pd.DataFrame()

    pivoted_single = df_single_cv.pivot_table(
        index='Country',
        columns='Fold',
        values='Geometric Mean F1',
        aggfunc='mean'
    ).add_prefix('Single CV OOF Train Score Fold_').reset_index()

    pivoted_nested = df_nested_cv.pivot_table(
        index='Country',
        columns='Fold',
        values='Geometric Mean F1',
        aggfunc='mean'
    ).add_prefix('Nested CV OOF Train Score Fold_').reset_index()

    return pd.merge(pivoted_single, pivoted_nested, on='Country', how='inner')

def merge_step3_data(config,df_test: pd.DataFrame, df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the step 3 test and train metrics.
    """
    is_softweights_cohort = isinstance(config.target_col, list)
    print("soft", is_softweights_cohort)
    if is_softweights_cohort:
      
        if df_test.empty or df_train.empty:
             return pd.DataFrame()
        else:
              merged_df = pd.merge(
                    df_test[['Country', 'Log Loss']],
                    df_train[['Country', 'Log Loss']],
                    on='Country',how='inner' )

              merged_df.rename(
                    columns={
                    'Geometric Mean F1_x': 'Log Loss_TestData(20%)',
                    'Geometric Mean F1_y': 'Log Loss_TrainData(80%)'},inplace=True)
      
    else:    
        if df_test.empty or df_train.empty:
             return pd.DataFrame()
        else:
            merged_df = pd.merge(
                   df_test[['Country', 'Geometric Mean F1']],
                   df_train[['Country', 'Geometric Mean F1']],
                    on='Country',how='inner')

            merged_df.rename(columns={
            'Geometric Mean F1_x': 'Geometric Mean F1_TestData(20%)',
            'Geometric Mean F1_y': 'Geometric Mean F1_TrainData(80%)'
             },inplace=True)

    return merged_df


def merge_prediction_data(test_df: pd.DataFrame, onekey_df: pd.DataFrame, is_countrywise: bool = False) -> pd.DataFrame:
    """
    Merges 'test data' and 'onekey' prediction DataFrames.
    For countrywise merge, the merge is on 'Country' column.
    For overall/class-based merge (is_countrywise=False), it merges on the class labels.
    """
    if test_df is None or onekey_df is None or test_df.empty or onekey_df.empty:
        print("One or both prediction distribution DataFrames are missing or empty. Skipping prediction merge.")
        return pd.DataFrame()

    merge_col = 'Country' # Default for countrywise
    
    if is_countrywise:
        # Standardize 'Country' column name for merge
        if 'country' in test_df.columns and 'Country' not in test_df.columns:
            test_df = test_df.rename(columns={'country': 'Country'})
        if 'country' in onekey_df.columns and 'Country' not in onekey_df.columns:
            onekey_df = onekey_df.rename(columns={'country': 'Country'})

        if merge_col not in test_df.columns or merge_col not in onekey_df.columns:
            print(f"Required '{merge_col}' column is missing for countrywise merge. Skipping.")
            return pd.DataFrame()
        
        # Merge logic for countrywise data
        test_cols_map = {col: f'{col}_TestData' for col in test_df.columns if col != merge_col}
        onekey_cols_map = {col: f'{col}_OneKeyData' for col in onekey_df.columns if col != merge_col}

        test_renamed_df = test_df.rename(columns=test_cols_map)
        onekey_renamed_df = onekey_df.rename(columns=onekey_cols_map)

        merged_df = pd.merge(
            test_renamed_df,
            onekey_renamed_df,
            on=merge_col,
            how='inner'
        )
        # Sort columns to have 'Country' first
        cols = [merge_col] + [col for col in merged_df.columns if col != merge_col]
        return merged_df[cols]

    else:
        # 💡 MERGE LOGIC FOR PREDICTION DISTRIBUTION (BASED ON CLASS LABELS) 💡
        # Assuming the structure is [Class Column 1, Percentage Column 1] and [Class Column 2, Percentage Column 2]

        # 1. Rename the first column of each DataFrame to a common key for merging.
        common_class_col = 'Prediction_Class'
        
        # Rename the class columns to the common key
        test_df = test_df.rename(columns={test_df.columns[0]: common_class_col})
        onekey_df = onekey_df.rename(columns={onekey_df.columns[0]: common_class_col})
        
        # 2. Rename the percentage columns to distinguish them in the final merged DataFrame.
        test_df = test_df.rename(columns={test_df.columns[1]: f'{test_df.columns[1]}_TestData'})
        onekey_df = onekey_df.rename(columns={onekey_df.columns[1]: f'{onekey_df.columns[1]}_OneKeyData'})
        
        # 3. Perform the merge on the common class column.
        merged_df = pd.merge(
            test_df,
            onekey_df,
            on=common_class_col,
            how='inner' # Use inner merge to only include classes present in both
        )
        
        # Sort columns to have the class label first
        cols = [common_class_col] + [col for col in merged_df.columns if col != common_class_col]
        return merged_df[cols]


# --- Main execution logic ---

def store_results(config_module):
    """
    Main function to orchestrate the data loading, processing, and merging,
    and writing to multiple sheets in the final Excel file.
    """
    # --- Step 1: Load and process step 3 data ---
    config = ModelConfig.from_module(config_module)
    print("TARGET_COL",config.target_col)
    METRICS_FILENAME_STEP3 = config.metrics_filename_step3
    OUTER_CV_SPLITS = config.outer_cv_splits
    OUTPUT_FOLD_RESULTS_FILE_PATH = config.output_fold_results_file_path
    COUNTRYWISE_PERM_IMPORTANCE_DIR = config.countrywise_perm_importance_dir
    COUNTRYWISE_PERM_IMPORTANCE_FILENAME = config.countrywise_perm_importance_filename
    CV1_ONEKEY_PREDICTIONS_DISTRIBUTION = config.cv1_onekey_predictions_distribution
    step2a_dir = config.step2a_dir
    Final_Results = config.final_results
    
    step3a_test_metrics = load_excel_data(METRICS_FILENAME_STEP3, 'Country test metrics')
    step3a_train_metrics = load_excel_data(METRICS_FILENAME_STEP3, 'Country train metrics')

    processed_test_metrics = process_metrics_dataframe(step3a_test_metrics, r'Test Data(.*)')
    processed_train_metrics = process_metrics_dataframe(step3a_train_metrics, r'Train full Data(.*)')

    step3_df = merge_step3_data(config, processed_test_metrics, processed_train_metrics)

    # --- Step 2: Load and process step 1 and 2 CV data ---
    step2a_metrics_df = load_all_fold_metrics(config, step2a_dir, num_folds=OUTER_CV_SPLITS)
    step1_outerfold_results = load_excel_data(OUTPUT_FOLD_RESULTS_FILE_PATH, 'Fold Results')

    processed_step2a = process_metrics_dataframe(step2a_metrics_df, r'Train Fold (\d+) (.*)')
    processed_step1 = process_metrics_dataframe(step1_outerfold_results, r'Outer Fold (\d+)\s*\((.*?)\)')
    
    print(f"--- Debugging df2 Sources ---")
    print(f"processed_step2a empty: {processed_step2a.empty}")
    print(f"processed_step1 empty: {processed_step1.empty}")
    if not processed_step2a.empty:
         print(f"processed_step2a columns: {processed_step2a.columns.tolist()}")
         print(f"processed_step2a 'Country' sample: {processed_step2a['Country'].head().tolist()}")
    if not processed_step1.empty:
         print(f"processed_step1 columns: {processed_step1.columns.tolist()}")
         print(f"processed_step1 'Country' sample: {processed_step1['Country'].head().tolist()}")
    print(f"-----------------------------")

    # --- Step 3: Pivot and merge the CV results ---
    df2 = pivot_and_merge_cv_results(processed_step2a, processed_step1)

    # --- Step 4: Load and Process Additional Data ---
    print("\nLoading additional data for new Excel tabs...")

    # New Tab 2 Data: Final Features from Model
    final_features_df = load_excel_data(METRICS_FILENAME_STEP3, 'Final_Features')
    if not final_features_df.empty and 'Unnamed: 0' in final_features_df.columns:
        final_features_df = final_features_df.drop(columns=['Unnamed: 0'])
        
    classification_report = load_excel_data(METRICS_FILENAME_STEP3, 'Classification Report')
    classification_report_df = classification_report.reset_index()
    if 'Unnamed: 0' in classification_report_df.columns:
         classification_report_df.rename(columns={'Unnamed: 0': 'metrics'}, inplace=True)
    elif classification_report_df.columns[0] != 'metrics':
         classification_report_df.rename(columns={classification_report_df.columns[0]: 'metrics'}, inplace=True)
    print(classification_report_df.columns)
    
    if not classification_report_df.empty and 'index' in classification_report_df.columns:
        classification_report_df = classification_report_df.drop(columns=['index'])

    print(classification_report_df.head())
    # New Tabs: Countrywise Permutation Importance (DYNAMICALLY LOAD ALL SHEETS)
    perm_importance_file_path = os.path.join(COUNTRYWISE_PERM_IMPORTANCE_DIR, COUNTRYWISE_PERM_IMPORTANCE_FILENAME)
    perm_importance_dfs = load_all_excel_sheets(perm_importance_file_path)

    # Clean up 'Unnamed: 0' column from all loaded permutation importance DataFrames
    cleaned_perm_importance_dfs = {}
    print(f"Attempting to process {len(perm_importance_dfs)} sheet(s) loaded from permutation importance file...")
    for sheet_name, df in perm_importance_dfs.items():
        if not df.empty:
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
                
            new_sheet_name = f"shap_{sheet_name}"
            cleaned_perm_importance_dfs[new_sheet_name] = df
            print(f"✅ Loaded and processed permutation importance sheet: '{sheet_name}'")
        else:
            print(f"⚠️ Skipping permutation importance sheet: '{sheet_name}' (Data is empty).")


    # --- Step 4a: Load and Merge Prediction Distribution Data (MODIFIED LOGIC) ---
    print("\nLoading and merging Prediction Distribution data...")
    predictions_dist_dfs = load_all_excel_sheets(CV1_ONEKEY_PREDICTIONS_DISTRIBUTION)
    
    # Define expected sheet names for merging
    overall_test_sheet = 'Test_Overall_Distribution'
    overall_onekey_sheet = 'OneKey_Overall_Distribution'
    countrywise_test_sheet = 'Test_Country_Distribution'
    countrywise_onekey_sheet = 'OneKey_Country_Distribution'

    merged_predictions_dfs = {}

    # Helper function to clean 'Unnamed: 0' if present
    def clean_df(df):
        if not df.empty and 'Unnamed: 0' in df.columns:
            return df.drop(columns=['Unnamed: 0'])
        return df

    # --- Merge Overall Predictions Distribution ---
    test_overall = clean_df(predictions_dist_dfs.get(overall_test_sheet, pd.DataFrame()))
    onekey_overall = clean_df(predictions_dist_dfs.get(overall_onekey_sheet, pd.DataFrame()))

    if not test_overall.empty and not onekey_overall.empty:
        merged_overall = merge_prediction_data(test_overall, onekey_overall, is_countrywise=False)
        if not merged_overall.empty:
            merged_predictions_dfs['Overall_Predictn_Distribtn'] = merged_overall
            print("✅ Created  Overall Predictions Distribution.")
    else:
        print("⚠️ Could not find both Overall Test and OneKey sheets for merging.")

    # --- Merge Countrywise Predictions Distribution ---
    test_countrywise = clean_df(predictions_dist_dfs.get(countrywise_test_sheet, pd.DataFrame()))
    onekey_countrywise = clean_df(predictions_dist_dfs.get(countrywise_onekey_sheet, pd.DataFrame()))

    if not test_countrywise.empty and not onekey_countrywise.empty:
        # Note: Setting is_countrywise=True so it merges on the 'Country' column.
        merged_countrywise = merge_prediction_data(test_countrywise, onekey_countrywise, is_countrywise=True)
        print(merged_countrywise.head())
        if not merged_countrywise.empty:
            merged_predictions_dfs['Countrywise_Predictn_Distribtn'] = merged_countrywise
            print("✅ Created Countrywise Predictions Distribution.")
    else:
        print("⚠️ Could not find both Countrywise Test and OneKey sheets for merging.")


    # --- Step 5: Final merge and writing to Excel with multiple sheets ---

    # Create the ExcelWriter object, explicitly using 'openpyxl'
    with pd.ExcelWriter(Final_Results, engine='openpyxl') as writer:
         
        if not classification_report_df.empty:
            classification_report_df.to_excel(writer, sheet_name='CV1_Classifctn_Report', index=False)
            print(f"✅ Sheet 'classification_report from Model' written to {Final_Results}")
        else:
            print("⚠️ Skipping 'classification_report from Model' sheet: Data is empty.")


        # 5a: Merged Prediction Distribution Sheets (NEW)
        if merged_predictions_dfs:
            print(f"\nWriting {len(merged_predictions_dfs)} Merged Prediction Distribution sheets to Excel...")
            for sheet_name, df in merged_predictions_dfs.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"✅ All merged prediction distribution sheets written.")
        else:
            print("⚠️ Skipping merged Prediction Distribution sheets: No merged data available.")
       
        print(f"\n--- Debugging General Evaluation Sheet ---")
        print(f"step3_df is empty: {step3_df.empty}")
        print(f"df2 is empty: {df2.empty}")
        print(f"step3_df columns: {step3_df.columns.tolist() if not step3_df.empty else 'N/A'}")
        print(f"df2 columns: {df2.columns.tolist() if not df2.empty else 'N/A'}")
        print(f"step3_df 'Country' sample: {step3_df['Country'].head().tolist() if 'Country' in step3_df.columns and not step3_df.empty else 'N/A'}")
        print(f"df2 'Country' sample: {df2['Country'].head().tolist() if 'Country' in df2.columns and not df2.empty else 'N/A'}")
        print(f"-----------------------------------------\n")

        # 5b: General evaluation (Sheet 1)
        if not step3_df.empty and not df2.empty:
            final_df = pd.merge(step3_df, df2, on='Country', how='inner')
            print("\n\nFinal Merged DataFrame (General evaluation - F1 geomean):")
            print(final_df.head(7))
            final_df.to_excel(writer, sheet_name='General evaluation - F1 geomean', index=False)
            print(f"✅ Sheet 'General evaluation - F1 geomean' written to {Final_Results}")
        else:
            print("⚠️ Could not create final merged DataFrame (General evaluation sheet).")

        # 5c: Final Features from Model (Sheet 2)
        if not final_features_df.empty:
            final_features_df.to_excel(writer, sheet_name='Final Features from Model', index=False)
            print(f"✅ Sheet 'Final Features from Model' written to {Final_Results}")
        else:
            print("⚠️ Skipping 'Final Features from Model' sheet: Data is empty.")

        # 5d: Permutation Importance Sheets (All sheets dynamically loaded)
        if cleaned_perm_importance_dfs:
            print(f"\nWriting {len(cleaned_perm_importance_dfs)} Permutation Importance sheets to Excel...")
            for sheet_name, df in cleaned_perm_importance_dfs.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            print("✅ All permutation importance sheets written.")
        else:
            if not perm_importance_dfs:
                print("⚠️ Skipping Permutation Importance sheets: The file was not loaded (check path/name).")
            else:
                print("⚠️ Skipping Permutation Importance sheets: All loaded sheets were empty.")


    print("\n\n🎉 All data has been written to the final Excel file.")


if __name__ == "__main__":
    store_results(inference_config)
    print("Cohort Results saving starts")
    store_results(cohort_config)