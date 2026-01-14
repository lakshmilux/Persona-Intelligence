import pandas as pd
import numpy as np
import cloudpickle
import os
import sys

from src.common.utils import setup_project_paths
project_root = setup_project_paths()
sys.path.insert(0, os.path.abspath(os.path.join(project_root, "src")))


from src.common.config import ModelConfig
from src.adoption_cohort import config as cohort_config
from src.adoption import config as inference_config

from src.common.OK_predictions_source import *


def run_OK_predictions(config):
    """
    The main pipeline for the cohort model, including data prep, training, evaluation, and SHAP.
    """
    
    # 1. Load Config and Data
    pipeline_config = ModelConfig.from_module(config)
    CV1_ONEKEY_PREDICTIONS = pipeline_config.cv1_onekey_predictions
    PREDICTIONS_FILENAME_STEP3 = pipeline_config.predictions_filename_step3
    CV1_ONEKEY_PREDICTIONS_DISTRIBUTION = pipeline_config.cv1_onekey_predictions_distribution
    ok_cohort_overall_dist = pd.DataFrame()
    ok_cohort_country_dist = pd.DataFrame()
    test_cohort_overall_dist = pd.DataFrame()
    test_cohort_country_dist = pd.DataFrame()
    
    try:
        df = pd.read_csv(pipeline_config.ok_data_path)
        print(f"Data loaded with shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {pipeline_config.ok_data_path}. Skipping pipeline.")
        return

    # Determine Model Type (Inference or Cohort/Softweights)
    is_softweights_cohort = isinstance(pipeline_config.target_col, list)
    
    # 2. Prepare Data based on Type
    print("\nPreparing data...")
    
    if is_softweights_cohort:
        # Use cohort_preprocess_data for cohort/training
        X_encoded, y, country_clean, processed_data, Country_Specialty = cohort_preprocess_data(
            config=pipeline_config, df=df, country_column="Country", new_column_name="Country_clean", min_obs=10, 
            id_columns=['Specialty (CDV2)', 'Country']
        )
        print("Cohort Preprocessing Completed. X_encoded shape:", Country_Specialty.shape)
        
        ok_country_prob = cohort_predictions_ok_data(config=pipeline_config,
                                                            ok_data = X_encoded,
                                                            Country_Specialty = Country_Specialty)
        print("prob",ok_country_prob.columns)
        print("head",ok_country_prob.head())
        
        if not ok_country_prob.empty:
            ok_country_prob.to_csv(CV1_ONEKEY_PREDICTIONS, index=False)
            print(f"✅ Softweights OneKey predictions saved to: {CV1_ONEKEY_PREDICTIONS}")
        else:
            print("Warning: ok_country_prob is empty, nothing to save.")
      
       # --- PART 2: CALCULATE DISTRIBUTIONS ---
        # Calculate for OneKey Data
        ok_cohort_overall_dist, ok_cohort_country_dist = cohort_calculate_distribution(
            df=ok_country_prob, 
            data_type="OneKey Softweights"
        )

        # Calculate for Test Data (if the file exists)
        try:
            total_Test = pd.read_excel(PREDICTIONS_FILENAME_STEP3)
            print("test",total_Test.columns)
            test_cohort_overall_dist, test_cohort_country_dist = cohort_calculate_distribution(
                df=total_Test, 
                data_type="Test Softweights"
            )
        except Exception as e:
            print(f"Skipping Test Data Distribution: {e}")
            test_cohort_overall_dist, test_cohort_country_dist = pd.DataFrame(), pd.DataFrame() 
        
        if ok_cohort_overall_dist.empty and test_cohort_overall_dist.empty:
            print("No valid distribution data to save. Aborting Excel write.")
        else:
            try:
                with pd.ExcelWriter(CV1_ONEKEY_PREDICTIONS_DISTRIBUTION, engine='openpyxl') as writer:
                    # 1. OneKey Distributions
                    if not ok_cohort_overall_dist.empty:
                        ok_cohort_overall_dist.to_excel(writer, sheet_name="OneKey_Overall_Distribution", index=False)
                        print("✅ Saved OneKey Overall Distribution.")
                    if not ok_cohort_country_dist.empty:
                        ok_cohort_country_dist.to_excel(writer, sheet_name="OneKey_Country_Distribution", index=False)
                        print("✅ Saved OneKey Country-wise Distribution.")

                    # 2. Test Data Distributions
                    if not test_cohort_overall_dist.empty:
                        test_cohort_overall_dist.to_excel(writer, sheet_name="Test_Overall_Distribution", index=False)
                        print("✅ Saved Test Data Overall Distribution.")
                    if not test_cohort_country_dist.empty:
                        test_cohort_country_dist.to_excel(writer, sheet_name="Test_Country_Distribution", index=False)
                        print("✅ Saved Test Data Country-wise Distribution.")

                print(f"\n🎉 All prediction distributions and feature names saved to {CV1_ONEKEY_PREDICTIONS_DISTRIBUTION}")
            except Exception as e:
                print(f"Error saving data to Excel file: {e}")

    else:
      
        # Use inference_preprocess_data for inference (single target column)
        okdata = inference_preprocess_data(config=pipeline_config, df=df)
        print(f"Inference Preprocessing Completed. DataFrame shape: {okdata.shape}")

        ok_inf,best_features = inference_predictions_ok_data(config=pipeline_config,ok_data=df)
        print("inf_preds",ok_inf.shape)

        ok_prob_overall_dist = pd.DataFrame()
        ok_inf_country_dist = pd.DataFrame()
        test_inf_overall_dist = pd.DataFrame()
        test_inf_country_dist = pd.DataFrame()

        # Calculate distributions for OneKey data
        ok_inf_overall_dist, ok_inf_country_dist = inference_calculate_distribution(config=pipeline_config, df = ok_inf, data_type="OneKey")

        # Save the full OneKey data with predictions
        ok_inf.to_csv(CV1_ONEKEY_PREDICTIONS, index=False)
        print(f"\nTotal OneKey data with predictions saved to {CV1_ONEKEY_PREDICTIONS}")

        # --- Part 2: Analyze Test Data Distribution (Assuming pre-calculated predictions) ---
        print("\n--- Analyzing Test Data Distribution ---")
        
        try:
            # Load the Test Data (ASSUMED to already contain a 'predictions' column from prior evaluation)
            total_Test = pd.read_excel(PREDICTIONS_FILENAME_STEP3)
            last_col = total_Test.columns[-1]
            print("last_col",last_col)
            if last_col.startswith('y_pred_'):
                  total_Test.rename(columns={last_col: 'predictions'}, inplace=True)
                  print(f"Renamed '{last_col}' to 'predictions'.")
            else:
                print("Warning: No column starting with 'y_pred_' was found.")
            #total_Test.rename({'y_pred_adoption profile':'predictions'},axis=1,inplace=True)
            print(f"Test Data loaded. Shape: {total_Test.shape}")

            # Calculate distributions for Test data
            test_inf_overall_dist, test_inf_country_dist = inference_calculate_distribution(config=pipeline_config, df = total_Test, data_type="Test Data")

        except FileNotFoundError:
            print(f"Error: The file {PREDICTIONS_FILENAME_STEP3} was not found. Skipping Test Data analysis.")
            test_inf_overall_dist = pd.DataFrame()
            test_inf_country_dist = pd.DataFrame()
        except Exception as e:
            print(f"An unexpected error occurred during Test Data analysis: {e}") 

        # --- Part 3: Save All Distributions to Excel ---
        print("\n--- Saving all distributions to Excel ---")

        if ok_inf_overall_dist.empty and test_inf_overall_dist.empty:
            print("No valid distribution data to save. Aborting Excel write.")
        else:
            try:
                with pd.ExcelWriter(CV1_ONEKEY_PREDICTIONS_DISTRIBUTION, engine='openpyxl') as writer:
                    # 1. OneKey Distributions
                    if not ok_inf_overall_dist.empty:
                        ok_inf_overall_dist.to_excel(writer, sheet_name="OneKey_Overall_Distribution", index=False)
                        print("✅ Saved OneKey Overall Distribution.")
                    if not ok_inf_country_dist.empty:
                        ok_inf_country_dist.to_excel(writer, sheet_name="OneKey_Country_Distribution", index=False)
                        print("✅ Saved OneKey Country-wise Distribution.")

                    # 2. Test Data Distributions
                    if not test_inf_overall_dist.empty:
                        test_inf_overall_dist.to_excel(writer, sheet_name="Test_Overall_Distribution", index=False)
                        print("✅ Saved Test Data Overall Distribution.")
                    if not test_inf_country_dist.empty:
                        test_inf_country_dist.to_excel(writer, sheet_name="Test_Country_Distribution", index=False)
                        print("✅ Saved Test Data Country-wise Distribution.")

                print(f"\n🎉 All prediction distributions and feature names saved to {CV1_ONEKEY_PREDICTIONS_DISTRIBUTION}")
            except Exception as e:
                print(f"Error saving data to Excel file: {e}")

if __name__ == "__main__":
    print("\n--- Running COHORT Pipeline ---")
    run_OK_predictions(cohort_config)
    
    print("\n--- Running INFERENCE Pipeline ---")
    run_OK_predictions(inference_config)

