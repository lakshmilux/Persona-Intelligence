#----------------------- COHORT  DIGITAL --------------------------------

import os

ID_COL = "ID"
TARGET_COL = ['Digital Profile Cohort 1: Digital Pioneers',
              'Digital Profile Cohort 2: Hybrid Engagers',
              'Digital Profile Cohort 3: Traditionalists']



PROFILE_NAMES = TARGET_COL 

RANDOM_STATE = 42
TEST_SIZE = 0.2
INNER_CV_SPLITS = 5
OUTER_CV_SPLITS = 5
N_TRIALS = 20
SINGLECV_STRATIFICATION = 5
STRATIFY_KEY = "Country_clean"

# Create output directory
output_dir = "outputs/digital_cohort"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


CATBOOST_DIR = "outputs/catboostdir/catboost_traininfo"


#Filepaths
OK_DATA_PATH = "data/raw/digital_cohort/cohort_data_ctry_spec_ok_v2_0311.csv"
DATA_PATH = "data/raw/digital_cohort/cohort_data_ctry_spec_v5.csv"
PROCESSED_DATAPATH_FOR_SHAP = "data/processed/digital_cohort/cohort_data_ctry_spec_v5_preprocessed.csv"
SHAP_PATH = "outputs/digital_cohort/step0_preprocessing/dgchrt_shap_feature_importance.csv"
INITIAL_MODELS_EVALUATION = "outputs/digital_cohort/step0_preprocessing/dgchrt_initial_evaluation_results.xlsx"
SHAP_PLOT_PATH = "outputs/digital_cohort/step0_preprocessing/dgchrt_shap_summary_plot_top_20_factors.png"

#DATA_PATH = COHORT_DATA_CTRY_SPEC_PATH
#SHAP_PATH = PREPROCESSED_DATA_PATH

STRATIFY_KEY = "Country_clean"

# TEST DATA
RAWTEST_DATAPATH = os.path.join(output_dir, "dgchrt_Test_data.csv")
TESTDATA_ORIGINALPRED = os.path.join(output_dir, "dgchtr_Testdata_original_pred.csv")
TOTAL_TESTDATA = os.path.join(output_dir, "dgchrt_TotalTest_Data.csv")


# ---------------------------------------------------------------------
# OUTPUT PATHS
# ---------------------------------------------------------------------

# Define directories


step0_dir = os.path.join(output_dir, "step0_preprocessing")# For feature information
os.makedirs(step0_dir, exist_ok=True)
step1_dir = os.path.join(output_dir, "step1_NestedCVfold_results")
os.makedirs(step1_dir, exist_ok=True)
step2a_dir = os.path.join(output_dir, "step2a_1CV_hyperparameters_optimisation")
os.makedirs(step2a_dir, exist_ok=True)
#step2b_dir = os.path.join(output_dir, "step2b_blending")
#os.makedirs(step2b_dir, exist_ok=True)
#
## Create the subdirectories directly here
#os.makedirs(os.path.join(step2b_dir, "metrics"), exist_ok=True)
#os.makedirs(os.path.join(step2b_dir, "predictions"), exist_ok=True)

step3a_dir = os.path.join(output_dir, "step3a_1CV_final_model")

# List of all step directories
step_dirs = [step3a_dir]

# Define subfolders to be added to each step directory
subfolders = ["model", "metrics", "predictions"]


# Function to create directories with subfolders
def create_step_directories(base_dirs, subfolders):
    for base_dir in base_dirs:
        for subfolder in subfolders:
            subfolder_path = os.path.join(base_dir, subfolder)
            os.makedirs(subfolder_path, exist_ok=True)


# Create all directories and subfolders
create_step_directories(step_dirs, subfolders)


# step1
OUTPUT_FOLD_RESULTS_FILE_PATH = os.path.join(step1_dir, "dgchrt_Fold_results.xlsx")
ALL_OUTER_FOLD_MODELS_CLOUDPICKLE = os.path.join(step1_dir, "outer_fold_models")
FEATURES_FILEPATH = os.path.join(step1_dir, "dgchrt_best_selected_features.json")
NESTEDCV_STRATIFICATION = os.path.join(step1_dir, "dgchrt_nestedinnercv_stratification.xlsx")


# STEP2A
# Will get model, metrics from optuna enhanced.py but as it is just hyperparameters optimization
# we dont get predictions here


# STEP 2B
#METRICS_BLENDING_APPROACH = os.path.join(step2b_dir, "metrics", "dgchrt_metrics.xlsx")
#BLENDED_ONEKEY_PREDICTIONS = os.path.join(step2b_dir, "predictions", "dgchrt_OK_predictions.csv")
#PREDICTIONS_BLENDING_APPROACH = os.path.join(step2b_dir, "predictions", "dgchrt_test_predictions.csv")


# Step 3
CLOUDPICKLE_FILENAME_STEP3 = os.path.join(step3a_dir, "model", "dgchrt_final_model.pkl")
METRICS_FILENAME_STEP3 = os.path.join(step3a_dir, "metrics", "dgchrt_metrics.xlsx")
PREDICTIONS_FILENAME_STEP3 = os.path.join(step3a_dir, "predictions", "dgchrt_test_predictions.xlsx")
SINGLECV_STRATIFICATION = os.path.join(step3a_dir, "dgchrt_1cv_stratification.xlsx")
CV1_ONEKEY_PREDICTIONS = os.path.join(step3a_dir, "predictions", "dgchrt_CV1_ONEKEY_PREDICTIONS.csv")
CV1_ONEKEY_PREDICTIONS_DISTRIBUTION = os.path.join(step3a_dir,"predictions", "dgchrt_CV1_ONEKEY_PRED_DISTRIBUTION.xlsx")
OK_PREDICTIONS_FILENAME_STEP3 = os.path.join(step3a_dir, "predictions", "dgchrt_OK_PREDICTIONS.csv")



# SHAP PERIMP EVALUATION
COUNTRYWISE_PERM_IMPORTANCE_DIR = os.path.join(output_dir, "Evaluation_shapperm_importance")
COUNTRYWISE_PERM_IMPORTANCE_FILENAME = "dgchrt_global_and_country_wise_permutation_importance.xlsx"


final_results_dir = os.path.join(output_dir, "Final_Results")
os.makedirs(final_results_dir, exist_ok=True)

# Results
Final_Results = os.path.join(output_dir, "Final_Results", "dgchrt_All_Combined_Results.xlsx")


## Cohort Assaignment
#step3_inf_output_dir = "outputs/digital"
#step3inf_dir = os.path.join(step3_inf_output_dir, "step3a_1CV_final_model")
#DG_OK_PRED_PROB  = os.path.join(step3inf_dir, "prob_predictions", "PROB_CV1_ONEKEY_PREDICTIONS.csv")
#DG_COHORT_OK_PRED = os.path.join(step3a_dir, "predictions", "OK_PREDICTIONS.csv")
#
#assign_dir = os.path.join(step3_inf_output_dir, "assignment")
#
##Cohort Output Paths
#ASSIGNED_OK_OUTPUT = os.path.join(assign_dir,"OK_COHORT_ASSIGNMENTS.csv")
#SKIPPED_ASSIGNED_OK_OUTPUT = os.path.join(assign_dir,"OK_COHORT_SKIPPED_NO_QUOTA.csv")
#                                


