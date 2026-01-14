#------------------------------INFERENCE ADOPTION  --------------------


import os
#import uuid

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

# Create output directory
output_dir = "outputs/adoption"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Input data paths
DATA_PATH = "data/raw/adoption/Adoption_hcp_1point30_6months.csv"
# all_v1 files are nothing but changes in mapped spclty column where USA,Korea not mapped correctly
#same exclusion criteria of 2 months 
#if files names doesnt have _v1 then also they belong to v1 itself
# now, all_v2 files will be with exclusion criteria of 6 months
OK_DATA_PATH = "data/processed/digital/OKtopredict30_v2.csv"
PROCESSED_DATAPATH_FOR_SHAP = DATA_PATH
SHAP_PATH = "outputs/adoption/step0_preprocessing/adptninf_shap_feature_importance.csv"
SHAP_PLOT_PATH = "outputs/adoption/step0_preprocessing/adptninf_Input_shap_global_importance.png"
INITIAL_MODELS_EVALUATION = "outputs/adoption/step0_preprocessing/adptninf_initial_evaluation_results.xlsx"




# TEST DATA
RAWTEST_DATAPATH = os.path.join(output_dir, "adptninf_Test_Data_v2.csv")
TESTDATA_ORIGINALPRED = os.path.join(output_dir, "adptninf_Testdata_original_pred_v2.csv")
TOTAL_TESTDATA = os.path.join(output_dir, "adptninf_TotalTest_Data_v2.csv")


# Model configuration
TARGET_COL = "adoption profile"
ID_COL = 'onekey id'
STRATIFY_KEY = None

TEST_SIZE = 0.2
OUTER_CV_SPLITS = 5
INNER_CV_SPLITS = 5
N_TRIALS = 20  # Adjust this number as needed for hyperparameter optimization
RANDOM_STATE = 42


# ---------------------------------------------------------------------
# OUTPUT PATHS
# ---------------------------------------------------------------------



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
subfolders = ["model", "metrics", "predictions", "prob_metrics" , "prob_predictions"]


# Function to create directories with subfolders
def create_step_directories(base_dirs, subfolders):
    for base_dir in base_dirs:
        for subfolder in subfolders:
            subfolder_path = os.path.join(base_dir, subfolder)
            os.makedirs(subfolder_path, exist_ok=True)


# Create all directories and subfolders
create_step_directories(step_dirs, subfolders)

# Define filenames
# STEP 0_EDA_input_feature SHAP importance
INPUT_SHAP_FEATURE_IMP_PLOT = os.path.join(step0_dir, "Input_shap_global_importance.png")

# step1
OUTPUT_FOLD_RESULTS_FILE_PATH = os.path.join(step1_dir, "adptninf_Fold_results.xlsx")
ALL_OUTER_FOLD_MODELS_CLOUDPICKLE = os.path.join(step1_dir, "outer_fold_models")
FEATURES_FILEPATH = os.path.join(step1_dir, "adptninf_best_selected_features.json")
NESTEDCV_STRATIFICATION = os.path.join(step1_dir, "adptninf_nestedinnercv_stratification.xlsx")

# STEP2A
# Will get model, metrics from optuna enhanced.py but as it is just hyperparameters optimization
# we dont get predictions here


## STEP 2B
#METRICS_BLENDING_APPROACH = os.path.join(step2b_dir, "metrics", "metrics.xlsx")
#BLENDED_ONEKEY_PREDICTIONS = os.path.join(step2b_dir, "predictions", "OK_predictions.csv")
#PREDICTIONS_BLENDING_APPROACH = os.path.join(step2b_dir, "predictions", "test_predictions.csv")


# Step 3
CLOUDPICKLE_FILENAME_STEP3 = os.path.join(step3a_dir, "model", "adptninf_final_model.pkl")
METRICS_FILENAME_STEP3 = os.path.join(step3a_dir, "metrics", "adptninf_metrics.xlsx")
PREDICTIONS_FILENAME_STEP3 = os.path.join(step3a_dir, "predictions", "adptninf_test_predictions.xlsx")
SINGLECV_STRATIFICATION = os.path.join(step3a_dir, "adptninf_1cv_stratification.xlsx")
CV1_ONEKEY_PREDICTIONS = os.path.join(step3a_dir, "predictions", "adptninf_CV1_ONEKEY_PREDICTIONS.csv")
CV1_ONEKEY_PREDICTIONS_DISTRIBUTION = os.path.join(step3a_dir,"predictions", "adptninf_CV1_ONEKEY_PRED_DISTRIBUTION.xlsx")

##Probabilities
#os.makedirs(os.path.join(step3a_dir, "prob_predictions"), exist_ok=True)
#PROB_METRICS_FILENAME_STEP3 = os.path.join(step3a_dir, "prob_metrics", "prob_metrics.xlsx")
#PROB_PREDICTIONS_FILENAME_STEP3 = os.path.join(step3a_dir, "prob_predictions", "prob_predictions.xlsx")
#PROB_CV1_ONEKEY_PREDICTIONS = os.path.join(step3a_dir, "prob_predictions", "PROB_CV1_ONEKEY_PREDICTIONS.csv")


# SHAP PERIMP EVALUATION
COUNTRYWISE_PERM_IMPORTANCE_DIR = os.path.join(output_dir, "Evaluation_shapperm_importance")
COUNTRYWISE_PERM_IMPORTANCE_FILENAME = "adptninf_global_and_country_wise_permutation_importance.xlsx"

final_results_dir = os.path.join(output_dir, "Final_Results")
os.makedirs(final_results_dir, exist_ok=True)

# Results
Final_Results = os.path.join(output_dir, "Final_Results", "adptninf_All_Combined_Results.xlsx")