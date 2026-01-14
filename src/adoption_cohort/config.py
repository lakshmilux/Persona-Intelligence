import os

ID_COL = "ID"
TARGET_COL = ['Adoption Profile Cohort 1: Innovators & Early Adopters',
        'Adoption Profile Cohort 2: Early Majority',
       'Adoption Profile Cohort 3: Late Majority & Laggards']

PROFILE_NAMES = TARGET_COL 

RANDOM_STATE = 42
TEST_SIZE = 0.2
INNER_CV_SPLITS = 5
OUTER_CV_SPLITS = 5
N_TRIALS = 20
STRATIFY_KEY = "Country_clean"

# Define directories
output_dir = "outputs/adoption_cohort"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


#Filepaths
OK_DATA_PATH = "data/raw/adoption_cohort/cohort_adoption_data_ctry_spec_ok_v2_0311.csv"
DATA_PATH = "data/raw/adoption_cohort/cohort_adoption_data_ctry_spec_v5.csv"
PROCESSED_DATAPATH_FOR_SHAP = "data/processed/adoption_cohort/cohort_adoption_data_ctry_spec_v5_preprocessed.csv"
SHAP_PATH = "outputs/adoption_cohort/step0_preprocessing/adptnchrt_shap_feature_importance.csv"
INITIAL_MODELS_EVALUATION = "outputs/adoption_cohort/step0_preprocessing/adptnchrt_initial_evaluation_results.xlsx"
SHAP_PLOT_PATH = "outputs/adoption_cohort/step0_preprocessing/adptnchrt_shap_summary_plot_top_20_factors.png"


# TEST DATA
RAWTEST_DATAPATH = os.path.join(output_dir, "adptnchrt_Test_data.csv")
TESTDATA_ORIGINALPRED = os.path.join(output_dir, "adptnchrt_Testdata_original_pred.csv")
TOTAL_TESTDATA = os.path.join(output_dir, "adptnchrt_TotalTest_Data.csv")

CATBOOST_DIR = "outputs/catboostdir/catboost_traininfo"

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
subfolders = ["model", "metrics", "predictions"]


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
OUTPUT_FOLD_RESULTS_FILE_PATH = os.path.join(step1_dir, "adptnchrt_Fold_results.xlsx")
ALL_OUTER_FOLD_MODELS_CLOUDPICKLE = os.path.join(step1_dir, "outer_fold_models")
FEATURES_FILEPATH = os.path.join(step1_dir, "adptnchrt_best_selected_features.json")
NESTEDCV_STRATIFICATION = os.path.join(step1_dir, "adptnchrt_nestedinnercv_stratification.xlsx")

# STEP2A
# Will get model, metrics from optuna enhanced.py but as it is just hyperparameters optimization
# we dont get predictions here


## STEP 2B
#METRICS_BLENDING_APPROACH = os.path.join(step2b_dir, "metrics", "metrics.xlsx")
#BLENDED_ONEKEY_PREDICTIONS = os.path.join(step2b_dir, "predictions", "OK_predictions.csv")
#PREDICTIONS_BLENDING_APPROACH = os.path.join(step2b_dir, "predictions", "test_predictions.csv")


# Step 3
CLOUDPICKLE_FILENAME_STEP3 = os.path.join(step3a_dir, "model", "adptnchrt_final_model.pkl")
METRICS_FILENAME_STEP3 = os.path.join(step3a_dir, "metrics", "adptnchrt_metrics.xlsx")
PREDICTIONS_FILENAME_STEP3 = os.path.join(step3a_dir, "predictions", "adptnchrt_test_predictions.xlsx")
SINGLECV_STRATIFICATION = os.path.join(step3a_dir, "adptnchrt_1cv_stratification.xlsx")
CV1_ONEKEY_PREDICTIONS = os.path.join(step3a_dir, "predictions", "adptnchrt_CV1_ONEKEY_PREDICTIONS.csv")
CV1_ONEKEY_PREDICTIONS_DISTRIBUTION = os.path.join(step3a_dir,"predictions", "adptnchrt_CV1_ONEKEY_PRED_DISTRIBUTION.xlsx")
OK_PREDICTIONS_FILENAME_STEP3 = os.path.join(step3a_dir, "predictions", "adptnchrt_OK_PREDICTIONS.csv")


# SHAP PERIMP EVALUATION
COUNTRYWISE_PERM_IMPORTANCE_DIR = os.path.join(output_dir, "Evaluation_shapperm_importance")
COUNTRYWISE_PERM_IMPORTANCE_FILENAME = "adptnchrt_global_and_country_wise_permutation_importance.xlsx"


final_results_dir = os.path.join(output_dir, "Final_Results")
os.makedirs(final_results_dir, exist_ok=True)

# Results
Final_Results = os.path.join(output_dir, "Final_Results", "adptnchrt_All_Combined_Results.xlsx")

