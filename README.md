**Challenge:** Assigning Persona for each HCP and even addressed a **99% digital bias** in raw country-specialty data.

**Solution:** - Implemented **Stacked Modeling** to assign Persona and **Cohort Modeling** to normalize skewed distributions.
-Developed custom ML pipelines in **Azure** to automate the balancing of skewed features and unified the code of Persona creation with cohort modelling.

**Outcome:** - Successfully **reduced bias by 23%**.
- Achieved a 23% absolute reduction in bias, enabling market-realistic insights.

# Data Directory
- `raw/digital`: sample data related to digital profile
- `raw/adoption`: sample data related to adoption profile
- change the input file names if you need to run this project as per config filenames
- `raw/processed`: processed data to shap analysis for digital,adoption,digital_cohort,adoption_cohort
- for cohort - sampel data isnt provided
  
# Notebooks Directory

This directory contains Jupyter notebooks for analysis and experimentation.

## Structure

- `adoption/`: Notebooks related to adoption analysis use case
- `digital/`: Notebooks related to digital analysis use case

# Data Directory

This directory contains all data files used in the project.

## Structure

- `raw/`: Contains original, immutable data files. Never modify files in this directory.
  - Input data from various sources
  - Original datasets before any processing
  - Reference data

- `processed/`: Contains cleaned and transformed data ready for analysis
  - Cleaned datasets
  - Feature-engineered data
  - Intermediate processing results
  - Model-ready datasets

## Best Practices

1. Always keep raw data immutable - never modify original files
2. Document data sources and collection dates
3. Use clear, descriptive filenames with dates where applicable
4. Include data documentation (data dictionaries, schemas) alongside the data files
5. For large files, consider adding them to .gitignore and storing them elsewhere

## Process Flow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    INITIALIZATION                                │
│  • Setup project paths                                           │
│  • Load configuration (ModelConfig)                             │
│  • Set random seeds for reproducibility                          │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LOADING & SPLITTING                      │
│  • Load main dataset and SHAP feature importance                │
│  • Organize features by importance                              │
│  • Split into train/test (80/20) with stratification            │
│  • Identify numerical vs categorical columns                     │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              HYPERPARAMETER OPTIMIZATION (Optuna)                │
│  • Create OptunaObjectiveEnhanced                               │
│  • Run N trials (default: 20)                                   │
│  • For each trial:                                               │
│    - Sample features (5 to all features)                         │
│    - Sample model(s): LGBM, BRF, CatBoost                       │
│    - Sample hyperparameters for selected model(s)                │
│    - Evaluate using nested 5-fold CV                            │
│      * 5 outer folds for evaluation                             │
│      * 5 inner folds for stacking OOF predictions               │
│    - Calculate custom log loss                                  │
│  • Select best hyperparameters and features                     │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              FINAL MODEL TRAINING                                │
│  • Convert data for weighted training                           │
│  • Build final model with best hyperparameters:                 │
│    - Single CatBoost → build_final_catboost_softweights_model   │
│    - Other models → build_final_model (StackingClassifier)      │
│  • Train on full training set                                   │
│  • Save model (cloudpickle)                                     │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              EVALUATION & PREDICTIONS                            │
│  • Predict on training set → Calculate train metrics            │
│  • Predict on test set → Calculate test metrics                │
│  • Predict on OK data → Generate OK predictions                 │
│  • Save all results to Excel                                    │
└─────────────────────────────────────────────────────────────────┘


# Notebook Structure
├── data/
     ├── raw/                # Raw data of HCP and OK for both Digital and adoption Profiles   
│   ├── processed/          # Processed data from SHAP of HCP Inference and OK for both Digital and adoption Profiles
├── notebooks/
│   ├── digital/            # Digital models (using unified script for both HCP inference and COHORT Models)
│   ├── adoption/           # Adoption models (using unified script both HCP inference and COHORT Models)
│     └── README.md
├── src/
│   ├── common/             # CORE UNIFIED LOGIC
│   │   ├── config.py       # ModelConfig class (Object Wrapper)
│   │   ├── data_preprocessing.py
│   │   ├── optuna_enhanced.py
│   │   ├── metrics.py
│   │   └── pipeline_factory_enhanced.py
│   │   ├── ok_predicitons.py
│   │   ├── model_factory_enhanced.py
│   │   ├── utils.py
│   │   └── stratification_tools.py
│   │
│   ├── digital/                   # Model-specific configurations
│   └── adoption/                  # Model-specific configurations
│   └── digital_cohort/            # Model-specific configurations
│   └── adoption_cohort/           # Model-specific configurations
└── requirements.txt
└── Outputs
        ├── digital/            # Digital Profile output files for Inference 
        ├── adoption/           # Adoption Profile output files for Inference
        ├── digital_cohort/     # Digital Profile output files for Cohort
        ├── adoption_cohort/    # Adoption Profile output files for Cohort 
