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