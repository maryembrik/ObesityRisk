
# Obesity Risk Classification

A complete, reproducible pipeline to predict obesity level from lifestyle and anthropometric data. It includes data quality steps, feature engineering (BMI), model comparison, hyperparameter tuning, calibrated probabilities, interpretation, and production-ready artifacts.

## Project structure
- `Obesity risk MClassification.ipynb`: main notebook (EDA → modeling → evaluation → interpretation)
- `train_obesity.csv`: dataset
- `best_obesity_pipeline.pkl`: saved preprocessing + model pipeline
- `best_obesity_pipeline_metadata.json`: metadata (model, score, classes, features, timestamp)
- `Obesity risk MClassification.html`: static export for sharing (optional)
- `obesity-Risk-Photo.jpg`: cover image (optional)

## Environment setup
```bash
# Optional: create a virtual environment
python -m venv .venv
# Windows PowerShell
. .\.venv\Scripts\Activate.ps1
# Install dependencies
pip install -U pip
pip install numpy pandas scikit-learn seaborn matplotlib jupyter
```

Optional (used in notebook for interpretation/calibration):
```bash
pip install shap
```

## Dataset
- Source: Kaggle obesity/lifestyle dataset.
- Preprocessing in notebook:
  - Column renaming for clarity
  - Drop `id`
  - Train/validation/test via stratified split

## What the pipeline does
- Data quality:
  - Numeric imputation (median), categorical imputation (most frequent)
  - IQR outlier capping (numeric)
  - Rare-category grouping to “Other” (categorical)
- Feature engineering:
  - `BMI = Weight / Height^2`
  - BMI buckets: Underweight/Normal/Overweight/Obese
- Preprocessing:
  - `ColumnTransformer` + `Pipeline`
  - Numeric: impute → cap outliers → scale
  - Categorical: impute → rare-group → `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
- Models compared (5-fold stratified CV):
  - Logistic Regression, SVC (RBF), Decision Tree, Random Forest, GaussianNB
  - Leaderboard metrics: accuracy, macro F1, balanced accuracy, Cohen’s kappa, fit time
- Hyperparameter tuning:
  - `GridSearchCV` on top candidates (LR, RF, SVC RBF) using macro F1
- Final model:
  - Best CV macro F1 model is refit and evaluated on test set
  - Reports accuracy, macro F1, confusion matrix, and classification report
- Calibration:
  - `CalibratedClassifierCV` (Platt scaling) compared on test set
- Interpretation:
  - Permutation importance on the tuned pipeline

## How to run
```bash
# Start Jupyter
jupyter notebook
# Open "Obesity risk MClassification.ipynb"
# From the top: Run All
```

To export a shareable HTML report:
```bash
jupyter nbconvert --to html "Obesity risk MClassification.ipynb"
```

## Reproducing results
1. Open the notebook and run all cells in order. The script:
   - Rebuilds the preprocessing and engineered features
   - Compares models with stratified CV
   - Tunes the strongest models
   - Evaluates on a held-out test split
   - Saves artifacts
2. Outputs:
   - `best_obesity_pipeline.pkl`
   - `best_obesity_pipeline_metadata.json`

## Inference
Use the saved pipeline directly (same Python environment and scikit-learn version):
```python
import pickle
import pandas as pd

pipe = pickle.load(open("best_obesity_pipeline.pkl", "rb"))
row = pd.DataFrame([{
    "Gender": "Male",
    "Age": 30,
    "Height": 1.75,
    "Weight": 82,
    "Family history with overweight": "yes",
    "Frequent consumption of high-caloric food": "no",
    "Frequency of vegetable consumption": 2.0,
    "Number of main meals the person eats per day": 3.0,
    "Consumption of food between meals": "Sometimes",
    "SMOKE": "no",
    "Daily water consumption": 2.0,
    "Whether the person takes calorie supplements": "no",
    "Physical activity frequency": 2.0,
    "Time spent using technology": 3.0,
    "Alcohol consumption": "Sometimes",
    "Means of transportation used": "Public_Transportation",
    # BMI fields optional if your serving layer computes them
    "BMI": 82 / (1.75**2),
    "BMI_bucket": "Overweight"
}])

pred = pipe.predict(row)[0]
prob = pipe.predict_proba(row)[0] if hasattr(pipe, "predict_proba") else None
print(pred, prob)
```

## Notes
- scikit-learn >= 1.2 uses `OneHotEncoder(sparse_output=False)`. For older versions, use `sparse=False`.
- Set `random_state=42` for reproducibility.

## License
Add a license of your choice (e.g., MIT) as `LICENSE`.

## Acknowledgements
Dataset authors and the open-source ML community.
