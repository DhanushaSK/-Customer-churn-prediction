# Customer Churn Prediction and Retention Analytics

## Fast setup
1. Copy your company CSV into `data/raw/`. Prefer `Customer Churn.csv` for the Telco dataset.
2. Open a terminal in this folder.
3. Install packages: `python -m pip install -r requirements.txt`
4. Run analysis/model: `python run_project.py`
5. Start the app: `streamlit run app.py`

## Outputs
- `data/processed/customer_churn_clean.csv`
- `outputs/plots/` (EDA and confusion-matrix charts)
- `models/churn_model.joblib`
- `reports/model_metrics.csv`
- `reports/run_summary.txt`

## Notes
- The script supports a target column named `Churn` or `Exited`.
- Use only one dataset at a time. Do not combine datasets describing different businesses.
- Model metrics depend on the dataset and test split. Do not claim a particular accuracy before running the script.
- Keep personal/company data private. Do not commit confidential datasets or secrets to a public GitHub repository.
