from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PLOTS = ROOT / "outputs" / "plots"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
for folder in [PROCESSED, PLOTS, MODELS, REPORTS]:
    folder.mkdir(parents=True, exist_ok=True)

# Put your company's "Customer Churn.csv" inside data/raw.
files = list(RAW.glob("*.csv"))
if not files:
    raise FileNotFoundError("Place your company dataset CSV inside data/raw, then run this script again.")
preferred = [f for f in files if "customer churn" in f.name.lower()]
DATA_FILE = preferred[0] if preferred else files[0]
print(f"Loading: {DATA_FILE.name}")
df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

target = "Churn" if "Churn" in df.columns else ("Exited" if "Exited" in df.columns else None)
if target is None:
    raise ValueError("Could not find a target column named 'Churn' or 'Exited'. Check your dataset columns.")

print("\n--- DATA OVERVIEW ---")
print("Rows, columns:", df.shape)
print("\nTarget counts:\n", df[target].value_counts(dropna=False))
print("\nMissing values before cleaning:\n", df.isna().sum().sort_values(ascending=False).head(15))

# Clean blank strings and common Telco data issue
df = df.replace(r"^\s*$", np.nan, regex=True)
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
# Drop duplicate records
df = df.drop_duplicates().copy()
# Drop ID-like columns that do not help prediction
for col in ["customerID", "CustomerID", "RowNumber", "Surname", "id", "ID"]:
    if col in df.columns:
        df = df.drop(columns=col)

# Convert target to 0/1
if target == "Churn":
    mapping = {"Yes": 1, "No": 0, "1": 1, "0": 0, 1: 1, 0: 0}
    df[target] = df[target].map(mapping)
else:
    df[target] = pd.to_numeric(df[target], errors="coerce")
df = df.dropna(subset=[target])
df[target] = df[target].astype(int)

df.to_csv(PROCESSED / "customer_churn_clean.csv", index=False)
print("\nCleaned data saved to data/processed/customer_churn_clean.csv")
print("Rows after cleaning:", len(df))
print("Churn rate (%):", round(df[target].mean() * 100, 2))

# Quick EDA visuals
sns.set_theme()
if target == "Churn":
    sns.countplot(data=df, x=target)
    plt.title("Customer Churn Distribution")
    plt.tight_layout()
    plt.savefig(PLOTS / "churn_distribution.png", dpi=150)
    plt.close()

if "Contract" in df.columns and target == "Churn":
    temp = df.copy()
    temp[target] = temp[target].map({0: "Stayed", 1: "Churned"})
    sns.countplot(data=temp, x="Contract", hue=target)
    plt.title("Churn by Contract Type")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(PLOTS / "churn_by_contract.png", dpi=150)
    plt.close()

if "tenure" in df.columns and target == "Churn":
    sns.boxplot(data=df, x=target, y="tenure")
    plt.title("Tenure by Churn Status")
    plt.tight_layout()
    plt.savefig(PLOTS / "tenure_by_churn.png", dpi=150)
    plt.close()

if "MonthlyCharges" in df.columns and target == "Churn":
    sns.boxplot(data=df, x=target, y="MonthlyCharges")
    plt.title("Monthly Charges by Churn Status")
    plt.tight_layout()
    plt.savefig(PLOTS / "monthly_charges_by_churn.png", dpi=150)
    plt.close()

# Split features and target
X = df.drop(columns=[target])
y = df[target]
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

candidates = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
}
results = []
best_name, best_pipeline, best_f1 = None, None, -1

for name, estimator in candidates.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    try:
        prob = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, prob)
    except Exception:
        auc = np.nan
    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1_Score": f1_score(y_test, pred, zero_division=0),
        "ROC_AUC": auc
    }
    results.append(row)
    print(f"\n{name} metrics:", row)
    if row["F1_Score"] > best_f1:
        best_f1 = row["F1_Score"]
        best_name, best_pipeline = name, pipe
        best_pred = pred
        best_prob = prob if "prob" in locals() else None

metrics = pd.DataFrame(results).sort_values("F1_Score", ascending=False)
metrics.to_csv(REPORTS / "model_metrics.csv", index=False)
joblib.dump(best_pipeline, MODELS / "churn_model.joblib")
print(f"\nBest model by F1-score: {best_name}")
print("\nClassification report:\n", classification_report(y_test, best_pipeline.predict(X_test), zero_division=0))

ConfusionMatrixDisplay.from_predictions(y_test, best_pipeline.predict(X_test))
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig(PLOTS / "confusion_matrix.png", dpi=150)
plt.close()

# Save a plain-language summary for the report
summary = f"""CUSTOMER CHURN PREDICTION - RUN SUMMARY

Dataset: {DATA_FILE.name}
Rows after duplicate removal and target validation: {len(df)}
Columns including target: {df.shape[1]}
Target column: {target}
Observed churn/exit rate: {df[target].mean()*100:.2f}%

Best model by F1-score: {best_name}
{metrics.to_string(index=False)}

Important: metrics are based on a held-out test split. They are not a guarantee of future performance.
This project identifies statistical patterns; it does not prove that any one factor causes churn.
"""
(REPORTS / "run_summary.txt").write_text(summary, encoding="utf-8")
print("\nFinished! Check outputs/plots, models/churn_model.joblib, and reports/model_metrics.csv")
