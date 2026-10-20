"""
Predictive Analysis Using Machine Learning Models
Walmart Weekly Sales - High vs Low Sales Week Classification
===============================================================

Pipeline:
  1. Load & clean data (handle missing values)
  2. Feature engineering (date decomposition, per-store target creation)
  3. Feature encoding (one-hot encoding for categorical variables)
  4. Feature scaling (StandardScaler)
  5. Train/test split (stratified)
  6. Train three classifiers: Logistic Regression, Random Forest, XGBoost
  7. Evaluate & compare: Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC-AUC
  8. Identify best-performing model and save all artifacts (plots + metrics table)
"""

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
np.random.seed(42)

# --------------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------------
df = pd.read_csv("Walmart_Sales.csv")
print(f"Raw shape: {df.shape}")
print(f"Missing values before cleaning:\n{df.isnull().sum()}\n")

# --------------------------------------------------------------------------
# 2. DATA CLEANING
# --------------------------------------------------------------------------
# Defensive median imputation (robust to outliers) in case of missing values;
# this real Kaggle file has none, but the step is kept so the pipeline is
# safe to run on other versions/extracts of the dataset.
for col in ["Temperature", "CPI", "Unemployment"]:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

# --------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# --------------------------------------------------------------------------
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

# Target: classify each week as a "High Sales" week (1) or "Low Sales" week (0)
# relative to that specific store's own median -> removes store-scale bias and
# turns the raw regression signal (Weekly_Sales) into a meaningful, balanced
# binary classification target.
df["Store_Median_Sales"] = df.groupby("Store")["Weekly_Sales"].transform("median")
df["High_Sales"] = (df["Weekly_Sales"] > df["Store_Median_Sales"]).astype(int)

print(f"Target balance:\n{df['High_Sales'].value_counts(normalize=True)}\n")

feature_cols_numeric = ["Temperature", "Fuel_Price", "CPI", "Unemployment", "Year", "Month", "Week"]
feature_cols_categorical = ["Store", "Holiday_Flag"]

X = df[feature_cols_numeric + feature_cols_categorical].copy()
y = df["High_Sales"]

# --------------------------------------------------------------------------
# 4. FEATURE ENCODING
# --------------------------------------------------------------------------
# One-hot encode Store (nominal categorical, 45 stores) and Holiday_Flag
X = pd.get_dummies(X, columns=["Store", "Holiday_Flag"], drop_first=True)
print(f"Feature matrix shape after encoding: {X.shape}\n")

# --------------------------------------------------------------------------
# 5. TRAIN / TEST SPLIT
# --------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------------------------------------
# 6. FEATURE SCALING
# --------------------------------------------------------------------------
scaler = StandardScaler()
num_cols = feature_cols_numeric
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

# --------------------------------------------------------------------------
# 7. MODEL TRAINING
# --------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.08, random_state=42,
        eval_metric="logloss", n_jobs=-1
    ),
}

results = []
conf_matrices = {}
trained_models = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, preds)

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1-Score": f1, "ROC-AUC": auc,
    })
    conf_matrices[name] = cm
    trained_models[name] = model

    print(f"--- {name} ---")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}\n")

# --------------------------------------------------------------------------
# 8. COMPARISON TABLE
# --------------------------------------------------------------------------
results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False).reset_index(drop=True)
results_df.to_csv("model_comparison_results.csv", index=False)
print("Final Model Comparison:")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
print(f"\nBest performing model: {best_model_name}")

with open("summary.json", "w") as f:
    json.dump({
        "best_model": best_model_name,
        "results": results_df.to_dict(orient="records"),
        "n_rows": int(df.shape[0]),
        "n_features_after_encoding": int(X.shape[1]),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }, f, indent=2)

# --------------------------------------------------------------------------
# 9. PLOTS
# --------------------------------------------------------------------------
# 9a. Metric comparison bar chart
metrics_melt = results_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                                var_name="Metric", value_name="Score")
plt.figure(figsize=(10, 6))
sns.barplot(data=metrics_melt, x="Metric", y="Score", hue="Model")
plt.title("Model Comparison Across Evaluation Metrics")
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()

# 9b. Confusion matrices, side by side
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, cm) in zip(axes, conf_matrices.items()):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Sales", "High Sales"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.close()

# 9c. Feature importance (Random Forest & XGBoost)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, name in zip(axes, ["Random Forest", "XGBoost"]):
    model = trained_models[name]
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    sns.barplot(x=importances.values, y=importances.index, ax=ax)
    ax.set_title(f"Top 10 Feature Importances - {name}")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

print("\nSaved: model_comparison_results.csv, model_comparison.png, confusion_matrices.png, feature_importance.png, summary.json")
