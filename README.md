# Predictive Analysis Using Machine Learning Models — Walmart Sales

Predicts whether a Walmart store's **weekly sales will be above or below its own historical median** ("High Sales" vs "Low Sales" week) from economic and seasonal indicators, using three supervised classification models trained and compared on the same data.

## Dataset
[Walmart Sales Dataset (Kaggle)](https://www.kaggle.com/) — **45 stores**, weekly records, **6,435 rows**, no missing values.
Columns: `Store`, `Date`, `Weekly_Sales`, `Holiday_Flag`, `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`.

## Pipeline
1. **Data cleaning** — defensive median imputation for any missing numeric values (none present in this extract)
2. **Feature engineering** — date decomposition (`Year`, `Month`, `Week`); binary target built from each store's own median weekly sales (removes store-size bias, yields a balanced target: ~50.3% / 49.7%)
3. **Feature encoding** — one-hot encoding of `Store` and `Holiday_Flag` → 52 model-ready features
4. **Feature scaling** — `StandardScaler` on all numeric features
5. **Modeling** — Logistic Regression, Random Forest, and XGBoost trained on an 80/20 stratified split
6. **Evaluation** — Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix for every model

## Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost** | **0.796** | **0.805** | 0.779 | **0.792** | **0.889** |
| Random Forest | 0.754 | 0.749 | **0.761** | 0.755 | 0.841 |
| Logistic Regression | 0.618 | 0.622 | 0.587 | 0.604 | 0.657 |

**Best performer: XGBoost** — highest F1-score and ROC-AUC. Logistic Regression trails both ensemble methods by a wide margin, suggesting the relationship between these features and weekly sales performance is meaningfully non-linear.

## Files
- `Walmart_Sales_Prediction.ipynb` — full notebook, narrative + code + rendered output charts
- `walmart_sales_prediction.py` — equivalent standalone script
- `Walmart_Sales.csv` — Kaggle dataset
- `model_comparison.png`, `confusion_matrices.png`, `feature_importance.png` — output visualizations
- `model_comparison_results.csv`, `summary.json` — metrics in raw form
- `requirements.txt` — dependencies

## How to Run
```bash
pip install -r requirements.txt
python walmart_sales_prediction.py
# or, for the notebook:
jupyter nbconvert --to notebook --execute --inplace Walmart_Sales_Prediction.ipynb
```

## Tech Stack
Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Seaborn
