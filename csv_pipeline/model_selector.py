"""
model_selector.py — Auto Model Selection for Bujji Babu CSV Agent

When user says "train a model" without specifying which:
  - Detect task type: classification vs regression vs clustering
  - Train 3 candidate models
  - Compare metrics in a clean table
  - Return best model + comparison

Task detection:
  - Classification: target is binary or low-cardinality categorical
  - Regression:     target is continuous numeric
  - Clustering:     no target specified / "find groups / segments"
"""

import re
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np


# ── Task detection ────────────────────────────────────────────────────────────

CLASSIFICATION_KEYWORDS = [
    "classif", "predict class", "predict category", "predict label",
    "is it", "will it", "churn", "fraud", "spam", "disease",
    "survival", "default", "diagnosis", "sentiment",
]

REGRESSION_KEYWORDS = [
    "regress", "predict price", "predict cost", "predict value",
    "predict salary", "predict score", "forecast", "estimate",
    "how much", "how many", "revenue", "sales",
]

CLUSTERING_KEYWORDS = [
    "cluster", "segment", "group", "find groups", "find segments",
    "similar", "kmeans", "k-means", "unsupervised",
]


def detect_task_type(
    question: str,
    df: pd.DataFrame,
    target_col: Optional[str] = None,
) -> str:
    """
    Detect ML task type: 'classification', 'regression', or 'clustering'.
    """
    q = question.lower()

    # Keyword check first
    if any(kw in q for kw in CLUSTERING_KEYWORDS):
        return "clustering"
    if any(kw in q for kw in REGRESSION_KEYWORDS):
        return "regression"
    if any(kw in q for kw in CLASSIFICATION_KEYWORDS):
        return "classification"

    # Infer from target column
    if target_col and target_col in df.columns:
        col    = df[target_col].dropna()
        n_uniq = col.nunique()
        dtype  = df[target_col].dtype

        if dtype == "object" or n_uniq <= 10:
            return "classification"
        return "regression"

    return "classification"  # safe default


def get_model_candidates(task_type: str) -> List[Dict]:
    """
    Return list of model configs for the task type.
    Each dict: {name, class_name, import_from, params, save_name}
    """
    candidates = {
        "classification": [
            {
                "name":        "Random Forest",
                "class_name":  "RandomForestClassifier",
                "save_name":   "random_forest",
                # n_jobs=-1 uses all CPU cores — 3-4x faster than single-threaded
                "params":      "n_estimators=100, max_depth=10, random_state=42, n_jobs=-1",
                "metric":      "accuracy",
            },
            {
                "name":        "Logistic Regression",
                "class_name":  "LogisticRegression",
                "save_name":   "logistic_regression",
                "params":      "max_iter=1000, random_state=42, n_jobs=-1",
                "metric":      "accuracy",
            },
            {
                "name":        "Gradient Boosting",
                "class_name":  "GradientBoostingClassifier",
                "save_name":   "gradient_boosting",
                # 50 estimators instead of 100 for comparison — still accurate, 2x faster
                "params":      "n_estimators=50, max_depth=4, random_state=42",
                "metric":      "accuracy",
            },
        ],
        "regression": [
            {
                "name":        "Random Forest",
                "class_name":  "RandomForestRegressor",
                "save_name":   "random_forest_reg",
                "params":      "n_estimators=100, max_depth=10, random_state=42, n_jobs=-1",
                "metric":      "r2",
            },
            {
                "name":        "Gradient Boosting",
                "class_name":  "GradientBoostingRegressor",
                "save_name":   "gradient_boosting_reg",
                "params":      "n_estimators=50, max_depth=4, random_state=42",
                "metric":      "r2",
            },
            {
                "name":        "Linear Regression",
                "class_name":  "LinearRegression",
                "save_name":   "linear_regression",
                "params":      "n_jobs=-1",
                "metric":      "r2",
            },
        ],
        "clustering": [
            {
                "name":        "KMeans (k=3)",
                "class_name":  "KMeans",
                "save_name":   "kmeans_3",
                "params":      "n_clusters=3, random_state=42",
                "metric":      "inertia",
            },
            {
                "name":        "KMeans (k=4)",
                "class_name":  "KMeans",
                "save_name":   "kmeans_4",
                "params":      "n_clusters=4, random_state=42",
                "metric":      "inertia",
            },
            {
                "name":        "KMeans (k=5)",
                "class_name":  "KMeans",
                "save_name":   "kmeans_5",
                "params":      "n_clusters=5, random_state=42",
                "metric":      "inertia",
            },
        ],
    }
    return candidates.get(task_type, candidates["classification"])


def build_comparison_code(
    task_type: str,
    candidates: List[Dict],
    target_col: str,
    df_columns: List[str],
) -> str:
    """
    Generate Python code that trains all candidate models,
    compares them, and saves the best one.
    """
    models_code = []
    results_var = "model_results"

    # Preprocessing block (shared)
    prep = f"""
# ── Preprocessing ─────────────────────────────────────────────────────────────
target = '{target_col}'
if target not in df.columns:
    target = df.columns[-1]

y = df[target].copy()
X = df.drop(columns=[target], errors='ignore').copy()

id_cols = [c for c in X.columns if X[c].nunique() == len(X)]
X = X.drop(columns=id_cols, errors='ignore')

for col in X.select_dtypes(include=['object','category']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

X = X.fillna(X.median(numeric_only=True)).fillna(0)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

{results_var} = []
best_score = -999
best_model = None
best_name  = ''
"""
    models_code.append(prep)

    # Per-model training
    for m in candidates:
        if task_type == "classification":
            block = f"""
# ── {m['name']} ──
try:
    m_{m['save_name']} = {m['class_name']}({m['params']})
    m_{m['save_name']}.fit(X_train, y_train)
    score_{m['save_name']} = accuracy_score(y_test, m_{m['save_name']}.predict(X_test))
    cv_{m['save_name']} = cross_val_score(m_{m['save_name']}, X_scaled, y, cv=3, n_jobs=-1).mean()
    {results_var}.append({{'model': '{m['name']}', 'test_accuracy': round(score_{m['save_name']},4), 'cv_accuracy': round(cv_{m['save_name']},4)}})
    if score_{m['save_name']} > best_score:
        best_score = score_{m['save_name']}
        best_model = m_{m['save_name']}
        best_name  = '{m['name']}'
    print(f"{m['name']}: accuracy={{score_{m['save_name']}:.4f}}, cv={{cv_{m['save_name']}:.4f}}")
except Exception as e:
    print(f"{m['name']} failed: {{e}}")
"""
        elif task_type == "regression":
            block = f"""
# ── {m['name']} ──
try:
    m_{m['save_name']} = {m['class_name']}({m['params']})
    m_{m['save_name']}.fit(X_train, y_train)
    score_{m['save_name']} = r2_score(y_test, m_{m['save_name']}.predict(X_test))
    rmse_{m['save_name']}  = mean_squared_error(y_test, m_{m['save_name']}.predict(X_test))**0.5
    {results_var}.append({{'model': '{m['name']}', 'r2': round(score_{m['save_name']},4), 'rmse': round(rmse_{m['save_name']},4)}})
    if score_{m['save_name']} > best_score:
        best_score = score_{m['save_name']}
        best_model = m_{m['save_name']}
        best_name  = '{m['name']}'
    print(f"{m['name']}: R²={{score_{m['save_name']}:.4f}}, RMSE={{rmse_{m['save_name']}:.4f}}")
except Exception as e:
    print(f"{m['name']} failed: {{e}}")
"""
        else:  # clustering
            block = f"""
# ── {m['name']} ──
try:
    m_{m['save_name']} = {m['class_name']}({m['params']})
    m_{m['save_name']}.fit(X_scaled)
    inertia_{m['save_name']} = m_{m['save_name']}.inertia_
    {results_var}.append({{'model': '{m['name']}', 'inertia': round(inertia_{m['save_name']},2), 'n_clusters': {m['params'].split(',')[0].split('=')[1].strip()}}})
    print(f"{m['name']}: inertia={{inertia_{m['save_name']}:.2f}}")
    # Lower inertia = better clustering (track best)
    if best_score == -999 or inertia_{m['save_name']} < -best_score:
        best_score = -inertia_{m['save_name']}   # negate so "higher is better" logic works
        best_model = m_{m['save_name']}
        best_name  = '{m['name']}'
except Exception as e:
    print(f"{m['name']} failed: {{e}}")
"""
        models_code.append(block)

    # Save best model + print comparison
    # Save best model + print comparison
    save_block = f"""
# ── Results comparison ────────────────────────────────────────────────────────
results_df = pd.DataFrame({results_var})
print("\\n=== Model Comparison ===")
print(results_df.to_string(index=False))

# Save best model
if best_model is not None:
    import joblib, os, json
    os.makedirs('models', exist_ok=True)
    save_path = f'models/{{best_name.lower().replace(" ","_")}}_best.pkl'
    joblib.dump(best_model, save_path)
    print(f'MODEL_SAVED:{{save_path}}')
    print(f'Best model: {{best_name}} (score={{best_score:.4f}})')

# Output results for follow-up chart queries
import json as _json
print(f'MODEL_RESULTS:{{_json.dumps({results_var})}}')
"""
    models_code.append(save_block)
    return "\n".join(models_code)