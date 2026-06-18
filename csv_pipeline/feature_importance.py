"""
feature_importance.py — Feature Importance for Bujji Babu CSV Agent

Automatically appends feature importance plotting code
after every trained model that supports it.

Supports:
  - Tree-based models: feature_importances_ attribute
  - Linear models: coef_ attribute
  - All others: permutation importance fallback
"""


def get_feature_importance_code(model_class: str, target_col: str) -> str:
    """
    Return Python code to compute and plot feature importance
    for the given model class.

    Args:
        model_class: e.g. 'RandomForestClassifier', 'LogisticRegression'
        target_col:  target column name for axis labels

    Returns:
        Python code string to append after model training
    """

    tree_models = [
        "RandomForestClassifier", "RandomForestRegressor",
        "GradientBoostingClassifier", "GradientBoostingRegressor",
        "DecisionTreeClassifier", "DecisionTreeRegressor",
        "ExtraTreesClassifier", "ExtraTreesRegressor",
    ]

    linear_models = [
        "LogisticRegression", "LinearRegression",
        "Ridge", "Lasso", "ElasticNet",
    ]

    if model_class in tree_models:
        return f"""
# ── Feature Importance ────────────────────────────────────────────────────────
try:
    fi_series = pd.Series(model.feature_importances_, index=X.columns)
    fi_series = fi_series.sort_values(ascending=True).tail(15)
    fi_df = pd.DataFrame({{'feature': fi_series.index, 'importance': fi_series.values}})
    fig = px.bar(
        fi_df, x='importance', y='feature', orientation='h',
        title='Feature Importance — {model_class}',
        labels={{'importance': 'Importance Score', 'feature': 'Feature'}},
    )
    fig.update_layout(
        template='plotly_white', height=500,
        font=dict(family='DM Sans, sans-serif', size=12),
        margin=dict(t=60, r=30, b=60, l=150),
        colorway=['#2c5f8a'],
    )
    print('PLOTLY_JSON:' + fig.to_json())
except Exception as e:
    print(f'Feature importance plot failed: {{e}}')
"""

    elif model_class in linear_models:
        return f"""
# ── Feature Importance (Coefficients) ────────────────────────────────────────
try:
    coefs = model.coef_.flatten() if hasattr(model.coef_, 'flatten') else model.coef_
    fi_series = pd.Series(abs(coefs), index=X.columns)
    fi_series = fi_series.sort_values(ascending=True).tail(15)
    fi_df = pd.DataFrame({{'feature': fi_series.index, 'importance': fi_series.values}})
    fig = px.bar(
        fi_df, x='importance', y='feature', orientation='h',
        title='Feature Coefficients — {model_class}',
        labels={{'importance': '|Coefficient|', 'feature': 'Feature'}},
    )
    fig.update_layout(
        template='plotly_white', height=500,
        font=dict(family='DM Sans, sans-serif', size=12),
        margin=dict(t=60, r=30, b=60, l=150),
        colorway=['#b8953a'],
    )
    print('PLOTLY_JSON:' + fig.to_json())
except Exception as e:
    print(f'Coefficient plot failed: {{e}}')
"""

    else:
        # Permutation importance fallback
        return f"""
# ── Feature Importance (Permutation) ─────────────────────────────────────────
try:
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
    fi_series = pd.Series(perm.importances_mean, index=X.columns)
    fi_series = fi_series.sort_values(ascending=True).tail(15)
    fi_df = pd.DataFrame({{'feature': fi_series.index, 'importance': fi_series.values}})
    fig = px.bar(
        fi_df, x='importance', y='feature', orientation='h',
        title='Permutation Importance — {model_class}',
        labels={{'importance': 'Mean Importance', 'feature': 'Feature'}},
    )
    fig.update_layout(
        template='plotly_white', height=500,
        font=dict(family='DM Sans, sans-serif', size=12),
        margin=dict(t=60, r=30, b=60, l=150),
        colorway=['#27ae60'],
    )
    print('PLOTLY_JSON:' + fig.to_json())
except Exception as e:
    print(f'Permutation importance failed: {{e}}')
"""


def get_universal_feature_importance_code(model_var: str, target_col: str) -> str:
    """
    Generate feature importance code that detects the model type at RUNTIME
    and uses the right method:
      - Tree models  → .feature_importances_
      - Linear models → .coef_ (absolute values)
      - Everything else → permutation importance

    Args:
        model_var:  Name of the Python variable holding the fitted model (e.g. 'best_model')
        target_col: Target column name — used in chart title only
    """
    return f"""
# ── Feature Importance (auto-detect model type) ───────────────────────────────
try:
    _fi_model = {model_var}
    _model_type = type(_fi_model).__name__
    _tree_types = {{'RandomForestClassifier','RandomForestRegressor',
                   'GradientBoostingClassifier','GradientBoostingRegressor',
                   'DecisionTreeClassifier','DecisionTreeRegressor',
                   'ExtraTreesClassifier','ExtraTreesRegressor'}}
    _linear_types = {{'LogisticRegression','LinearRegression','Ridge','Lasso','ElasticNet'}}

    if _model_type in _tree_types:
        _fi_vals = _fi_model.feature_importances_
        _fi_label = 'Importance Score'
        _chart_color = '#2c5f8a'
    elif _model_type in _linear_types:
        _coefs = _fi_model.coef_.flatten() if hasattr(_fi_model.coef_, 'flatten') else _fi_model.coef_
        _fi_vals = abs(_coefs)
        _fi_label = '|Coefficient|'
        _chart_color = '#b8953a'
    else:
        from sklearn.inspection import permutation_importance as _pi
        _perm = _pi(_fi_model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
        _fi_vals = _perm.importances_mean
        _fi_label = 'Mean Permutation Importance'
        _chart_color = '#27ae60'

    _fi_series = pd.Series(_fi_vals, index=X.columns).sort_values(ascending=True).tail(15)
    _fi_df = pd.DataFrame({{'feature': _fi_series.index.tolist(), 'importance': _fi_series.values.tolist()}})
    fig = px.bar(
        _fi_df, x='importance', y='feature', orientation='h',
        title=f'Feature Importance — {{_model_type}}',
        labels={{'importance': _fi_label, 'feature': 'Feature'}},
    )
    fig.update_layout(
        template='plotly_white', height=max(400, len(_fi_df) * 32 + 100),
        font=dict(family='DM Sans, sans-serif', size=12),
        margin=dict(t=60, r=30, b=60, l=160),
        colorway=[_chart_color],
    )
    print('PLOTLY_JSON:' + fig.to_json())
except Exception as _fi_err:
    print(f'Feature importance plot failed: {{_fi_err}}')
"""


def extract_model_class(question: str, code: str) -> str:
    """
    Detect which model class was used from the question or generated code.
    """
    import re
    # Check code first (most reliable)
    for cls in [
        "RandomForestClassifier", "RandomForestRegressor",
        "GradientBoostingClassifier", "GradientBoostingRegressor",
        "LogisticRegression", "LinearRegression",
        "DecisionTreeClassifier", "DecisionTreeRegressor",
        "SVC", "SVR", "GaussianNB", "KMeans",
    ]:
        if cls in code:
            return cls

    # Fall back to question keywords
    q = question.lower()
    if "random forest" in q:
        return "RandomForestClassifier"
    if "logistic" in q:
        return "LogisticRegression"
    if "gradient boost" in q:
        return "GradientBoostingClassifier"
    if "decision tree" in q:
        return "DecisionTreeClassifier"
    if "linear regression" in q:
        return "LinearRegression"

    return "RandomForestClassifier"  # safe default