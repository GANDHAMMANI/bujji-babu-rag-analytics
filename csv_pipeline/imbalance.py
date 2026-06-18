"""
imbalance.py — Class Imbalance Detection for Bujji Babu CSV Agent

Detects class imbalance in classification targets and
returns the appropriate handling strategy as code injection.

Strategies:
  - Mild imbalance (20-40% minority): class_weight='balanced'
  - Severe imbalance (<20% minority): SMOTE oversampling
  - Extreme imbalance (<5% minority): SMOTE + undersampling note
"""

from typing import Tuple, Optional
import pandas as pd


MILD_THRESHOLD    = 0.20   # below this minority ratio → severe
SEVERE_THRESHOLD  = 0.05   # below this → extreme


def detect_imbalance(
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[str, float, dict]:
    """
    Detect class imbalance in target column.

    Returns:
        (severity, minority_ratio, class_counts)
        severity: 'balanced' | 'mild' | 'severe' | 'extreme'
    """
    if target_col not in df.columns:
        return "unknown", 0.5, {}

    counts     = df[target_col].value_counts()
    total      = len(df)
    minority   = counts.min()
    ratio      = minority / total

    class_counts = counts.to_dict()
    class_counts = {str(k): int(v) for k, v in class_counts.items()}

    if ratio < SEVERE_THRESHOLD:
        severity = "extreme"
    elif ratio < MILD_THRESHOLD:
        severity = "severe"
    elif ratio < 0.40:
        severity = "mild"
    else:
        severity = "balanced"

    print(f"[IMBALANCE] target='{target_col}' minority_ratio={ratio:.3f} severity={severity}")
    return severity, round(ratio, 3), class_counts


def get_imbalance_code_injection(severity: str, target_col: str) -> str:
    """
    Return Python code snippet to inject into ML pipeline for handling imbalance.
    """
    if severity == "balanced":
        return "# No class imbalance detected — proceeding normally\n"

    if severity == "mild":
        return (
            f"# Mild class imbalance detected in '{target_col}' — using class_weight='balanced'\n"
            "# This adjusts model weights to compensate for minority class\n"
        )

    if severity in ("severe", "extreme"):
        return f"""
# {'Severe' if severity == 'severe' else 'Extreme'} class imbalance in '{target_col}'
# Applying SMOTE oversampling to balance classes
try:
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {{pd.Series(y_train).value_counts().to_dict()}}")
except ImportError:
    print("Note: imbalanced-learn not installed. Using class_weight='balanced' instead.")
    # class_weight='balanced' will be applied in model params
"""

    return ""


def imbalance_note(severity: str, ratio: float, counts: dict) -> str:
    """Short note to include in LLM prompt."""
    if severity == "balanced":
        return ""
    counts_str = ", ".join(f"'{k}': {v}" for k, v in list(counts.items())[:5])
    advice = "Use class_weight='balanced' in all classifiers." if severity == "mild" else "Apply SMOTE before training."
    return (
        f"\nCLASS IMBALANCE DETECTED: {severity} ({ratio:.1%} minority class). "
        f"Class distribution: {{{counts_str}}}. "
        f"{advice}\n"
    )