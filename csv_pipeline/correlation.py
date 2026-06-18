"""
correlation.py — Correlation Analysis for Bujji Babu CSV Agent

Adds to stat artifact:
  - Full correlation matrix (numeric columns)
  - Top correlated pairs ranked by absolute correlation
  - Flags multicollinearity (|r| > 0.85)
  - Flags columns with zero/near-zero variance
  - Target correlation ranking (if target known)
"""

import json
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


# ── Thresholds ────────────────────────────────────────────────────────────────

HIGH_CORR_THRESHOLD  = 0.75   # flag as highly correlated
MULTI_COLLIN_THRESH  = 0.85   # flag as multicollinear
LOW_VARIANCE_THRESH  = 0.01   # near-zero variance threshold
MAX_CORR_PAIRS       = 10     # top N pairs to return


def build_correlation_artifact(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
) -> Dict:
    """
    Build a correlation analysis artifact.

    Returns:
        {
            "matrix":          dict  — full correlation matrix as nested dict
            "top_pairs":       list  — top correlated pairs [{col1, col2, r}]
            "multicollinear":  list  — pairs with |r| > 0.85
            "low_variance":    list  — columns with near-zero variance
            "target_corr":     list  — correlation with target, sorted desc
        }
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if len(num_cols) < 2:
        return {
            "matrix":         {},
            "top_pairs":      [],
            "multicollinear": [],
            "low_variance":   [],
            "target_corr":    [],
            "note":           "Not enough numeric columns for correlation analysis.",
        }

    # ── Correlation matrix ────────────────────────────────────────────────────
    corr_df = df[num_cols].corr(method="pearson")

    # Convert to nested dict with rounded values
    matrix = {
        col: {
            other: round(float(corr_df.loc[col, other]), 3)
            for other in num_cols
        }
        for col in num_cols
    }

    # ── Top correlated pairs ──────────────────────────────────────────────────
    pairs = []
    seen  = set()
    for i, col1 in enumerate(num_cols):
        for col2 in num_cols[i+1:]:
            key = tuple(sorted([col1, col2]))
            if key in seen:
                continue
            seen.add(key)
            r = corr_df.loc[col1, col2]
            if pd.isna(r):
                continue
            pairs.append({
                "col1":    col1,
                "col2":    col2,
                "r":       round(float(r), 3),
                "abs_r":   round(abs(float(r)), 3),
                "strength": _corr_strength(abs(float(r))),
            })

    pairs.sort(key=lambda x: x["abs_r"], reverse=True)
    top_pairs = pairs[:MAX_CORR_PAIRS]

    # ── Multicollinear pairs ──────────────────────────────────────────────────
    multicollinear = [
        p for p in pairs
        if p["abs_r"] >= MULTI_COLLIN_THRESH and p["col1"] != p["col2"]
    ]

    # ── Low variance columns ──────────────────────────────────────────────────
    low_variance = []
    for col in num_cols:
        var = float(df[col].var())
        if var < LOW_VARIANCE_THRESH:
            low_variance.append({
                "col":      col,
                "variance": round(var, 6),
            })

    # ── Target correlation ────────────────────────────────────────────────────
    target_corr = []
    if target_col and target_col in num_cols:
        for col in num_cols:
            if col == target_col:
                continue
            r = corr_df.loc[col, target_col]
            if not pd.isna(r):
                target_corr.append({
                    "col":      col,
                    "r":        round(float(r), 3),
                    "abs_r":    round(abs(float(r)), 3),
                    "strength": _corr_strength(abs(float(r))),
                })
        target_corr.sort(key=lambda x: x["abs_r"], reverse=True)

    return {
        "matrix":         matrix,
        "top_pairs":      top_pairs,
        "multicollinear": multicollinear,
        "low_variance":   low_variance,
        "target_corr":    target_corr,
        "num_cols":       num_cols,
        "col_count":      len(num_cols),
    }


def _corr_strength(abs_r: float) -> str:
    if abs_r >= 0.85: return "very strong"
    if abs_r >= 0.70: return "strong"
    if abs_r >= 0.50: return "moderate"
    if abs_r >= 0.30: return "weak"
    return "negligible"


def correlation_summary(corr_artifact: Dict) -> str:
    """
    Natural language summary of correlation findings for LLM.
    """
    parts = []

    top = corr_artifact.get("top_pairs", [])
    if top:
        top3 = top[:3]
        pairs_str = ", ".join(
            f"'{p['col1']}' & '{p['col2']}' (r={p['r']})"
            for p in top3
        )
        parts.append(f"Top correlations: {pairs_str}")

    mc = corr_artifact.get("multicollinear", [])
    if mc:
        mc_str = ", ".join(f"'{p['col1']}'/'{p['col2']}'" for p in mc[:3])
        parts.append(f"Potential multicollinearity: {mc_str}")

    lv = corr_artifact.get("low_variance", [])
    if lv:
        lv_str = ", ".join(f"'{c['col']}'" for c in lv)
        parts.append(f"Near-zero variance columns: {lv_str}")

    tc = corr_artifact.get("target_corr", [])
    if tc:
        tc_str = ", ".join(f"'{c['col']}' ({c['r']})" for c in tc[:3])
        parts.append(f"Strongest predictors: {tc_str}")

    return " | ".join(parts) if parts else "No significant correlations found."