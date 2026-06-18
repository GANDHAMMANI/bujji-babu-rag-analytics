"""
target_detector.py — Smart ML Target Column Detector for Bujji Babu CSV Agent

Problem:
  Current code always uses df.columns[-1] as target.
  This is wrong ~40% of the time.

  "Predict customer churn" → target is 'Churn' not the last column
  "Classify the diagnosis" → target is 'Diagnosis'
  "Build a model for survival" → target is 'Survived'

How it works:
  1. Extract candidate target words from the question (nouns after predict/classify/for)
  2. Fuzzy-match against actual column names
  3. Score by: exact match > substring match > semantic similarity > position heuristic
  4. Fallback to last column only if nothing matches

Handles:
  - "predict X"         → X is target
  - "classify X"        → X is target
  - "model for X"       → X is target
  - "X prediction"      → X is target
  - "target is X"       → X is target
  - Common ML datasets  → auto-recognise (Titanic, Iris, Diabetes...)
"""

import re
from typing import List, Optional, Tuple
import pandas as pd


# ── Extraction patterns ───────────────────────────────────────────────────────

EXTRACTION_PATTERNS = [
    r"predict(?:ing)?\s+([a-zA-Z_][a-zA-Z0-9_\s]{1,30}?)(?:\s+using|\s+from|\s+based|$|[,.])",
    r"classif(?:y|ying|ication)\s+(?:the\s+)?([a-zA-Z_][a-zA-Z0-9_\s]{1,25}?)(?:\s+using|\s+from|\s+based|$|[,.])",
    r"(?:target|label|output|dependent variable)\s+(?:is|column|variable|:)?\s*['\"]?([a-zA-Z_][a-zA-Z0-9_\s]{1,25}?)['\"]?(?:\s|$|[,.])",
    r"model\s+(?:for|to predict|to classify)\s+([a-zA-Z_][a-zA-Z0-9_\s]{1,25}?)(?:\s+using|\s+from|\s+based|$|[,.])",
    r"([a-zA-Z_][a-zA-Z0-9_\s]{1,25}?)\s+(?:prediction|classification|forecast|detection|recognition)",
    r"forecast(?:ing)?\s+([a-zA-Z_][a-zA-Z0-9_\s]{1,25}?)(?:\s+using|\s+from|\s+based|$|[,.])",
    r"detect(?:ing)?\s+([a-zA-Z_][a-zA-Z0-9_\s]{1,25}?)(?:\s+using|\s+from|\s+based|$|[,.])",
]

# ── Known ML dataset targets ──────────────────────────────────────────────────

KNOWN_TARGETS = {
    # Classification
    "survived":    ["survived", "survival"],
    "species":     ["species", "iris", "flower"],
    "churn":       ["churn", "churned", "attrition"],
    "diagnosis":   ["diagnosis", "diagnose", "disease", "condition"],
    "default":     ["default", "loan default", "credit default"],
    "fraud":       ["fraud", "fraudulent", "is_fraud"],
    "sentiment":   ["sentiment", "opinion", "review"],
    "income":      ["income", ">50k", "salary class"],
    "target":      ["target"],
    "label":       ["label", "class", "category"],
    "outcome":     ["outcome", "result", "output"],
    "approved":    ["approved", "approval", "accepted"],
    # Regression
    "price":       ["price", "cost", "value", "sale price", "house price"],
    "medv":        ["medv", "house value", "median value"],
    "charges":     ["charges", "insurance cost", "medical charges"],
    "mpg":         ["mpg", "fuel efficiency", "miles per gallon"],
}


def _normalise(text: str) -> str:
    """Lowercase, remove punctuation, collapse spaces."""
    return re.sub(r'[^a-z0-9\s]', '', text.lower()).strip()


def _extract_candidates(question: str) -> List[str]:
    """Extract candidate target words/phrases from the question."""
    candidates = []
    q = question.lower()

    for pattern in EXTRACTION_PATTERNS:
        matches = re.findall(pattern, q)
        for m in matches:
            cleaned = _normalise(m.strip())
            if cleaned and len(cleaned) > 1:
                candidates.append(cleaned)

    return list(dict.fromkeys(candidates))  # deduplicate preserving order


def _fuzzy_match(candidate: str, columns: List[str]) -> Optional[Tuple[str, float]]:
    """
    Match a candidate string against actual column names.
    Returns (best_column, score) or None.

    Scoring:
      1.0  exact match (case-insensitive)
      0.9  normalised exact match
      0.8  column name is substring of candidate or vice versa
      0.6  partial word overlap >= 0.5
      0.0  no match
    """
    cand_norm = _normalise(candidate)
    best_col   = None
    best_score = 0.0

    for col in columns:
        col_norm = _normalise(col)

        # Exact match
        if cand_norm == col_norm:
            return col, 1.0

        # One is substring of the other
        if cand_norm in col_norm or col_norm in cand_norm:
            score = 0.8
            if score > best_score:
                best_score = score
                best_col   = col
            continue

        # Word overlap
        cand_words = set(cand_norm.split())
        col_words  = set(col_norm.split())
        if cand_words and col_words:
            overlap = len(cand_words & col_words) / max(len(cand_words), len(col_words))
            if overlap >= 0.5:
                score = 0.5 + overlap * 0.2
                if score > best_score:
                    best_score = score
                    best_col   = col

    return (best_col, best_score) if best_col else None


def _known_target_match(question: str, columns: List[str]) -> Optional[str]:
    """
    Check if the question mentions a known ML dataset target keyword
    and that column (or similar) exists.
    """
    q_norm    = _normalise(question)
    col_norms = {_normalise(c): c for c in columns}

    for col_key, keywords in KNOWN_TARGETS.items():
        for kw in keywords:
            if kw in q_norm:
                # Check if a column matching col_key exists
                for cn, original in col_norms.items():
                    if col_key in cn or cn in col_key:
                        return original

    return None


def _heuristic_target(df: pd.DataFrame) -> str:
    """
    Heuristic fallback when nothing else matches:
    - Prefer binary columns (2 unique values) — most likely classification target
    - Prefer columns named 'target', 'label', 'class', 'y', 'output'
    - Fall back to last column
    """
    columns    = list(df.columns)
    col_norms  = [_normalise(c) for c in columns]

    # Check for common target names
    priority_names = ["target", "label", "class", "y", "output",
                      "result", "outcome", "response"]
    for pname in priority_names:
        for col, cn in zip(columns, col_norms):
            if cn == pname or pname in cn:
                return col

    # Prefer binary columns
    binary_cols = [c for c in columns if df[c].nunique() == 2]
    if binary_cols:
        # Among binary, prefer last (convention)
        return binary_cols[-1]

    # Last column convention
    return columns[-1]


# ── Public API ────────────────────────────────────────────────────────────────

def detect_target(
    question: str,
    df: pd.DataFrame,
    threshold: float = 0.6,
) -> Tuple[str, str]:
    """
    Detect the most likely ML target column for a given question + dataframe.

    Args:
        question:   User's question
        df:         The loaded dataframe
        threshold:  Minimum fuzzy match score to accept (default 0.6)

    Returns:
        (target_column: str, method: str)
        method is one of: "exact", "fuzzy", "known", "heuristic", "last_column"
    """
    columns = list(df.columns)

    if not columns:
        return "", "empty"

    # ── 1. Extract candidates from question ───────────────────────────────────
    candidates = _extract_candidates(question)

    # ── 2. Try exact / fuzzy match for each candidate ─────────────────────────
    best_col   = None
    best_score = 0.0
    best_method = ""

    for candidate in candidates:
        result = _fuzzy_match(candidate, columns)
        if result:
            col, score = result
            if score > best_score:
                best_score  = score
                best_col    = col
                best_method = "exact" if score == 1.0 else "fuzzy"

    if best_col and best_score >= threshold:
        print(f"[TARGET] '{question[:50]}' → '{best_col}' (method={best_method}, score={best_score:.2f})")
        return best_col, best_method

    # ── 3. Known dataset target matching ──────────────────────────────────────
    known = _known_target_match(question, columns)
    if known:
        print(f"[TARGET] Known target match: '{known}'")
        return known, "known"

    # ── 4. Heuristic fallback ─────────────────────────────────────────────────
    heuristic = _heuristic_target(df)
    if heuristic != columns[-1]:
        print(f"[TARGET] Heuristic target: '{heuristic}'")
        return heuristic, "heuristic"

    # ── 5. Last column fallback ───────────────────────────────────────────────
    print(f"[TARGET] Fallback to last column: '{columns[-1]}'")
    return columns[-1], "last_column"


def target_detection_note(target: str, method: str) -> str:
    """
    Return a brief note for the LLM about how target was detected.
    Helps the LLM understand why this column was chosen.
    """
    notes = {
        "exact":       f"Target column '{target}' was explicitly mentioned in your question.",
        "fuzzy":       f"Target column '{target}' was inferred from your question.",
        "known":       f"Target column '{target}' was recognised as a standard ML target.",
        "heuristic":   f"Target column '{target}' was selected as it appears to be the outcome variable.",
        "last_column": f"Target column '{target}' was selected by convention (last column).",
    }
    return notes.get(method, f"Target column: '{target}'")