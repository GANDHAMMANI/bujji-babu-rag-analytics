"""
nl_filter.py — Natural Language Filter Queries for Bujji Babu CSV Agent

Converts plain English filters to pandas expressions and executes them.

Examples:
  "show patients over 50"          → df[df['Age'] > 50]
  "filter rows where income > 5000" → df[df['Income'] > 5000]
  "show only female customers"     → df[df['Gender'] == 'Female']
  "find rows with missing values"  → df[df.isnull().any(axis=1)]
  "top 10 by salary"               → df.nlargest(10, 'Salary')
  "rows where diagnosis is cancer" → df[df['Diagnosis'] == 'cancer']
"""

import re
from typing import Dict, Optional, Tuple
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
from llm_client import get_client as _get_client, get_model as _get_model
_groq = _get_client()


# ── Intent detection ──────────────────────────────────────────────────────────

# Specific multi-word patterns that clearly signal a row-filter intent
FILTER_KEYWORDS = [
    "filter", "where ", "rows where", "records where",
    "only rows", "only records", "only patients", "only customers",
    "only female", "only male", "only the", "show only",
    "patients with", "customers with", "employees with", "entries where",
    "greater than", "less than", "equal to", "more than",
    "over ", "under ", "above ", "below ",
    "missing values", "null values", "has nulls", "not null",
    "between ", "in range",
    "top ", "bottom ",
    "rows with", "records with",
]

# Single-word keywords that are too broad on their own — only use when paired
# with a comparison operator or column reference pattern
_WEAK_KEYWORDS = ["show", "find", "get", "list", "display", "select"]
_COMPARISON_PATTERNS = re.compile(
    r"(\bwhere\b|\bwith\b|\bgreater\b|\bless\b|\bover\b|\bunder\b"
    r"|\babove\b|\bbelow\b|\bmissing\b|\bnull\b|\btop\s+\d|\bbottom\s+\d"
    r"|\bfirst\s+\d|\blast\s+\d|\b==\b|\b>=\b|\b<=\b|\b!=\b)"
)


# Keywords that are clearly analytics/stats intent — NOT row-filtering.
# If any of these appear in the question, FILTER_KEYWORDS like "between "
# must not override them (e.g. "correlation between X and Y" is NOT a filter).
_ANALYTICS_OVERRIDE = [
    "correlation", "correlat",
    "distribution", "histogram",
    "heatmap", "scatter",
    "average", "mean ", "median", "std dev", "standard deviation",
    "regression", "classification", "cluster",
    "model", "train", "predict",
    "analysis", "analyse", "analyze",
    "overview", "summary", "summarize",
]


def is_filter_query(question: str) -> bool:
    """
    Check if question is a row-filter/selection query.
    Uses specific multi-word patterns first; falls back to weak keywords
    only when paired with a comparison operator or structural keyword.

    Analytics intent (correlation, distribution, model, etc.) always wins —
    a question like "correlation between X and Y" must NOT be treated as a
    row filter just because it contains the word "between".
    """
    q = question.lower()

    # Analytics intent override — these are never row filters
    if any(kw in q for kw in _ANALYTICS_OVERRIDE):
        return False

    # Specific patterns — high confidence
    if any(kw in q for kw in FILTER_KEYWORDS):
        return True
    # Weak single-word + comparison pattern — e.g. "show rows where age > 30"
    if any(kw in q for kw in _WEAK_KEYWORDS) and bool(_COMPARISON_PATTERNS.search(q)):
        return True
    return False


# ── LLM-based filter generation ───────────────────────────────────────────────

def generate_filter_code(
    question: str,
    df: pd.DataFrame,
    eda_summary: str,
) -> str:
    """
    Generate pandas filter code from natural language question.
    Returns executable Python code string.
    """
    col_info = "\n".join(
        f"  '{c}' ({df[c].dtype}) — sample: {df[c].dropna().head(3).tolist()}"
        for c in df.columns
    )

    prompt = f"""You are a pandas expert. Convert this natural language filter to pandas code.

Dataset columns:
{col_info}

Question: "{question}"

Rules:
- Output ONLY raw Python code, no markdown, no explanation
- Result must be stored in variable: filtered_df = df[...]
- For top-N: filtered_df = df.nlargest(N, 'col')
- For missing values: filtered_df = df[df.isnull().any(axis=1)]
- For string comparison: use case-insensitive where possible: df['col'].str.lower() == 'value'
- Always print the shape: print(f"Filtered: {{len(filtered_df)}} rows")
- Always print the head: print(filtered_df.head(10).to_string())
- After printing, set: result_df = filtered_df (for export)

Code:"""

    try:
        resp = _groq.chat.completions.create(
            model=_get_model("strong"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.05,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown if present
        fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
        return "\n".join(fenced).strip() if fenced else raw
    except Exception as e:
        print(f"[NL_FILTER] LLM failed: {e}")
        return ""


# ── Result formatter ──────────────────────────────────────────────────────────

def format_filter_result(output: str, question: str) -> Dict:
    """
    Parse the filter output and return structured result for frontend.
    """
    lines  = output.strip().splitlines()
    shape_line = next((l for l in lines if l.startswith("Filtered:")), "")
    table_lines = [l for l in lines if not l.startswith("Filtered:")]
    table_str   = "\n".join(table_lines).strip()

    return {
        "type":    "filter_result",
        "summary": shape_line,
        "table":   table_str,
    }