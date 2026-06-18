"""
csv_agent.py — Bujji Babu CSV Intelligence Agent (v3)

Key fixes:
- _wants_chart() covers ALL visual keywords: learning curve, roc, confusion matrix, etc.
- ALL chart code forced to PLOTLY_JSON print — fig.show() never called
- exec() replaced with subprocess + 60s timeout — no GUI windows, no server hangs
- Histogram fix: low-cardinality columns use px.bar(value_counts) not px.histogram
- df_sample passed to LLM so it sees real data values
- GaussianNB added to imports
- Save preferences per session
"""

import os
import io
import re
import sys
import json
import hashlib
import textwrap
import traceback
import subprocess
import tempfile
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from groq import Groq
from target_detector import detect_target, target_detection_note
from correlation import build_correlation_artifact, correlation_summary
from autofix import surgical_fix
from nl_filter import is_filter_query, generate_filter_code, format_filter_result
from model_selector import detect_task_type, get_model_candidates, build_comparison_code
from imbalance import detect_imbalance, imbalance_note
from feature_importance import get_feature_importance_code, get_universal_feature_importance_code, extract_model_class
from llm_client import get_client, get_model, chat_text
from exporter import is_export_query, get_export_code, register_for_export, get_export_path
from csv_session import save_csv_session, load_csv_session, save_csv_turn, get_csv_chat_history, delete_csv_session, list_csv_sessions
from dotenv import load_dotenv

load_dotenv()

groq_client = get_client()  # managed by llm_client.py — switches online/offline

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

_sessions: Dict[str, Any] = {}

# Appended to system prompts so the LLM naturally invites continued conversation.
_CONV_TAIL = (
    "\n\nAfter answering, add one short natural sentence that invites the user "
    "to keep exploring — offer to go deeper on a specific finding from your answer, "
    "or suggest a related analysis they might want to try. "
    "Make it specific to what you just said (e.g. 'Want me to break this down by category?' "
    "not 'Let me know if you have more questions.'). "
    "Do NOT prefix it with 'Follow-up:' or any label — just write it naturally."
)


# ── ID ────────────────────────────────────────────────────────────────────────

def _csv_id(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()[:10]


# ── Follow-up resolver ────────────────────────────────────────────────────────

_VAGUE_RE = re.compile(
    r"^(i want |give me |show me |tell me |can you |please )?"
    r"(more|more detail|more details|in more detail|in detail|detailed|"
    r"elaborate|explain more|explain further|go deeper|expand|expand on that|"
    r"dig deeper|be more specific|more information|more info|more analysis|"
    r"more breakdown|break it down|further analysis|deeper analysis|"
    r"what else|more about that|more about this|tell me more|go on|continue)"
    r"[.!?]?$",
    re.IGNORECASE,
)

# Short confirmation phrases that mean "yes, do what you just offered"
_CONFIRM_RE = re.compile(
    r"^(yes|yeah|sure|ok|okay|go ahead|do it|proceed|sounds good|please do|"
    r"absolutely|yep|yup|do that|let's do it|let's go|go for it|"
    r"please|do so|that sounds good|yes please|yes do it|sounds great)[.!?!]*$",
    re.IGNORECASE,
)

def _is_vague_followup(q: str) -> bool:
    return bool(_VAGUE_RE.match(q.strip())) or bool(_CONFIRM_RE.match(q.strip()))


def _resolve_followup(question: str, last_q: str, last_a: str) -> str:
    """Rewrite a vague follow-up as a specific question using the previous turn."""
    try:
        resp = get_client().chat.completions.create(
            model=get_model("fast"),
            messages=[
                {"role": "system", "content": (
                    "You are resolving a follow-up data analysis question. "
                    "Given the previous question and answer, rewrite the follow-up "
                    "as a specific, self-contained analysis request. "
                    "Return ONLY the rewritten question — no explanation, no prefix. "
                    "Keep it under 20 words."
                )},
                {"role": "user", "content": (
                    f"Previous question: {last_q}\n"
                    f"Previous answer: {last_a[:400]}\n"
                    f"Follow-up: {question}\n"
                    f"Rewritten:"
                )},
            ],
            max_tokens=60,
            temperature=0.1,
        )
        rewritten = resp.choices[0].message.content.strip().strip('"\'')
        print(f"[CSV RESOLVER] '{question}' → '{rewritten}'")
        return rewritten
    except Exception as e:
        print(f"[CSV RESOLVER] Failed: {e}")
        return question


# ── EDA summary ───────────────────────────────────────────────────────────────

def _build_eda_summary(df: pd.DataFrame) -> str:
    buf = []
    buf.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    buf.append(f"Columns (exact names, use verbatim): {list(df.columns)}")
    buf.append(f"Dtypes: {df.dtypes.astype(str).to_dict()}")
    nulls = {k: int(v) for k, v in df.isnull().sum().items() if v > 0}
    buf.append(f"Nulls: {nulls or 'none'}")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        desc = df[num_cols].describe().round(3)
        buf.append(f"Numeric stats:\n{json.dumps(desc.to_dict())}")
        for col in num_cols:
            n_unique = df[col].nunique()
            if n_unique <= 15:
                vals = sorted(df[col].dropna().unique().tolist())
                buf.append(
                    f"DISCRETE COLUMN NOTE: '{col}' has only {n_unique} unique values {vals} "
                    f"— use safe_value_counts(df['{col}']) then px.bar(counts, x='value', y='count') "
                    f"NOT px.histogram for this column — histogram will be EMPTY"
                )

    for col in df.select_dtypes(include=["object", "category"]).columns[:5]:
        buf.append(f"'{col}' top values: {df[col].value_counts().head(5).to_dict()}")

    return "\n".join(buf)


# ── Intent detection ──────────────────────────────────────────────────────────

def _wants_feature_importance(q: str) -> bool:
    kw = ["feature importance", "feature_importance", "important features",
          "which features", "top features", "feature weights", "feature scores"]
    return any(k in q.lower() for k in kw)


def _wants_chart(q: str) -> bool:
    kw = [
        "chart", "plot", "graph", "histogram", "bar", "pie", "scatter",
        "heatmap", "box", "line", "distribution", "visuali", "show me",
        "display", "draw", "correlation matrix",
        "learning curve", "roc curve", " roc", "roc ", "auc",
        "confusion matrix", "feature importance", "feature_importance",
        "precision recall", "residual", "calibration curve",
        "decision boundary", "cluster plot", "elbow",
        "pairplot", "pair plot", "violin", "swarm",
        "dendrogram", "dendogram", "wordcloud", "word cloud",
    ]
    return any(k in q.lower() for k in kw)


def _is_chart_followup(q: str) -> bool:
    """Detect if user wants to change chart type from previous query."""
    kw = [
        "not scatter", "use bar", "use line", "change to",
        "instead of scatter", "make it a bar", "bar instead",
        "line instead", "bar chart instead", "no scatter",
    ]
    return any(k in q.lower() for k in kw)


def _wants_model(q: str) -> bool:
    kw = [
        "train", "classif", "predict", "regression", "cluster",
        "random forest", "xgboost", "svm", "logistic", "decision tree",
        "machine learning", "ml model", "build a model", "pkl", "naive bayes",
        "gradient boost", "linear regression", "kmeans", "k-means",
        "fit a", "run a",
        # "model" alone is intentionally omitted — too broad ("what model can we build?")
        # "apply" is intentionally omitted — "apply a filter" / "apply insights" would mismatch
    ]
    return any(k in q.lower() for k in kw)


def _wants_analysis(q: str) -> bool:
    kw = [
        # Core
        "analys", "overview", "summary", "summarize", "summarise",
        "describe", "tell me about", "tell me", "explain",
        "what is in", "explore", "statistic", "stat ", "stats",
        "insight", "profile", "eda", "data quality",
        # Natural phrasings
        "about the dataset", "about this dataset",
        "about the data", "about this data",
        "about the csv", "explain the", "explain me",
        "what does this", "what do we have",
        "give me an idea", "understand the",
        "information about", "info about",
        "what kind of", "what type of",
        "dataset look", "data look", "how is the data",
    ]
    return any(k in q.lower() for k in kw)


def _wants_recommendation(q: str) -> bool:
    """
    Detect model recommendation requests — analyse the dataset and advise,
    but do NOT train anything yet.

    Strategy: keyword list PLUS a regex that catches any
    "what/which [words] model [question phrase]" pattern so we don't
    need a separate entry for every adjective the user might insert
    (e.g. "what classification model", "which regression model can we build").
    """
    q_lower = q.lower()

    # ── 1. Direct keyword matches ─────────────────────────────────────────────
    kw = [
        "which model", "what model", "best model", "recommend a model",
        "suggest a model", "which algorithm", "what algorithm",
        "which classifier", "which regressor", "works best",
        "should i use", "what should i train", "which one to use",
        "what kind of model", "what type of model",
        "what models can", "which models can",
        "what models should", "which models would",
        "suitable model", "appropriate model", "good model for",
        "can we build a model", "what can we predict",
        "is it possible to", "can we predict",
        # Feasibility phrasings — asking IF/WHAT we can train, not commanding it
        "can we train", "could we train", "should we train",
        "what can we train", "what type of model can",
        "what kind of model can", "what model can we",
        # "what can we do / what we can do with the data"
        "what can we do", "what can i do", "what to do with",
        "we can do with", "i can do with",
        "what analysis", "how to analyze", "what to analyze",
        "what can be done", "what kind of analysis",
        "what insights", "what patterns",
    ]
    if any(k in q_lower for k in kw):
        return True

    # ── 2. Pattern: "what/which [adjectives] model [can/should/would/…]" ─────
    # Catches: "what classification model can we build",
    #          "which regression models should we try",
    #          "what clustering model would work", etc.
    if re.search(
        r"\b(what|which)\b.{0,40}\bmodel\b.{0,30}\b(can|should|would|could|to|for|works?|is|are)\b",
        q_lower,
    ):
        return True

    # ── 3. Modal feasibility pattern ─────────────────────────────────────────
    # "could we train a model", "can i train something", "should we build a classifier"
    if re.search(
        r"\b(can|could|should|would|shall|might)\s+(we|i|you)\s+(train|build|fit|use)\b",
        q_lower,
    ):
        return True

    return False

# ── Code cleaning ─────────────────────────────────────────────────────────────

def _clean_code(raw: str) -> str:
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
    if fenced:
        return "\n".join(fenced).strip()

    lines = raw.splitlines()
    cleaned = []
    code_starters = (
        "import", "from", "print", "df", "fig", "model", "X", "y",
        "for", "if", "try", "#", "plt", "px", "go", "joblib",
        "np", "pd", "sklearn", "scaler", "le ", "encoder", "target",
        "X_train", "X_test", "y_train", "y_test", "results",
    )
    started = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        if not started and stripped and not any(stripped.startswith(s) for s in code_starters):
            continue
        started = True
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _sanitize_imports(code: str) -> str:
    """
    Fix known LLM code mistakes before execution.

    1. SimpleImputer from sklearn.preprocessing → sklearn.impute
    2. px.bar(counts_*, x='index', y=<orig_col>) → x='value', y='count'
       (safe_value_counts returns ['value','count'] columns — LLM often forgets)
    3. px.bar(counts_*, x=<orig_col>, y='count') → x='value', y='count'
       (similar mismatch where LLM uses the original column name instead of 'value')
    """
    fixed_lines = []
    for line in code.splitlines():
        stripped = line.strip()

        # Fix 1: SimpleImputer in wrong module
        if stripped.startswith("from sklearn.preprocessing import") and "SimpleImputer" in stripped:
            new_line = re.sub(r",?\s*SimpleImputer\s*,?", "", line)
            new_line = re.sub(r",\s*,", ",", new_line)
            new_line = re.sub(r"\(\s*,\s*", "(", new_line)
            new_line = re.sub(r",\s*\)", ")", new_line)
            new_line = re.sub(r",\s*$", "", new_line)
            after_import = new_line.split("import", 1)[-1].strip().strip("()")
            if not after_import.replace(",", "").strip():
                continue
            fixed_lines.append(new_line)
            continue

        # Fix 2 & 3: px.bar(counts_*, x=<anything>, y=<anything>) → x='value', y='count'
        # Handles both single-quoted and double-quoted string args from the LLM
        if re.search(r"px\.bar\(counts_\w+", line):
            # Replace x=<any quoted string that isn't already 'value'/"value">
            line = re.sub(r'\bx=(["\'])(?!value\1)(?!value")[^"\']*\1', "x='value'", line)
            # Replace y=<any quoted string that isn't already 'count'/"count">
            line = re.sub(r'\by=(["\'])(?!count\1)(?!count")[^"\']*\1', "y='count'", line)

        fixed_lines.append(line)
    return "\n".join(fixed_lines)


# ── Model name ────────────────────────────────────────────────────────────────

def _model_name(question: str) -> str:
    q = question.lower()
    if "random forest" in q:                     return "random_forest"
    if "logistic" in q:                          return "logistic_regression"
    if "naive bayes" in q or "naive_bayes" in q: return "naive_bayes"
    if "svm" in q or "svc" in q:                 return "svm"
    if "decision tree" in q:                     return "decision_tree"
    if "gradient boost" in q:                    return "gradient_boosting"
    if "kmeans" in q or "k-means" in q:          return "kmeans"
    if "linear regression" in q:                 return "linear_regression"
    return "trained_model"


# ── Code generation ───────────────────────────────────────────────────────────

def _generate_code(
    question: str,
    eda_summary: str,
    history: list,
    wants_chart: bool,
    wants_model: bool,
    df_columns: Optional[list] = None,
    df_sample: Optional[str] = None,
    detected_target: Optional[str] = None,
    target_method: Optional[str] = None,
) -> str:

    history_str = "\n".join(
        f"Q: {h['question']}\nCode:\n{h['code']}\nOutput: {h.get('output','')[:200]}"
        for h in history[-4:]
    ) if history else "None"

    col_list   = df_columns or []
    # Use smart-detected target if available
    target_col = detected_target or (col_list[-1] if col_list else 'last column')
    target_note = target_detection_note(target_col, target_method or 'last_column')

    # Build per-column counts variable names for the prompt
    counts_vars = ""
    if df_columns:
        entries = []
        for col in df_columns:
            safe_var = "counts_" + re.sub(r"[^a-zA-Z0-9_]", "_", col)
            entries.append(f"  {safe_var}  ← value counts for '{col}'")
        counts_vars = "\n".join(entries)

    chart_block = ""
    if wants_chart:
        chart_block = f"""
CHART RULES — MANDATORY, NO EXCEPTIONS:
- Use ONLY plotly.express (px) or plotly.graph_objects (go)
- NEVER call fig.show() or plt.show() — opens GUI window, BREAKS the web app
- At the very END write EXACTLY:
    print('PLOTLY_JSON:' + fig.to_json())
- Apply this layout on every chart using EXACTLY this single call on ONE LINE:
    fig.update_layout(template='plotly_white', height=500, font=dict(family='DM Sans, sans-serif', size=12), title_font_size=15, margin=dict(t=60,r=30,b=60,l=70), colorway=['#2c5f8a','#b8953a','#27ae60','#c0392b','#8e44ad','#16a085'])
    (Keep it on ONE LINE — never split update_layout across multiple lines)
- For charts: just create the figure and end with print('PLOTLY_JSON:' + fig.to_json())
  Do NOT add any extra print statements — stats are handled separately

PRE-COMPUTED VALUE COUNT VARIABLES (already in scope — USE THESE, don't recompute):
{counts_vars}

Each variable is a DataFrame with columns: 'value' and 'count'.
USE THEM DIRECTLY — do NOT call value_counts() or reset_index() yourself.

CHART TYPE RULES:
- Distribution of a discrete/integer column: use the pre-computed variable:
    fig = px.bar(counts_Age, x='value', y='count', title='Distribution of Age')
    (replace 'Age' with actual column, spaces/special chars → underscores in var name)
    The 'value' column is ALREADY str — this gives correct categorical bars
- NEVER use px.histogram for integer/categorical columns — gives EMPTY chart
- ALWAYS use 'df' as the data source for ALL charts — NEVER use corr_matrix or _col_stats
- px.histogram(df, x='ColName') — df is the original dataset, use it for all visualizations
- Continuous column (many unique floats): px.histogram(df, x='col', nbins=30)
- Comparison of TWO numeric columns: px.scatter(df, x='col1', y='col2', title='col1 vs col2')
- Pair plot / scatter matrix (COPY EXACTLY — do NOT split or modify):
    fig = px.scatter_matrix(df.select_dtypes(include='number').iloc[:, :6], title='Pair Plot')
    fig.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(size=3, opacity=0.5))
    fig.update_layout(height=700, margin=dict(t=80, b=80, l=80, r=40))
- Scatter with color grouping: px.scatter(df, x='col1', y='col2', color='category_col')
- ALWAYS cast x-axis categorical values to str before plotting:
    df['col'] = df['col'].astype(str)  before using in px.bar or px.histogram
- Multiple columns comparison: use the PRE-COMPUTED counts_* variables — ALWAYS x='value', y='count'
    CORRECT:   fig = px.bar(counts_MyCol, x='value', y='count', title='...')
    WRONG:     fig = px.bar(counts_MyCol, x='index', y='MyCol')   ← NEVER do this
- Learning curve: sklearn learning_curve → go.Scatter
- Confusion matrix: px.imshow(confusion_matrix_array, text_auto=True)
- Feature importance: px.bar sorted descending
- ROC curve: roc_curve + auc → go.Scatter
- Groupby/aggregation: df.groupby('col').agg() → px.bar directly
- Correlation heatmap (ALWAYS use this exact code):
    corr_matrix = df.select_dtypes(include='number').corr()
    fig = px.imshow(corr_matrix, text_auto='.2f', zmin=-1, zmax=1,
                    color_continuous_scale='RdBu_r', title='Correlation Heatmap',
                    labels=dict(color='Correlation'))
    (NEVER skip zmin/zmax — they are required for proper color scaling)
"""

    model_block = ""
    if wants_model:
        mname = _model_name(question)
        model_block = f"""
MODEL SAVE RULES:
- After training save:
    joblib.dump(model, 'models/{mname}.pkl')
    print('MODEL_SAVED:models/{mname}.pkl')
- Always print accuracy_score and classification_report
- Run cross_val_score(model, X_scaled, y, cv=3, n_jobs=-1) and print mean ± std
  (cv=3 not 5 — faster, still reliable for datasets under 100k rows)
- For RandomForest/GradientBoosting always pass n_jobs=-1 to speed up training
"""

    system = f"""You are an expert Python data scientist writing code for a live web application.
Output ONLY raw executable Python. No prose. No markdown fences. No explanations.
The very first character of your response must be valid Python.

ABSOLUTE RULES — ANY VIOLATION BREAKS THE APP:
1. NO markdown (no ```)
2. NO fig.show() or plt.show() — opens a GUI window and hangs the web server
3. NO hardcoded column names — always reference df.columns
4. NEVER assume a column exists — guard with: if 'col' in df.columns
5. NEVER use sklearn Pipeline or ColumnTransformer
6. For charts: ALWAYS end with print('PLOTLY_JSON:' + fig.to_json())

DATASET FACTS:
Exact columns in df: {col_list}
Target column: '{target_col}'  # {target_note}
First 3 rows:
{df_sample or '(not available)'}

MANDATORY ML PREPROCESSING ORDER:
# 1. Sample if large
if len(df) > 50000:
    df = df.sample(n=50000, random_state=42)
# 2. Set target
target = df.columns[-1]
y = df[target].copy()
X = df.drop(columns=[target], errors='ignore').copy()
# 3. Drop ID-like columns
id_cols = [c for c in X.columns if X[c].nunique() == len(X)]
X = X.drop(columns=id_cols, errors='ignore')
# 4. Encode categoricals
for col in X.select_dtypes(include=['object','category']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
# 5. Fill nulls
X = X.fillna(X.median(numeric_only=True)).fillna(0)
# 6. Scale
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
# 7. Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

ML PARAMS (use exactly these — n_jobs=-1 speeds up training significantly):
- RandomForest: n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
- LogisticRegression: max_iter=1000, random_state=42, n_jobs=-1
- DecisionTree: max_depth=10, random_state=42
- GradientBoosting: n_estimators=100, max_depth=5, random_state=42
- GaussianNB: no params

PRE-AVAILABLE (already imported in scope, do NOT re-import ANY of these):
pd, np, px, go, joblib,
train_test_split, cross_val_score,
StandardScaler, RobustScaler, MinMaxScaler,
LabelEncoder, OneHotEncoder,
SimpleImputer,   ← from sklearn.impute (NOT sklearn.preprocessing — NEVER put it in preprocessing import)
RandomForestClassifier, RandomForestRegressor,
GradientBoostingClassifier, GradientBoostingRegressor,
LogisticRegression, LinearRegression,
SVC, SVR, DecisionTreeClassifier, DecisionTreeRegressor,
KMeans, GaussianNB,
accuracy_score, classification_report,
mean_squared_error, r2_score, confusion_matrix, cross_val_score

IMPORT RULE: Do NOT write any import statements at all — everything is already imported.
If you need SimpleImputer it is ALREADY available. Never write:
  from sklearn.preprocessing import SimpleImputer  ← WRONG, will fail
  from sklearn.impute import SimpleImputer          ← also unnecessary, already imported

{chart_block}
{model_block}

PREVIOUS OPERATIONS:
{history_str}

DATASET EDA:
{eda_summary}
"""

    resp = get_client().chat.completions.create(
        model=get_model("strong"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Task: {question}"},
        ],
        max_tokens=2500,
        temperature=0.05,
    )
    raw = resp.choices[0].message.content.strip()
    return _clean_code(raw)


# ── Auto-fix ──────────────────────────────────────────────────────────────────

def _fix_code(bad_code: str, error: str, eda_summary: str, df_columns: list) -> str:
    resp = get_client().chat.completions.create(
        model=get_model("strong"),
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a Python debugging expert. Fix the broken code. "
                    f"Output ONLY corrected raw Python. No markdown. No explanation.\n\n"
                    f"Rules:\n"
                    f"- NEVER use fig.show() or plt.show()\n"
                    f"- For charts: end with print('PLOTLY_JSON:' + fig.to_json())\n"
                    f"- For column distributions use pre-computed counts_ColName vars (e.g. counts_Age for 'Age' column)\n"
                    f"- Each counts_X var has columns 'value' and 'count' — use: px.bar(counts_Age, x='value', y='count')\n"
                    f"- If counts_ColName not available: safe_value_counts(df['col']) returns same format\n"
                    f"- NEVER call value_counts().reset_index() manually — column names differ by pandas version\n"
                    f"- NEVER use px.histogram for integer/discrete columns — gives empty chart\n"
                    f"- Never use Pipeline or ColumnTransformer\n"
                    f"- Only use columns that exist: {df_columns}\n\n"
                    f"Dataset EDA:\n{eda_summary}"
                ),
            },
            {
                "role": "user",
                "content": f"Broken code:\n{bad_code}\n\nError:\n{error}\n\nFix it.",
            },
        ],
        max_tokens=2500,
        temperature=0.05,
    )
    return _clean_code(resp.choices[0].message.content.strip())


# ── Safe subprocess execution ─────────────────────────────────────────────────

def _execute_code(code: str, df: pd.DataFrame, timeout: int = 60) -> dict:
    """
    Run generated code in an isolated subprocess.
    - Windows-safe: uses forward-slash paths via json embed, not repr()
    - fig.show() / plt.show() silenced (matplotlib Agg backend)
    - Hangs killed after timeout seconds
    - value_counts() fix: normalises column names for both old and new pandas
    """
    import json as _json

    df_tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8")
    df.to_csv(df_tmp.name, index=False)
    df_tmp.close()
    # Use forward slashes — works on Windows and Linux
    df_path_safe = df_tmp.name.replace("\\", "/")

    # ── Build pre-computed column helpers so LLM code never breaks ──────────────
    # Pre-compute value_counts for every column with the correct pandas 2.0 API.
    # This means the LLM can just write: fig = px.bar(counts_Age, x='value', y='count')
    # without needing to call any helper at all.
    col_names   = list(df.columns)
    vc_precompute_lines = [
        "corr_matrix = df.select_dtypes(include='number').corr()",
        "_col_stats = {c: {'mean': round(float(df[c].mean()),2), 'median': round(float(df[c].median()),2), 'std': round(float(df[c].std()),2), 'skew': round(float(df[c].skew()),2)} for c in df.select_dtypes(include='number').columns}",
        "print('STATS_JSON:' + __import__('json').dumps(_col_stats))",
    ]
    import json as _json_col
    for col in col_names:
        safe_var = "counts_" + re.sub(r"[^a-zA-Z0-9_]", "_", col)
        # Use json.dumps for the column name so backslashes/quotes are escaped correctly.
        # Wrap in try/except so an all-null column never kills the entire subprocess.
        col_json = _json_col.dumps(col)   # produces a valid JSON string literal
        vc_precompute_lines.append(
            f"try:\n"
            f"    _s = df[{col_json}].dropna()\n"
            f"    if len(_s) == 0: raise ValueError('all-null')\n"
            f"    _v = _s.value_counts().reset_index()\n"
            f"    _v.columns = ['value', 'count']\n"
            f"    _v['value'] = _v['value'].astype(str)\n"
            f"    {safe_var} = _v[['value','count']]\n"
            f"except Exception:\n"
            f"    {safe_var} = pd.DataFrame({{'value': [], 'count': []}})"
        )
    vc_block = "\n".join(vc_precompute_lines)

    # Also inject a generic safe_value_counts(df['col']) function
    vc_func = (
        "def safe_value_counts(series):\n"
        "    vc = series.value_counts().reset_index()\n"
        "    vc.columns = ['value', 'count']\n"
        "    vc['value'] = vc['value'].astype(str)  # categorical axis\n"
        "    return vc\n"
    )

    script_header = f"""import os, sys, warnings
warnings.filterwarnings('ignore')
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler,
    LabelEncoder, OneHotEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_squared_error, r2_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
)

# ── safe_value_counts helper ──────────────────────────────────────────────────
{vc_func}
# ── Pre-computed value counts for every column ────────────────────────────────
# Use counts_ColumnName directly in px.bar(counts_ColumnName, x='value', y='count')
df = pd.read_csv("{df_path_safe}")
{vc_block}
# ─────────────────────────────────────────────────────────────────────────────
"""

    # Sanitize generated code — fix common LLM mistakes before execution
    import re as _re
    # Fix wrong imports (SimpleImputer in sklearn.preprocessing, etc.)
    code = _sanitize_imports(code)
    # KEY FIX: remove astype(str) on numeric columns before histogram
    # e.g. df["Age"] = df["Age"].astype(str) → empty histogram
    code = _re.sub(r'df\[(["\'])(\w+)\1\]\s*=\s*df\[\1\2\1\]\.astype\(["\']?str["\']?\)', '', code)
    # Replace wrong data sources (only non-counts_* bar charts)
    for _bad in ("corr_matrix", "_col_stats", "col_stats"):
        code = code.replace(f"px.histogram({_bad}", "px.histogram(df")
        code = code.replace(f"px.scatter({_bad}",  "px.scatter(df")
        # Only replace px.bar(bad_source) — NOT px.bar(counts_* which is intentional)
        code = _re.sub(rf'px\.bar\(\s*{re.escape(_bad)}\s*,', 'px.bar(df,', code)
    # Add nbins to histogram calls that don't already have it
    code = _re.sub(r'px\.histogram\(df,(?!\s*nbins)', 'px.histogram(df, nbins=30,', code)

    full_script = script_header + "\n" + code
    # Compute the exact header line count so autofix uses the right offset
    _header_lines = len(script_header.splitlines())

    script_tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8")
    script_tmp.write(full_script)
    script_tmp.close()

    plotly_json = None
    model_path  = None
    error       = None
    output      = ""
    _proc       = None    # track process for kill-on-timeout

    try:
        _proc = subprocess.Popen(
            [sys.executable, script_tmp.name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        try:
            raw_stdout, raw_stderr = _proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _proc.kill()
            _proc.communicate()   # drain pipes so process can exit cleanly
            raise

        lines = []
        plotly_candidates = []
        stats_json = None
        model_results = None
        export_path_from_stdout = None          # parsed from EXPORT_SAVED: sentinel
        for line in raw_stdout.splitlines():
            if line.startswith("PLOTLY_JSON:"):
                candidate = line[len("PLOTLY_JSON:"):]
                stripped = candidate.strip()
                if stripped.startswith("{") or stripped.startswith("["):
                    plotly_candidates.append(candidate)
            elif line.startswith("STATS_JSON:"):
                stats_json = line[len("STATS_JSON:"):]
            elif line.startswith("MODEL_SAVED:"):
                model_path = line[len("MODEL_SAVED:"):].strip()
            elif line.startswith("MODEL_RESULTS:"):
                model_results = line[len("MODEL_RESULTS:"):].strip()
            elif line.startswith("EXPORT_SAVED:"):
                export_path_from_stdout = line[len("EXPORT_SAVED:"):].strip()
            else:
                lines.append(line)
        if plotly_candidates:
            plotly_json = plotly_candidates[-1]
        output = "\n".join(lines).strip()

        # Report error when returncode != 0 AND we got nothing useful
        if _proc.returncode != 0 and not plotly_json and not model_path:
            tb_lines = (raw_stderr or "Unknown error").strip().split("\n")
            error = "\n".join(tb_lines[-10:])
        elif _proc.returncode != 0:
            # Partial success: log but don't fail
            print(f"[EXEC] returncode={_proc.returncode} but got chart/model — stderr: {raw_stderr[:200]}")

    except subprocess.TimeoutExpired:
        error = f"TimeoutError: execution exceeded {timeout}s — try a simpler query or smaller dataset."
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    finally:
        # Ensure process is gone even on unexpected exceptions
        if _proc and _proc.poll() is None:
            try: _proc.kill(); _proc.wait(timeout=2)
            except Exception: pass
        # Clean up temp files — on Windows a killed process may still hold the
        # script file briefly, so we retry once
        for _tmpf in (df_tmp.name, script_tmp.name):
            for _attempt in range(2):
                try: os.unlink(_tmpf); break
                except Exception:
                    import time; time.sleep(0.05)

    return {
        "output":       output,
        "plotly_json":  plotly_json,
        "model_path":   model_path,
        "error":        error,
        "stats_json":   stats_json,
        "model_results": model_results,
        "export_path":  export_path_from_stdout,
        "header_lines": _header_lines,   # passed to surgical_fix for accurate offset
    }


# ── NL answer ─────────────────────────────────────────────────────────────────

def _build_nl_system(has_chart: bool, has_model: bool = False) -> str:
    _name_rule = (
        "\n\nCRITICAL — NAMES AND VALUES:"
        "\n- Always use the EXACT model names, column names, and values from the output."
        "\n- NEVER invent generic placeholders like 'Model A', 'Model B', 'Feature 1'."
        "\n- If the output says 'Random Forest', 'Logistic Regression', 'Gradient Boosting' — use those exact names."
        "\n- If you include a markdown table, copy the real names/values from the output — do NOT fabricate rows."
    )
    if has_model:
        # Model results always take priority — even when a comparison chart also exists.
        # The LLM needs to report real accuracy numbers, not just describe a bar chart.
        base = (
            "You are a data analyst. Machine learning models were just trained and compared. "
            "Summarize the results from the output: mention each model by its exact name, "
            "key metrics (accuracy, R², RMSE etc.), which model performed best and why. "
            "Be concise, 2-4 sentences. Do not mention Python, code, or execution. "
            "Use the EXACT model names from the output — never invent generic names."
        )
    elif has_chart:
        base = (
            "You are a data analyst. The user asked for a chart/visualization. "
            "Based on the question and any printed stats, give a clear interpretation: "
            "describe the relationship, trend, or pattern visible. "
            "Mention specific values or ranges. Be insightful, 2-4 sentences. "
            "Do NOT say 'the chart shows' — just state the insight directly."
        )
    else:
        base = (
            "You are a data analyst. Given raw Python output, write a clear "
            "natural language summary (3-6 sentences). Highlight key findings, "
            "patterns, numbers. Do not mention Python, code, or execution."
        )
    return base + _name_rule + _CONV_TAIL


def _build_user_content(question: str, clean_out: str, has_chart: bool) -> str:
    """
    Build the user message for NL answer generation.

    When a chart was produced but there is no textual output (the code only
    emitted PLOTLY_JSON and nothing else), we must NOT send an empty
    "Output: " section — the LLM will reply with "I don't see any output."
    Instead we tell it a chart was generated and ask for an interpretation
    based on the question, which produces a meaningful response.
    """
    if has_chart and not clean_out.strip():
        return (
            f"Question: {question}\n\n"
            f"A chart was successfully generated. "
            f"Based on the question and what this type of chart typically reveals, "
            f"write 2-3 sentences interpreting the likely pattern, distribution shape, "
            f"or trend — include specific references to what the axis/data represents."
        )
    return f"Question: {question}\n\nOutput:\n{clean_out[:4000]}"


def build_nl_messages(question: str, raw_output: str, has_chart: bool, has_model: bool = False) -> list:
    """Return the messages list for NL answer generation — used by streaming endpoint."""
    # Strip PLOTLY_JSON blob before passing to LLM (it's large and not useful as text)
    clean_out = (raw_output or "").split("PLOTLY_JSON:")[0].strip()
    # Pass up to 4000 chars so model comparison tables (which appear near the end
    # of the output after per-model logs) are never truncated
    return [
        {"role": "system", "content": _build_nl_system(has_chart, has_model=has_model)},
        {"role": "user",   "content": _build_user_content(question, clean_out, has_chart)},
    ]


def _nl_answer(question: str, raw_output: str, has_chart: bool = False, has_model: bool = False) -> str:
    raw_output = (raw_output or "").split("PLOTLY_JSON:")[0].strip()
    system_prompt = _build_nl_system(has_chart, has_model=has_model)
    resp = get_client().chat.completions.create(
        model=get_model("fast"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": _build_user_content(question, raw_output, has_chart)},
        ],
        max_tokens=350, temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


# ── Stat artifact ─────────────────────────────────────────────────────────────

def _build_stat_artifact(df: pd.DataFrame) -> dict:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric = {}
    for col in num_cols[:10]:
        s = df[col].describe()
        numeric[col] = {
            "mean":   round(float(s["mean"]), 3),
            "std":    round(float(s["std"]),  3),
            "min":    round(float(s["min"]),  3),
            "max":    round(float(s["max"]),  3),
            "median": round(float(df[col].median()), 3),
            "nulls":  int(df[col].isnull().sum()),
        }

    categorical = {}
    for col in cat_cols[:6]:
        vc = df[col].value_counts()
        _mode = df[col].mode()
        categorical[col] = {
            "unique":    int(df[col].nunique()),
            "top":       str(_mode.iloc[0]) if not _mode.empty else "",
            "top_count": int(vc.iloc[0]) if not vc.empty else 0,
            "nulls":     int(df[col].isnull().sum()),
        }

    return {
        "type":        "stat_artifact",
        "shape":       {"rows": df.shape[0], "cols": df.shape[1]},
        "numeric":     numeric,
        "categorical": categorical,
        "null_total":  int(df.isnull().sum().sum()),
        "duplicates":  int(df.duplicated().sum()),
    }


# ── Save preferences ──────────────────────────────────────────────────────────

def update_save_preference(csv_id: str, save_charts: bool, save_models: bool, save_history: bool):
    session = _sessions.get(csv_id)
    if session:
        session["prefs"] = {
            "save_charts":  save_charts,
            "save_models":  save_models,
            "save_history": save_history,
        }
        print(f"[CSV] Prefs updated for {csv_id}: charts={save_charts} models={save_models} history={save_history}")



def _build_simple_filter(question: str, columns: list) -> str:
    """
    Rule-based filter code generator for common patterns.
    Handles: col > N, col < N, col == val, col is val, top N by col, missing values.
    Returns Python code string or None if pattern not matched.
    """
    import re
    q = question.lower().strip()
    col_lower = {c.lower(): c for c in columns}

    # Pattern: "where ColName is/= value" or "ColName == value"
    m = re.search(r'where\s+(\w+)\s+(?:is|==|=|equals?)\s+(\S+)', q)
    if m:
        col_key, val = m.group(1), m.group(2).strip("\'\".,").rstrip("?")
        col = col_lower.get(col_key)
        if col:
            # Try numeric first
            try:
                num_val = float(val)
                return (f"filtered_df = df[df['{col}'] == {num_val}]\n"
                        f"print(f'Filtered: {{len(filtered_df)}} rows')\n"
                        f"print(filtered_df.head(20).to_string())")
            except ValueError:
                return (f"filtered_df = df[df['{col}'].astype(str).str.lower() == '{val.lower()}']\n"
                        f"print(f'Filtered: {{len(filtered_df)}} rows')\n"
                        f"print(filtered_df.head(20).to_string())")

    # Pattern: "ColName > N" or "greater than N"
    m = re.search(r'(\w+)\s*(?:>|greater than|more than|above|over)\s*(\d+\.?\d*)', q)
    if m:
        col_key, val = m.group(1), m.group(2)
        col = col_lower.get(col_key)
        if col:
            return (f"filtered_df = df[df['{col}'] > {val}]\n"
                    f"print(f'Filtered: {{len(filtered_df)}} rows')\n"
                    f"print(filtered_df.head(20).to_string())")

    # Pattern: "ColName < N" or "less than N"
    m = re.search(r'(\w+)\s*(?:<|less than|below|under)\s*(\d+\.?\d*)', q)
    if m:
        col_key, val = m.group(1), m.group(2)
        col = col_lower.get(col_key)
        if col:
            return (f"filtered_df = df[df['{col}'] < {val}]\n"
                    f"print(f'Filtered: {{len(filtered_df)}} rows')\n"
                    f"print(filtered_df.head(20).to_string())")

    # Pattern: "top N by ColName"
    m = re.search(r'top\s+(\d+)\s+(?:by|based on|sorted by)?\s*(\w+)', q)
    if m:
        n, col_key = m.group(1), m.group(2)
        col = col_lower.get(col_key)
        if col:
            return (f"filtered_df = df.nlargest({n}, '{col}')\n"
                    f"print(f'Top {n} by {col}: {{len(filtered_df)}} rows')\n"
                    f"print(filtered_df.to_string())")

    # Pattern: "missing values" / "null values"
    if any(kw in q for kw in ["missing", "null", "na values", "empty"]):
        return ("filtered_df = df[df.isnull().any(axis=1)]\n"
                "print(f'Rows with missing values: {len(filtered_df)}')\n"
                "print(filtered_df.to_string())")

    return None  # No simple pattern matched — fall through to LLM


# ── Starter question suggestions ──────────────────────────────────────────────

def generate_csv_suggestions(df: pd.DataFrame, eda_summary: str) -> list:
    """
    Generate 5 diverse starter questions tailored to this specific dataset.

    Covers different analysis types so the user immediately sees the range
    of things they can ask:
      • Overview / EDA
      • Chart / distribution
      • Correlation
      • Filter / slice
      • Model / prediction (if numeric target likely exists)

    Falls back to a safe rule-based set if the LLM call fails.
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Build a compact schema hint so the LLM uses real column names
    schema = ", ".join(
        f"'{c}' ({df[c].dtype})" for c in df.columns[:15]
    )
    if len(df.columns) > 15:
        schema += f" … (+{len(df.columns)-15} more)"

    # Domain hint from column names (helps LLM generate contextual questions)
    col_lower_str = " ".join(df.columns).lower()

    prompt = f"""You are a data analyst. A user just uploaded a CSV dataset.
Generate exactly 5 short, specific questions they can ask about this data.

Dataset schema ({df.shape[0]} rows × {df.shape[1]} cols):
{schema}

Rules:
- Use the EXACT column names from the schema above (quoted or unquoted, but exact)
- Cover these 5 types — one question each:
  1. Overview/EDA    — e.g. "What does this dataset look like?"
  2. Chart           — e.g. "Show me the distribution of <num_col>"
  3. Correlation     — e.g. "Show me the correlation between features"
  4. Filter/slice    — e.g. "Show rows where <col> > <value>" or "top 10 by <col>"
  5. Model           — e.g. "Can we predict <target_col>?" or "Train a model to classify <col>"
- Keep each question under 12 words
- Do NOT number them — output one question per line, nothing else

Questions:"""

    try:
        resp = get_client().chat.completions.create(
            model=get_model("fast"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.4,
        )
        raw = resp.choices[0].message.content.strip()
        # Parse lines — strip bullets / numbers / quotes
        lines = [
            re.sub(r'^[\d\.\-\*\•\"\'\s]+', '', l).strip().strip('"\'')
            for l in raw.splitlines()
            if l.strip() and len(l.strip()) > 5
        ]
        # Deduplicate and cap at 5
        seen = set()
        suggestions = []
        for q in lines:
            lq = q.lower()
            if lq not in seen and len(q) > 5:
                seen.add(lq)
                suggestions.append(q)
            if len(suggestions) == 5:
                break
        if len(suggestions) >= 3:
            return suggestions
    except Exception as e:
        print(f"[SUGGESTIONS] LLM failed: {e}")

    # ── Rule-based fallback ───────────────────────────────────────────────────
    fallback = ["What does this dataset look like?"]
    if num_cols:
        fallback.append(f"Show me the distribution of {num_cols[0]}")
    if len(num_cols) >= 2:
        fallback.append("Show me the correlation heatmap")
    if cat_cols:
        top_val = str(df[cat_cols[0]].value_counts().index[0]) if not df[cat_cols[0]].empty else "value"
        fallback.append(f"Filter rows where {cat_cols[0]} is {top_val}")
    if num_cols:
        fallback.append(f"Can we predict {num_cols[-1]}?")
    return fallback[:5]


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_csv(csv_path: str) -> dict:
    csv_id = _csv_id(csv_path)
    df = pd.read_csv(csv_path)
    eda_summary = _build_eda_summary(df)

    _sessions[csv_id] = {
        "df":              df,
        "csv_path":        csv_path,
        "eda_summary":     eda_summary,
        "history":         [],
        "last_model_path": None,
        "prefs": {
            "save_charts":  True,
            "save_models":  True,
            "save_history": True,
        },
    }
    print(f"[CSV] Ingested {csv_path} → {df.shape}, id={csv_id}")

    # Persist session to SQLite for restart recovery
    save_csv_session(
        csv_id      = csv_id,
        csv_path    = csv_path,
        filename    = Path(csv_path).name,
        eda_summary = eda_summary,
        columns     = list(df.columns),
        dtypes      = {c: str(df[c].dtype) for c in df.columns},
        shape       = df.shape,
        prefs       = {},
    )
    suggestions = generate_csv_suggestions(df, eda_summary)
    return {
        "csv_id":       csv_id,
        "rows":         df.shape[0],
        "cols":         df.shape[1],
        "columns":      list(df.columns),
        "dtypes":       df.dtypes.astype(str).to_dict(),
        "nulls":        {k: int(v) for k, v in df.isnull().sum().items() if v > 0},
        "suggestions":  suggestions,
    }


def _save_and_return(csv_id: str, question: str, result: dict) -> dict:
    """Save turn to SQLite before returning from any path in query_csv."""
    try:
        _sa = result.get("stat_artifact")
        save_csv_turn(
            csv_id        = csv_id,
            question      = question,
            answer        = result.get("answer") or "",
            code          = result.get("code"),
            has_chart     = bool(result.get("has_chart") or result.get("plotly_json")),
            has_model     = bool(result.get("has_model") or result.get("model_path")),
            model_path    = result.get("model_path"),
            plotly_json   = result.get("plotly_json"),
            stat_artifact = json.dumps(_sa) if _sa and not isinstance(_sa, str) else _sa,
        )
    except Exception as e:
        print(f"[CSV_SESSION] save_and_return failed: {e}")
    return result


def query_csv(csv_id: str, question: str, skip_nl_answer: bool = False) -> dict:
    session = _sessions.get(csv_id)
    if not session:
        return {"error": "CSV session not found. Please re-upload the file."}

    df: pd.DataFrame = session["df"]
    eda_summary: str = session["eda_summary"]
    history: list    = session["history"]
    df_columns: list = list(df.columns)
    prefs: dict      = session.get("prefs", {})

    # ── Resolve vague follow-ups before any routing ───────────────────────────
    if _is_vague_followup(question):
        # ── Special case: confirmation after a model recommendation offer ────
        # "sure / yes / ok" after "Want me to train one of these models?"
        # must route straight to model training — NOT through the LLM resolver.
        # The recommendation turn is stored in session flags because it bypasses
        # the normal history append (early-return path).
        if _CONFIRM_RE.match(question.strip()) and session.get("last_was_recommendation"):
            _task  = session.get("last_recommendation_task",  "classification")
            _tcol  = session.get("last_recommendation_target", "")
            _target_hint = f" to predict '{_tcol}'" if _tcol else ""
            question = f"train and compare {_task} models{_target_hint} on this dataset"
            session["last_was_recommendation"] = False
            print(f"[CSV RESOLVER] Confirm after recommendation → '{question}'")

        elif history:
            # Normal vague follow-up — LLM rewrites using last turn context
            last = history[-1]
            question = _resolve_followup(
                question,
                last_q = last.get("question", ""),
                last_a = last.get("output", "") or last.get("answer", ""),
            )

    # Clear recommendation flag when user asks a real new question
    if not _CONFIRM_RE.match(question.strip()):
        session["last_was_recommendation"] = False

    q_lower = question.lower()

    # When skip_nl_answer=True the streaming endpoint handles answer generation.
    # _make_answer is a drop-in that either calls _nl_answer or returns "" + stores
    # the raw output needed for streaming.
    def _make_answer(q: str, raw_out: str, has_chart: bool = False, has_model: bool = False) -> str:
        if skip_nl_answer:
            return ""          # endpoint will stream this live
        return _nl_answer(q, raw_out, has_chart=has_chart, has_model=has_model)

    # ── Shortcut: column listing ──────────────────────────────────────────────
    list_only = any(k in q_lower for k in [
        "list column", "show column", "what column",
        "all column", "column name", "columns?", "list all",
    ])
    explain_req = any(k in q_lower for k in [
        "explain column", "describe column", "tell me about column",
        "what does", "what are the column", "meaning of column",
        "explain the column", "explain each column", "explain all column",
        "what are the columns", "what is each column", "column meaning",
        "column description", "what do the columns",
    ])

    if list_only and not explain_req:
        return _save_and_return(csv_id, question, {
            "answer": (
                f"This dataset has **{df.shape[1]} columns** across **{df.shape[0]:,} rows**:\n\n"
                + "\n".join(f"• `{c}` — {df[c].dtype}" for c in df.columns)
            ),
            "stat_artifact": None, "plotly_json": None, "model_path": None,
            "code": None, "error": None, "has_chart": False, "has_model": False,
        })

    # ── Shortcut: explain columns ─────────────────────────────────────────────
    if explain_req and "column" in q_lower:
        col_info = "\n".join(
            f"- {c} ({df[c].dtype}): samples={df[c].dropna().head(3).tolist()}"
            for c in df.columns
        )
        resp = get_client().chat.completions.create(
            model=get_model("fast"),
            messages=[
                {"role": "system", "content": (
                    "You are a data analyst. Explain each column in plain English. "
                    "Numbered list. One sentence per column. Be specific."
                ) + _CONV_TAIL},
                {"role": "user", "content": f"Columns:\n{col_info}\n\nEDA:\n{eda_summary}\n\nQuestion: {question}"},
            ],
            max_tokens=900, temperature=0.3,
        )
        return _save_and_return(csv_id, question, {
            "answer": resp.choices[0].message.content.strip(),
            "stat_artifact": None, "plotly_json": None, "model_path": None,
            "code": None, "error": None, "has_chart": False, "has_model": False,
        })

    wants_chart    = _wants_chart(question)
    # Correlation heatmap → use stat artifact (more reliable than Plotly heatmap)
    is_corr_heatmap = any(k in question.lower() for k in [
        "correlation heatmap", "heatmap", "correlation matrix",
        "feature correlation", "corr matrix", "corr heatmap",
        # Natural phrasings
        "give me correlation", "show correlation", "show me correlation",
        "show the correlation",          # "Show the correlation between X and Y"
        "correlation of the", "correlation of this", "correlation of our",
        "correlations in", "correlations of", "get correlation",
        "correlation between",           # "correlation between Glucose and Insulin"
        "between .* correlation",        # reverse phrasing
        "correlation analysis", "correlation plot",
        "plot correlation", "visualize correlation", "visualise correlation",
        "what is the correlation", "find the correlation",
        "check the correlation", "calculate the correlation",
        "compute the correlation",
    ])
    if is_corr_heatmap:
        wants_chart = False  # route to artifact, not code gen
    wants_model    = _wants_model(question)
    is_chart_followup      = _is_chart_followup(question)
    is_feature_importance  = _wants_feature_importance(question)
    if is_chart_followup:
        wants_chart = True
        wants_model = False
    is_analysis    = _wants_analysis(question)
    is_filter      = is_filter_query(question) and not _wants_chart(question)
    is_export      = is_export_query(question) and not wants_model and not wants_chart
    is_recommend   = _wants_recommendation(question)

    # Explicit training intent overrides recommendation detection.
    # "train different models and give me the best model" should TRAIN, not recommend.
    #
    # IMPORTANT: "can we train", "could we train", "what type of model can we train"
    # are FEASIBILITY / RECOMMENDATION questions — they must NOT trigger actual training.
    # Guard: only treat "train" as explicit if it is NOT preceded by a modal verb.
    _FEASIBILITY_RE = re.compile(
        r"\b(can|could|should|would|shall|might)\s+(we|i|you)\s+train\b"
        r"|\bwe\s+(can|could|should|would)\s+train\b"
        r"|\bwhat\s+(type|kind|sort)\s+of\b.{0,60}\btrain\b",
        re.IGNORECASE,
    )
    _is_feasibility = bool(_FEASIBILITY_RE.search(question))

    _explicit_train = (
        not _is_feasibility and          # feasibility phrasing → always recommend
        any(k in q_lower for k in [
            "train a", "train the", "train different", "train all",
            "fit a", "fit the", "build and compare", "build the model", "build all",
            "run a model", "evaluate model", "compare model", "compare different",
            "which one performs best", "test different", "try different",
            "let's train", "start training", "go ahead and train",
        ])
    )
    if is_recommend and _explicit_train:
        is_recommend = False   # training intent wins — go to code gen
    if is_recommend:
        wants_model = False

    print(f"[CSV QUERY] '{question}' | chart={wants_chart} model={wants_model} analysis={is_analysis} filter={is_filter} export={is_export}")

    # ── Shortcut: feature importance — load saved model and plot ─────────────
    # Only fire this shortcut when the user ONLY wants importance (not also training).
    # "build a model AND give feature importance" must fall through to code generation
    # so both steps happen together in one run.
    if is_feature_importance and not wants_model:
        model_path = session.get("last_model_path")
        if not model_path or not os.path.exists(model_path):
            return _save_and_return(csv_id, question, {
                "answer": (
                    "No trained model found in this session. "
                    "Train a model first (e.g. 'build a random forest classifier'), "
                    "then ask for feature importance."
                ),
                "stat_artifact": None, "plotly_json": None, "model_path": None,
                "code": None, "error": None, "has_chart": False, "has_model": False,
            })
        model_cls = extract_model_class(model_path, model_path)
        if "random_forest" in model_path:  model_cls = "RandomForestClassifier"
        elif "logistic" in model_path:     model_cls = "LogisticRegression"
        elif "gradient" in model_path:     model_cls = "GradientBoostingClassifier"
        elif "linear" in model_path:       model_cls = "LinearRegression"

        fi_code = f"""
import joblib, pandas as pd, numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

model = joblib.load(r'{model_path}')
target = '{df_columns[-1]}'
y = df[target].copy()
X = df.drop(columns=[target], errors='ignore').copy()
for _c in X.select_dtypes(include=['object','category']).columns:
    X[_c] = LabelEncoder().fit_transform(X[_c].astype(str))
X = X.fillna(X.median(numeric_only=True)).fillna(0)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X = X_scaled
""" + get_universal_feature_importance_code("model", df_columns[-1])

        result = _execute_code(fi_code, df)
        answer = _make_answer(question, result.get("output",""), has_chart=bool(result.get("plotly_json")))
        return _save_and_return(csv_id, question, {
            "answer":        answer or "Here is the feature importance chart.",
            "stat_artifact": None,
            "plotly_json":   result.get("plotly_json"),
            "model_path":    model_path,
            "code":          fi_code,
            "error":         result.get("error"),
            "has_chart":     bool(result.get("plotly_json")),
            "has_model":     False,
        })

    # ── Shortcut: model comparison chart — use saved results ─────────────────
    # Detection is intentionally broad to catch typos ("comparision"),
    # non-adjacent word pairs ("model ... chart"), and natural phrasings.
    _q_lower_comp = question.lower()
    is_comparison_chart = wants_chart and (
        # Exact keyword list (handles most cases)
        any(k in _q_lower_comp for k in [
            "comparison", "comparision",        # typo tolerance
            "compare model", "model chart",
            "accuracy chart", "model performance", "compare accuracy",
            "compare the model", "comparison of model", "different model",
            "model results", "show models", "visualize model",
            "model vs", "vs model",
        ])
        # Regex: "model" anywhere + "chart/compare/comparison" anywhere in same query
        or bool(re.search(r'\bmodel\b', _q_lower_comp)) and bool(
            re.search(r'\b(chart|compare|comparison|comparision|performance|accuracy|result)\b',
                      _q_lower_comp)
        )
    )
    if is_comparison_chart and session.get("last_model_results"):
        import json as _json
        _raw_results = session["last_model_results"]
        # Normalise to a Python list regardless of whether it was stored as JSON string or list
        if isinstance(_raw_results, str):
            try:
                results_list = _json.loads(_raw_results)
            except Exception:
                results_list = []
        else:
            results_list = _raw_results
        # Guard: reject generic LLM-invented names like "Model A", "Model 1", "model_0"
        # These appear when the comparison code was generated by LLM instead of build_comparison_code
        _GENERIC_NAME_RE = re.compile(r"^model[\s_\-]?[a-z0-9]$", re.IGNORECASE)
        if results_list and any(_GENERIC_NAME_RE.match(str(r.get("model", ""))) for r in results_list):
            print("[CSV] last_model_results has generic model names — clearing for retraining")
            results_list = []
            session["last_model_results"] = None
        if not results_list:
            pass   # fall through to retraining path below
        else:
            # Embed as a safe JSON literal — avoids any quote/escape issues
            results_json_literal = _json.dumps(results_list)
            sample = results_list[0]
            y_col = "test_accuracy" if "test_accuracy" in sample else "r2" if "r2" in sample else list(sample.keys())[1]
            has_cv = "cv_accuracy" in sample or "cv_r2" in sample
            cv_col = "cv_accuracy" if "cv_accuracy" in sample else ("cv_r2" if "cv_r2" in sample else None)
            if has_cv and cv_col:
                comp_code = f"""
import pandas as pd, json as _j, plotly.express as px
results_df = pd.DataFrame(_j.loads({repr(results_json_literal)}))
fig = px.bar(
    results_df.melt(id_vars='model', value_vars=['{y_col}', '{cv_col}'],
                    var_name='Metric', value_name='Score'),
    x='model', y='Score', color='Metric', barmode='group',
    title='Model Comparison — Test vs CV Accuracy', text='Score',
    color_discrete_map={{'{y_col}':'#2c5f8a', '{cv_col}':'#b8953a'}},
)
fig.update_traces(texttemplate='%{{text:.3f}}', textposition='outside')
fig.update_layout(template='plotly_white', height=480,
    font=dict(family='DM Sans, sans-serif', size=12),
    margin=dict(t=70, r=30, b=60, l=70),
    yaxis=dict(range=[0, 1.05]),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
print('PLOTLY_JSON:' + fig.to_json())
"""
            else:
                comp_code = f"""
import pandas as pd, json as _j, plotly.express as px
results_df = pd.DataFrame(_j.loads({repr(results_json_literal)}))
fig = px.bar(results_df, x='model', y='{y_col}',
             title='Model Comparison', text='{y_col}',
             color='model',
             color_discrete_sequence=['#2c5f8a','#b8953a','#27ae60'])
fig.update_traces(texttemplate='%{{text:.3f}}', textposition='outside')
fig.update_layout(template='plotly_white', height=480,
    font=dict(family='DM Sans, sans-serif', size=12),
    margin=dict(t=60,r=30,b=60,l=70),
    yaxis=dict(range=[0, 1.05]), showlegend=False)
print('PLOTLY_JSON:' + fig.to_json())
"""
            result = _execute_code(comp_code, df)
            _comp_raw = f"Model comparison results:\n{_json.dumps(results_list)}"
            answer = _make_answer(question, _comp_raw, has_chart=bool(result.get("plotly_json")), has_model=True)
            return _save_and_return(csv_id, question, {
                "answer":        answer or "Here is the model comparison chart.",
                "stat_artifact": None,
                "plotly_json":   result.get("plotly_json"),
                "model_path":    session.get("last_model_path"),
                "code":          comp_code,
                "error":         result.get("error"),
                "has_chart":     bool(result.get("plotly_json")),
                "has_model":     True,
                "raw_output":    _comp_raw,
            })

    # ── Fallback: comparison chart requested but no stored results → re-train, chart only ──
    if is_comparison_chart and not session.get("last_model_results"):
        _t_col, _t_method = detect_target(question, df)
        _task = detect_task_type(question, df, _t_col)
        _cands = get_model_candidates(_task)
        # Build comparison code WITHOUT feature importance — user only wants the chart
        _rerun_code = build_comparison_code(_task, _cands, _t_col, df_columns)
        _result = _execute_code(_rerun_code, df, timeout=180)
        if _result.get("model_results"):
            session["last_model_results"] = _result["model_results"]
        if _result.get("model_path"):
            session["last_model_path"] = _result["model_path"]
        _rerun_raw = _result.get("output") or ""
        if _result.get("model_results"):
            import json as _json2
            _rerun_raw = f"Model comparison results:\n{_json2.dumps(_result['model_results'])}\n{_rerun_raw}".strip()
        _rerun_has_model = bool(_result.get("model_path")) or bool(_result.get("model_results"))
        _answer = _make_answer(question, _rerun_raw, has_chart=bool(_result.get("plotly_json")), has_model=_rerun_has_model)
        return _save_and_return(csv_id, question, {
            "answer":        _answer or "Here is the model comparison.",
            "stat_artifact": None,
            "plotly_json":   _result.get("plotly_json"),
            "model_path":    _result.get("model_path"),
            "code":          _rerun_code,
            "error":         _result.get("error"),
            "has_chart":     bool(_result.get("plotly_json")),
            "has_model":     _rerun_has_model,
            "raw_output":    _rerun_raw or None,
        })

    # ── Shortcut: export dataset ──────────────────────────────────────────────
    if is_export:
        # Always write a fresh timestamped file — no stale cache
        export_path = register_for_export(csv_id, df, "original")
        return _save_and_return(csv_id, question, {
            "answer": f"Your dataset is ready to download ({df.shape[0]:,} rows × {df.shape[1]} columns).",
            "stat_artifact": None, "plotly_json": None, "model_path": None,
            "code": None, "error": None, "has_chart": False, "has_model": False,
            "export_path": export_path, "has_export": True,
        })

    # ── Shortcut: correlation heatmap → stat artifact ─────────────────────────
    if is_corr_heatmap and not wants_model:
        artifact = _build_stat_artifact(df)
        t_col, _ = detect_target(question, df)
        corr_art  = build_correlation_artifact(df, t_col)
        artifact["correlation"] = corr_art
        # Use correlation_summary() for a rich natural-language context to the LLM
        corr_text = correlation_summary(corr_art)
        top_pairs = corr_art.get("top_pairs", [])
        top3_str  = "; ".join(
            f"'{p['col1']}' & '{p['col2']}' (r={p['r']}, {p['strength']})"
            for p in top_pairs[:3]
        ) if top_pairs else "no significant correlations found"
        resp = get_client().chat.completions.create(
            model=get_model("fast"),
            messages=[
                {"role": "system", "content": (
                    "You are a data analyst. Given the correlation data below, write 2-3 clear sentences "
                    "explaining the strongest relationships and what they suggest about the data. "
                    "Be specific — name the actual column pairs and r-values. "
                    "Do NOT say the data is incomplete or NaN."
                ) + _CONV_TAIL},
                {"role": "user", "content": (
                    f"Question: {question}\n\n"
                    f"Top correlations: {top3_str}\n"
                    f"Full summary: {corr_text}"
                )},
            ],
            max_tokens=250, temperature=0.3,
        )
        return _save_and_return(csv_id, question, {
            "answer":        resp.choices[0].message.content.strip(),
            "stat_artifact": artifact,
            "plotly_json":   None, "model_path": None,
            "code":          None, "error": None,
            "has_chart":     False, "has_model": False,
        })

    # ── Shortcut: model recommendation ───────────────────────────────────────
    if is_recommend:
        t_col, t_method = detect_target(question, df)
        task_type_rec   = detect_task_type(question, df, t_col)
        corr_art        = build_correlation_artifact(df, t_col)
        top_preds       = ", ".join(
            f"{c['col']} (r={c['r']})"
            for c in corr_art.get("target_corr", [])[:4]
        ) or "unknown"
        imb_sev, imb_ratio, _ = detect_imbalance(df, t_col) if t_col else ("balanced", 0.5, {})
        rec_prompt = (
            f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns. "
            f"Task: {task_type_rec}. Target: '{t_col}'. "
            f"Top predictors: {top_preds}. "
            f"Class balance: {imb_sev} ({imb_ratio:.1%} minority). "
            f"User question: {question}"
        )
        resp = get_client().chat.completions.create(
            model=get_model("fast"),
            messages=[
                {"role": "system", "content": (
                    "You are a data science advisor. The user wants to know what ML model(s) "
                    "are suitable for their dataset — they are NOT asking to train yet. "
                    "Based on the dataset info provided:\n"
                    "1. State the task type (classification / regression / clustering) and why.\n"
                    "2. Name 2-3 recommended algorithms with a one-line reason each.\n"
                    "3. Mention any data considerations (size, class balance, key predictors).\n"
                    "4. End with a short invite: 'Want me to train one of these models?'\n"
                    "Be concise and direct. Do NOT train or write any code."
                )},
                {"role": "user", "content": rec_prompt},
            ],
            max_tokens=350, temperature=0.3,
        )
        rec_answer = resp.choices[0].message.content.strip()

        # Remember that we just made a recommendation so a follow-up
        # "sure / yes / ok" can short-circuit directly to model training
        # without going through the LLM resolver (which has no history of
        # this turn and would generate a wrong rewrite).
        session["last_was_recommendation"] = True
        session["last_recommendation_task"] = task_type_rec
        session["last_recommendation_target"] = t_col or ""

        # Also push to in-memory history so the LLM resolver has context
        # if it IS called (e.g. for elaborate follow-ups beyond a simple "sure")
        history.append({
            "question": question,
            "code":     None,
            "output":   "",
            "answer":   (rec_answer or "")[:400],
        })
        session["history"] = history[-10:]

        return _save_and_return(csv_id, question, {
            "answer": rec_answer,
            "stat_artifact": None, "plotly_json": None, "model_path": None,
            "code": None, "error": None, "has_chart": False, "has_model": False,
        })

    # ── Shortcut: stat overview ───────────────────────────────────────────────
    if is_analysis and not wants_chart and not wants_model and not explain_req:
        artifact = _build_stat_artifact(df)
        t_col, _ = detect_target(question, df)
        corr_art = build_correlation_artifact(df, t_col)
        artifact["correlation"] = corr_art
        resp = get_client().chat.completions.create(
            model=get_model("fast"),
            messages=[
                {"role": "system", "content": (
                    "You are a data analyst. Summarize this dataset in 3-5 sentences. "
                    "Highlight the most interesting patterns, data quality, and key stats."
                ) + _CONV_TAIL},
                {"role": "user", "content": f"Question: {question}\n\nInfo:\n{eda_summary}\n\nCorrelation: {correlation_summary(corr_art)}"},
            ],
            max_tokens=350, temperature=0.3,
        )
        return _save_and_return(csv_id, question, {
            "answer":        resp.choices[0].message.content.strip(),
            "stat_artifact": artifact,
            "plotly_json":   None, "model_path": None,
            "code":          None, "error": None,
            "has_chart":     False, "has_model": False,
        })

    # ── Enrich follow-up chart questions with previous column context ─────────
    if is_chart_followup and history:
        last_q = history[-1].get("question", "")
        import re as _re
        prev_cols_found = [c for c in df_columns if c.lower() in last_q.lower()]
        if prev_cols_found:
            question = question + f" (use columns: {', '.join(prev_cols_found)} from the previous chart)"
            print(f"[CSV] Chart followup enriched: {question[:80]}")

    # ── Sample for LLM context ────────────────────────────────────────────────
    try:
        df_sample = df.head(3).to_string(index=False, max_cols=12)
    except Exception:
        df_sample = None

    # ── Smart target detection ────────────────────────────────────────────────
    detected_target = None
    target_method   = None
    task_type       = None
    imb_note        = ""
    if wants_model:
        detected_target, target_method = detect_target(question, df)
        print(f"[TARGET] Detected: '{detected_target}' via '{target_method}'")
        task_type = detect_task_type(question, df, detected_target)
        print(f"[TASK] Type: {task_type}")
        if task_type == "classification" and detected_target:
            sev, ratio, counts = detect_imbalance(df, detected_target)
            imb_note = imbalance_note(sev, ratio, counts)

    # ── NL filter shortcut ────────────────────────────────────────────────────
    if is_filter and not wants_model and not wants_chart:
        simple_code = _build_simple_filter(question, df_columns)
        filter_code = simple_code or generate_filter_code(question, df, eda_summary)

        if filter_code:
            # Use get_export_code() for timestamped filenames — prevents stale cache
            save_code = get_export_code(csv_id, label="filtered")
            full_filter_code = filter_code + save_code
            result = _execute_code(full_filter_code, df)
            print(f"[FILTER] error={result.get('error','none')[:200] if result.get('error') else 'none'}")

            if not result.get("error"):
                export_path = result.get("export_path")
                if export_path:
                    session["last_export_path"] = export_path
                answer = _make_answer(question, result.get("output",""), has_chart=bool(result.get("plotly_json")))
                return _save_and_return(csv_id, question, {
                    "answer": answer or "Here are the filtered results.",
                    "stat_artifact": None, "plotly_json": None,
                    "model_path": None, "code": filter_code,
                    "raw_output": result.get("output"),
                    "error": None, "has_chart": False, "has_model": False,
                    "export_path": export_path,
                    "has_export": bool(export_path),
                    "is_filter": True,
                })
            else:
                fixed = _fix_code(filter_code, result["error"], eda_summary, df_columns)
                result2 = _execute_code(fixed + save_code, df)
                if not result2.get("error"):
                    export_path = result2.get("export_path")
                    if export_path:
                        session["last_export_path"] = export_path
                    answer = _make_answer(question, result2.get("output",""), has_chart=bool(result2.get("plotly_json")))
                    return _save_and_return(csv_id, question, {
                        "answer": answer or "Here are the filtered results.",
                        "stat_artifact": None, "plotly_json": None,
                        "model_path": None, "code": fixed,
                        "raw_output": result2.get("output"),
                        "error": None, "has_chart": False, "has_model": False,
                        "export_path": export_path, "has_export": bool(export_path),
                        "is_filter": True,
                    })

    # ── Check if auto model comparison ────────────────────────────────────────
    # True when user doesn't name a specific model → train & compare all candidates
    # NOT True for "give me comparison chart" (that's a visualization of existing results)
    use_model_comparison = (
        wants_model and task_type and
        not is_comparison_chart and          # chart of results ≠ re-training
        not any(kw in question.lower() for kw in [
            "random forest","logistic","gradient boost","decision tree",
            "svm","naive bayes","linear regression","kmeans","xgboost",
        ])
    )

    # ── Generate → subprocess exec → auto-fix loop ────────────────────────────
    MAX_RETRIES = 3
    if use_model_comparison and detected_target:
        candidates = get_model_candidates(task_type)
        code = build_comparison_code(task_type, candidates, detected_target, df_columns)
        # Append FI only when user asked for it explicitly — not on every comparison run
        if is_feature_importance:
            code = code + "\n" + get_universal_feature_importance_code("best_model", detected_target)
    else:
        code = _generate_code(
            question, eda_summary, history,
            wants_chart, wants_model, df_columns, df_sample,
            detected_target=detected_target,
            target_method=target_method,
        )
        if wants_model and not wants_chart:
            model_cls = extract_model_class(question, code)
            # Alias 'model' to whatever variable the LLM used, then use universal FI
            fi_code = (
                "# Ensure 'model' variable exists for feature importance\n"
                "try:\n"
                "    _fi_candidate = model\n"
                "except NameError:\n"
                "    _fi_candidate = best_model if 'best_model' in dir() else None\n"
            )
            fi_code += get_universal_feature_importance_code("_fi_candidate", detected_target or "target")
            code = code + "\n" + fi_code
    print(f"[CSV] Generated code ({len(code)} chars)")

    # ── Inject model_results if session has stored results ────────────────────
    # When the LLM generates comparison chart code using model_results as a
    # variable (because it saw training results in conversation history), the
    # subprocess would raise NameError.  Prepend the real data so it's available.
    _stored_results = session.get("last_model_results")
    if _stored_results and "model_results" in code:
        import json as _mr_json
        _mr_str = _mr_json.dumps(
            _stored_results if isinstance(_stored_results, list)
            else _mr_json.loads(_stored_results)
        )
        _inject = (
            f"import json as _mr_json\n"
            f"model_results = _mr_json.loads({repr(_mr_str)})\n"
        )
        code = _inject + code
        print(f"[CSV] Injected model_results ({len(_stored_results)} entries) into code")

    result     = None
    last_error = None

    # Model training (especially comparison) needs more time than a chart.
    # 60s → fine for charts / analysis
    # 180s → model comparison (3 models × 3-fold CV + feature importance)
    exec_timeout = 180 if wants_model else 60

    for attempt in range(MAX_RETRIES):
        print(f"[CSV] Attempt {attempt + 1}/{MAX_RETRIES} (timeout={exec_timeout}s)")
        result = _execute_code(code, df, timeout=exec_timeout)
        print(f"[CSV CODE] {code[:400].replace(chr(10), chr(124))}")

        if not result["error"]:
            print(f"[CSV] ✓ Success on attempt {attempt + 1}")
            break

        last_error = result["error"]
        print(f"[AUTO-FIX] Attempt {attempt + 1} failed:\n{last_error[:300]}")

        if attempt < MAX_RETRIES - 1:
            print("[AUTO-FIX] Calling surgical_fix…")
            code = surgical_fix(code, last_error, df_columns, eda_summary,
                               header_lines=result.get("header_lines", 0))
            print(f"[AUTO-FIX] Fixed code ({len(code)} chars)")

    # ── NL answer ─────────────────────────────────────────────────────────────
    answer  = ""
    raw_out = ""   # hoisted so the return dict can reference the enriched version
    if result:
        raw_out = result.get("output") or ""
        err     = result.get("error")  or ""

        _has_chart = bool(result.get("plotly_json"))
        _stats = result.get("stats_json")
        if _stats:
            raw_out = f"Column stats: {_stats}\n{raw_out}".strip()
        if raw_out or _has_chart:
            answer = _make_answer(question, raw_out, has_chart=_has_chart)
        elif not result.get("plotly_json") and not result.get("model_path"):
            answer = (
                "I wasn't able to compute a result for that question. "
                "Try rephrasing, or ask something more specific about the data."
            )
            if err:
                print(f"[CSV] All retries exhausted. Last error:\n{err[:400]}")

        if result.get("model_path"):
            session["last_model_path"] = result["model_path"]

        # Store model comparison results for follow-up chart queries
        if result.get("model_results"):
            session["last_model_results"] = result["model_results"]

        # Track export path for follow-up session access
        if result.get("export_path"):
            session["last_export_path"] = result["export_path"]

    # ── History ───────────────────────────────────────────────────────────────
    if prefs.get("save_history", True):
        history.append({
            "question": question,
            "code":     code,
            "output":   (result.get("output") or "")[:300],
            "answer":   (answer or "")[:400],   # kept for follow-up resolver
        })
        session["history"] = history[-10:]

    # Persist this turn to SQLite
    _sa = result.get("stat_artifact") if result else None
    save_csv_turn(
        csv_id        = csv_id,
        question      = question,
        answer        = answer or "",
        code          = code,
        has_chart     = bool(result and result.get("plotly_json")),
        has_model     = bool(result and result.get("model_path")),
        model_path    = result.get("model_path") if result else None,
        plotly_json   = result.get("plotly_json") if result else None,
        stat_artifact = json.dumps(_sa) if _sa else None,
    )

    _exec_export = result.get("export_path") if result else None
    return {
        "answer":        answer,
        "stat_artifact": None,
        "plotly_json":   result.get("plotly_json") if result else None,
        "model_path":    result.get("model_path")  if result else None,
        "code":          code if prefs.get("save_history", True) else None,
        # Use the stats-enriched raw_out (not bare result["output"]) so the
        # streaming endpoint LLM has column stats to interpret even when the
        # code only printed PLOTLY_JSON and nothing else.
        "raw_output":    raw_out or None,
        "error":         None,
        "has_chart":     bool(result and result.get("plotly_json")),
        "has_model":     bool(result and result.get("model_path")),
        "export_path":   _exec_export,
        "has_export":    bool(_exec_export),
    }


def get_model_path(csv_id: str) -> Optional[str]:
    s = _sessions.get(csv_id)
    return s.get("last_model_path") if s else None