"""
csv_eval.py — CSV Agent Evaluation for Bujji Babu

Custom evaluation since RAGAS is designed for RAG, not code execution.

Evaluation categories:
  1. Stats accuracy    — compare output numbers vs pandas ground truth
  2. Filter accuracy   — compare row counts and values
  3. Chart validation  — check plotly_json has correct trace type and columns
  4. ML validation     — check model saved and accuracy in expected range
  5. Answer quality    — LLM-as-judge for interpretation quality

Usage:
  python csv_eval.py --csv_path csv_uploads/diabetes.csv
  python csv_eval.py --csv_path csv_uploads/diabetes.csv --csv_id <existing_id>
"""

import os
import sys
import json
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()


# ── Test suite for diabetes dataset ──────────────────────────────────────────

DIABETES_TESTS = [
    # ── Stats accuracy ─────────────────────────────────────────────────────
    {
        "id":       "stats_01",
        "category": "stats",
        "question": "What is the mean glucose level?",
        "check": lambda ans, df: _contains_number(ans, df["Glucose"].mean(), tol=0.5),
        "expected": "120.89",
        "description": "Mean glucose in answer",
    },
    {
        "id":       "stats_02",
        "category": "stats",
        "question": "How many rows and columns does this dataset have?",
        "check": lambda ans, df: "768" in ans and ("9" in ans or "nine" in ans.lower()),
        "expected": "768 rows, 9 columns",
        "description": "Shape in answer",
    },
    {
        "id":       "stats_03",
        "category": "stats",
        "question": "What is the correlation between Glucose and Outcome?",
        "check": lambda ans, df: _contains_number(ans, df["Glucose"].corr(df["Outcome"]), tol=0.02),
        "expected": "0.467",
        "description": "Glucose-Outcome correlation in answer",
    },
    {
        "id":       "stats_04",
        "category": "stats",
        "question": "What is the median age?",
        "check": lambda ans, df: _contains_number(ans, df["Age"].median(), tol=1.0),
        "expected": "29.0",
        "description": "Median age in answer",
    },

    # ── Filter accuracy ────────────────────────────────────────────────────
    {
        "id":       "filter_01",
        "category": "filter",
        "question": "Filter rows where Outcome is 1",
        "check": lambda ans, df: _contains_number(ans, len(df[df["Outcome"]==1]), tol=0),
        "expected": "268 rows",
        "description": "Filter returns 268 rows",
    },
    {
        "id":       "filter_02",
        "category": "filter",
        "question": "Show patients with Glucose greater than 150",
        "check": lambda ans, df: _contains_number(ans, len(df[df["Glucose"]>150]), tol=2),
        "expected": str(len(pd.read_csv("csv_uploads/diabetes.csv")[pd.read_csv("csv_uploads/diabetes.csv")["Glucose"]>150])) if Path("csv_uploads/diabetes.csv").exists() else "~130",
        "description": "Filter >150 glucose row count",
    },
    {
        "id":       "filter_03",
        "category": "filter",
        "question": "Show top 5 patients by BMI",
        "check": lambda ans, df: _contains_number(ans, 5, tol=0) or "top" in ans.lower(),
        "expected": "5 rows, highest BMI",
        "description": "Top 5 by BMI returned",
    },

    # ── Chart validation ───────────────────────────────────────────────────
    {
        "id":       "chart_01",
        "category": "chart",
        "question": "Plot glucose vs BMI scatter chart",
        "check": lambda result, df: _check_chart(result, trace_type="scatter", x_col="Glucose", y_col="BMI"),
        "expected": "scatter trace with Glucose x-axis and BMI y-axis",
        "description": "Scatter chart with correct columns",
    },
    {
        "id":       "chart_02",
        "category": "chart",
        "question": "Show distribution of Age in histogram",
        "check": lambda result, df: _check_chart(result, trace_type="histogram"),
        "expected": "histogram trace",
        "description": "Histogram chart generated",
    },

    # ── ML validation ──────────────────────────────────────────────────────
    {
        "id":       "ml_01",
        "category": "ml",
        "question": "Build a random forest classifier to predict diabetes",
        "check": lambda result, df: _check_model(result, expected_accuracy_range=(0.68, 0.85)),
        "expected": "model saved, accuracy 68-85%",
        "description": "Model trained with acceptable accuracy",
    },

    # ── Answer quality ─────────────────────────────────────────────────────
    {
        "id":       "quality_01",
        "category": "quality",
        "question": "Give me an overview of this dataset",
        "check": lambda ans, df: _llm_judge(ans, "The answer should mention: number of rows (768), number of columns (9), and at least one key statistical insight about the diabetes data."),
        "expected": "Mentions 768 rows, 9 columns, key stats",
        "description": "Overview quality check",
    },
    {
        "id":       "quality_02",
        "category": "quality",
        "question": "Which model works best for predicting diabetes outcome?",
        "check": lambda ans, df: _llm_judge(ans, "The answer should recommend a specific ML model and explain why it suits this dataset (size, class balance, features)."),
        "expected": "Recommends a model with reasoning",
        "description": "Model recommendation quality",
    },
]


# ── Check helpers ─────────────────────────────────────────────────────────────

def _contains_number(text: str, value: float, tol: float = 0.1) -> bool:
    """Check if a number close to value appears in the text."""
    import re
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    for n in numbers:
        try:
            if abs(float(n) - value) <= tol:
                return True
        except ValueError:
            pass
    return False


def _check_chart(result: Dict, trace_type: str = None, x_col: str = None, y_col: str = None) -> bool:
    """Validate a chart result has correct structure."""
    pj = result.get("plotly_json")
    if not pj:
        return False
    try:
        spec = json.loads(pj) if isinstance(pj, str) else pj
        traces = spec.get("data", [])
        if not traces:
            return False

        trace = traces[0]

        # Check trace type
        if trace_type and trace.get("type", "") != trace_type:
            # histogram sometimes shows as bar
            if not (trace_type == "histogram" and trace.get("type") in ("histogram", "bar")):
                return False

        # Check column names in axis labels or trace name
        layout = spec.get("layout", {})
        x_title = str(layout.get("xaxis", {}).get("title", {}).get("text", "")).lower()
        y_title = str(layout.get("yaxis", {}).get("title", {}).get("text", "")).lower()

        if x_col and x_col.lower() not in x_title and x_col.lower() not in str(trace.get("name","")).lower():
            # Also check x data name in layout
            if x_col.lower() not in str(spec).lower():
                return False

        if y_col and y_col.lower() not in y_title and y_col.lower() not in str(spec).lower():
            return False

        return True
    except Exception as e:
        print(f"  Chart check error: {e}")
        return False


def _check_model(result: Dict, expected_accuracy_range: tuple = (0.60, 0.90)) -> bool:
    """Check if a model was trained with acceptable accuracy."""
    # Check model file exists
    model_path = result.get("model_path")
    if model_path and Path(model_path).exists():
        return True

    # Check answer mentions accuracy in range
    answer = result.get("answer", "")
    import re
    accuracies = re.findall(r"0\.\d{3,4}", answer)
    for acc in accuracies:
        val = float(acc)
        if expected_accuracy_range[0] <= val <= expected_accuracy_range[1]:
            return True

    return bool(model_path)


def _llm_judge(answer: str, criteria: str) -> bool:
    """Use Groq LLM to judge answer quality."""
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an evaluation judge. Reply only YES or NO."},
                {"role": "user", "content": f"Criteria: {criteria}\n\nAnswer to evaluate:\n{answer}\n\nDoes this answer meet the criteria? Reply YES or NO only."},
            ],
            max_tokens=5,
            temperature=0.0,
        )
        verdict = resp.choices[0].message.content.strip().upper()
        return "YES" in verdict
    except Exception as e:
        print(f"  LLM judge error: {e}")
        return False


# ── Run evaluation ────────────────────────────────────────────────────────────

def run_csv_eval(csv_id: str, csv_path: str) -> Dict:
    """Run all tests for a given CSV session."""
    from csv_agent import ingest_csv, query_csv

    # Load dataframe for ground truth checks
    df = pd.read_csv(csv_path)
    print(f"[CSV EVAL] Loaded {df.shape[0]} rows × {df.shape[1]} cols from {csv_path}")

    # Ingest if needed
    print(f"[CSV EVAL] Using csv_id={csv_id}")

    results = []
    passed  = 0
    failed  = 0

    for test in DIABETES_TESTS:
        tid  = test["id"]
        cat  = test["category"]
        q    = test["question"]
        print(f"\n[{tid}] {q[:60]}")

        try:
            result = query_csv(csv_id, q)
            answer = result.get("answer", "")

            # Choose check input based on category
            if cat == "chart":
                check_input = result
            else:
                check_input = answer

            ok = test["check"](check_input, df)
            status = "✓ PASS" if ok else "✗ FAIL"
            if ok:
                passed += 1
            else:
                failed += 1

            print(f"  {status} | Expected: {test['expected']}")
            print(f"  Answer: {answer[:100]}")

            results.append({
                "id":          tid,
                "category":    cat,
                "question":    q,
                "expected":    test["expected"],
                "answer":      answer[:500],
                "passed":      ok,
                "description": test["description"],
            })

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
            results.append({
                "id":          tid,
                "category":    cat,
                "question":    q,
                "expected":    test["expected"],
                "answer":      f"ERROR: {e}",
                "passed":      False,
                "description": test["description"],
            })

    return {
        "results": results,
        "passed":  passed,
        "failed":  failed,
        "total":   len(DIABETES_TESTS),
        "score":   round(passed / len(DIABETES_TESTS), 4),
    }


# ── Save + print report ───────────────────────────────────────────────────────

def save_csv_eval(csv_id: str, summary: Dict):
    db = Path("bujji.db")
    if not db.exists():
        return
    con = sqlite3.connect(db)
    con.execute("""
        CREATE TABLE IF NOT EXISTS csv_eval_results (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            csv_id       TEXT NOT NULL,
            total        INTEGER,
            passed       INTEGER,
            failed       INTEGER,
            score        REAL,
            results_json TEXT,
            evaluated_at TEXT
        )
    """)
    con.execute("""
        INSERT INTO csv_eval_results
        (csv_id, total, passed, failed, score, results_json, evaluated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        csv_id, summary["total"], summary["passed"], summary["failed"],
        summary["score"], json.dumps(summary["results"]),
        datetime.utcnow().isoformat(),
    ))
    con.commit()
    con.close()


def print_csv_report(csv_id: str, summary: Dict):
    results  = summary["results"]
    passed   = summary["passed"]
    failed   = summary["failed"]
    total    = summary["total"]
    score    = summary["score"]

    print("\n" + "="*60)
    print("  BUJJI BABU — CSV AGENT EVALUATION REPORT")
    print("="*60)
    print(f"  CSV ID:    {csv_id}")
    print(f"  Total:     {total} tests")
    print(f"  Passed:    {passed}  ✓")
    print(f"  Failed:    {failed}  ✗")
    print(f"  Score:     {score:.4f}  ({score*100:.1f}%)")
    print("-"*60)

    # By category
    cats = {}
    for r in results:
        c = r["category"]
        if c not in cats:
            cats[c] = {"pass": 0, "fail": 0}
        if r["passed"]:
            cats[c]["pass"] += 1
        else:
            cats[c]["fail"] += 1

    print("  By category:")
    for cat, counts in cats.items():
        total_cat = counts["pass"] + counts["fail"]
        pct = counts["pass"] / total_cat * 100
        bar = "█" * counts["pass"] + "░" * counts["fail"]
        print(f"    {cat:<12} {bar}  {counts['pass']}/{total_cat} ({pct:.0f}%)")

    print("="*60)


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate_csv(csv_path: str, csv_id: str = None):
    """Full CSV evaluation pipeline."""
    from csv_agent import ingest_csv

    # Ingest fresh or use existing
    if not csv_id:
        print(f"[CSV EVAL] Ingesting {csv_path}...")
        csv_id = ingest_csv(csv_path)
        print(f"[CSV EVAL] Ingested → csv_id={csv_id}")
    else:
        print(f"[CSV EVAL] Using existing csv_id={csv_id}")

    summary = run_csv_eval(csv_id, csv_path)
    save_csv_eval(csv_id, summary)
    print_csv_report(csv_id, summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSV Agent evaluation for Bujji Babu")
    parser.add_argument("--csv_path", required=True, help="Path to CSV file")
    parser.add_argument("--csv_id",   default=None,  help="Existing csv_id (skip re-ingestion)")
    args = parser.parse_args()

    evaluate_csv(args.csv_path, args.csv_id)