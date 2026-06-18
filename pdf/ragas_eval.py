"""
ragas_eval.py — PDF RAG Evaluation using RAGAS v0.4+

Uses new RAGAS API with Groq via OpenAI-compatible client.
Reads all keys from .groq_keys.txt for rotation.
"""

import os
import sys
import json
import argparse
import sqlite3
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv
load_dotenv()


# ── Load all Groq keys ────────────────────────────────────────────────────────

def _load_keys() -> List[str]:
    keys = []
    env_key = os.getenv("GROQ_API_KEY", "")
    if env_key:
        keys.append(env_key)
    kf = Path(".groq_keys.txt")
    if kf.exists():
        for line in kf.read_text().splitlines():
            k = line.strip()
            if k and k.startswith("gsk_") and k not in keys:
                keys.append(k)
    print(f"[RAGAS] Loaded {len(keys)} Groq key(s)")
    return keys

_KEYS = _load_keys()
_KEY_IDX = 0

def _get_key() -> str:
    return _KEYS[_KEY_IDX % len(_KEYS)] if _KEYS else os.getenv("GROQ_API_KEY","")

def _rotate():
    global _KEY_IDX
    _KEY_IDX = (_KEY_IDX + 1) % max(1, len(_KEYS))
    print(f"[RAGAS] Rotated to key index {_KEY_IDX}")


# ── RAGAS setup with Groq ─────────────────────────────────────────────────────

def make_ragas_llm():
    """Create RAGAS LLM using Groq via OpenAI-compatible endpoint."""
    from openai import OpenAI
    from ragas.llms import llm_factory
    client = OpenAI(
        api_key=_get_key(),
        base_url="https://api.groq.com/openai/v1",
    )
    return llm_factory("llama-3.1-8b-instant", provider="openai", client=client)


# Note: Groq does not serve embeddings.
# AnswerRelevancy is excluded — it requires an embedding model.
# We use Faithfulness, ContextPrecision, ContextRecall (LLM-only metrics).


# ── Synthetic Q&A generation ──────────────────────────────────────────────────

def generate_qa_pairs(chunks: List[str], n: int = 10) -> List[Dict]:
    """Generate Q&A pairs from document chunks using Groq."""
    from groq import Groq

    step = max(1, len(chunks) // n)
    sampled = [chunks[i] for i in range(0, len(chunks), step)][:n]
    qa_pairs = []

    for i, chunk in enumerate(sampled):
        if len(chunk.strip()) < 50:
            continue
        # Rotate key every call to spread load across 30 keys
        _rotate()

        client = Groq(api_key=_get_key())
        prompt = f"""Given this document excerpt, generate ONE factual question and its answer.

Excerpt:
{chunk[:800]}

Rules:
- Question must be answerable from the excerpt alone
- Answer must be 1-2 sentences, specific
- Return ONLY valid JSON: {{"question": "...", "answer": "..."}}"""

        for attempt in range(len(_KEYS)):
            try:
                resp = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=250,
                    temperature=0.3,
                )
                raw = resp.choices[0].message.content.strip()
                raw = raw.replace("```json","").replace("```","").strip()
                parsed = json.loads(raw)
                if "question" in parsed and "answer" in parsed:
                    qa_pairs.append({
                        "question":     parsed["question"],
                        "ground_truth": parsed["answer"],
                        "context":      chunk[:800],
                    })
                    print(f"  ✓ Q{i+1}: {parsed['question'][:60]}")
                break
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    _rotate()
                    client = Groq(api_key=_get_key())
                    time.sleep(2)
                else:
                    print(f"  ✗ Q{i+1}: {e}")
                    break

    return qa_pairs


# ── Run queries through Bujji Babu ────────────────────────────────────────────

def run_queries(pdf_id: str, qa_pairs: List[Dict], user_id: int = 1) -> List[Dict]:
    from query import query_pdf
    from ingestion import get_all_nodes

    results = []
    for i, qa in enumerate(qa_pairs):
        q = qa["question"]
        print(f"\n[EVAL] Q{i+1}/{len(qa_pairs)}: {q[:60]}")
        try:
            result = query_pdf(pdf_id=pdf_id, question=q, top_k=5, user_id=user_id)
            answer = result.get("answer", "")

            # Get retrieved text contexts
            all_nodes = get_all_nodes(pdf_id)
            contexts = []
            if all_nodes:
                from rank_bm25 import BM25Okapi
                corpus = [n.text for n in all_nodes if hasattr(n,'text') and len(n.text.strip())>30]
                if corpus:
                    bm25 = BM25Okapi([t.split() for t in corpus])
                    scores = bm25.get_scores(q.split())
                    top_idx = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)[:5]
                    contexts = [corpus[j] for j in top_idx]

            if not contexts:
                contexts = [qa["context"]]

            results.append({
                "question":     q,
                "answer":       answer,
                "contexts":     contexts,
                "ground_truth": qa["ground_truth"],
            })
            print(f"  → {answer[:80]}")
        except Exception as e:
            print(f"  ✗ Query failed: {e}")
            results.append({
                "question":     q,
                "answer":       "",
                "contexts":     [qa["context"]],
                "ground_truth": qa["ground_truth"],
            })
    return results


# ── Run RAGAS ─────────────────────────────────────────────────────────────────

def run_ragas(results: List[Dict]) -> Dict:
    import warnings
    from datasets import Dataset
    from ragas import evaluate, RunConfig

    # Use old-style singleton metrics — they inherit ragas.metrics.base.Metric
    # which is what evaluate() checks with isinstance().
    # New ragas.metrics.collections classes inherit BaseMetric and FAIL the check.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import (
            faithfulness,
            context_precision,
            context_recall,
        )

    # Set LLM on each singleton
    ragas_llm = make_ragas_llm()
    faithfulness.llm      = ragas_llm
    context_precision.llm = ragas_llm
    context_recall.llm    = ragas_llm

    metrics = [faithfulness, context_precision, context_recall]

    dataset = Dataset.from_dict({
        "question":     [r["question"] for r in results],
        "answer":       [r["answer"] for r in results],
        "contexts":     [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })

    # With 30 keys we can run 3 workers safely — ~3x faster
    run_config = RunConfig(
        timeout     = 180,
        max_retries = len(_KEYS),
        max_wait    = 60,
        max_workers = 3,
    )

    print("\n[RAGAS] Running evaluation (3 metrics: faithfulness, context_precision, context_recall)...")
    score = evaluate(
        dataset          = dataset,
        metrics          = metrics,
        run_config       = run_config,
        raise_exceptions = False,
        show_progress    = True,
    )

    def _scalar(v):
        """Handle both scalar and list returns from RAGAS."""
        import numpy as np
        if isinstance(v, (list, tuple)):
            vals = [x for x in v if x is not None and str(x) != 'nan']
            return round(float(np.mean(vals)) if vals else 0.0, 4)
        try:
            return round(float(v), 4)
        except (TypeError, ValueError):
            return 0.0

    return {
        "faithfulness":      _scalar(score["faithfulness"]),
        "context_precision": _scalar(score["context_precision"]),
        "context_recall":    _scalar(score["context_recall"]),
    }


# ── Save + print ──────────────────────────────────────────────────────────────

def save_results(pdf_id: str, scores: Dict, n: int):
    db = Path("bujji.db")
    if not db.exists(): return
    con = sqlite3.connect(db)
    con.execute("""CREATE TABLE IF NOT EXISTS ragas_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pdf_id TEXT, n_questions INTEGER,
        faithfulness REAL, context_precision REAL,
        context_recall REAL, evaluated_at TEXT)""")
    con.execute("INSERT INTO ragas_results VALUES (NULL,?,?,?,?,?,?)", (
        pdf_id, n, scores["faithfulness"],
        scores["context_precision"], scores["context_recall"],
        datetime.utcnow().isoformat()))
    con.commit(); con.close()


def print_report(pdf_id: str, scores: Dict, results: List[Dict]):
    overall = sum(scores.values()) / len(scores)
    print("\n" + "="*60)
    print("  BUJJI BABU — PDF RAG EVALUATION REPORT")
    print("="*60)
    print(f"  PDF:              {pdf_id}")
    print(f"  Questions:        {len(results)}")
    print(f"  Keys available:   {len(_KEYS)}")
    print("-"*60)
    print(f"  Faithfulness:       {scores['faithfulness']:.4f}  ({scores['faithfulness']*100:.1f}%)")
    print(f"  Context Precision:  {scores['context_precision']:.4f}  ({scores['context_precision']*100:.1f}%)")
    print(f"  Context Recall:     {scores['context_recall']:.4f}  ({scores['context_recall']*100:.1f}%)")
    print(f"  {'─'*40}")
    print(f"  Overall:            {overall:.4f}  ({overall*100:.1f}%)")
    print(f"  (AnswerRelevancy excluded — requires embedding model)")
    print("="*60)


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate_pdf(pdf_id: str, n_questions: int = 5, user_id: int = 1):
    print(f"\n[RAGAS] Evaluating pdf_id={pdf_id}, n={n_questions}, keys={len(_KEYS)}")

    from ingestion import get_all_nodes
    all_nodes = get_all_nodes(pdf_id)
    if not all_nodes:
        print("[RAGAS] No nodes — run ingestion first"); return None

    chunks = [n.text for n in all_nodes
              if hasattr(n,'text') and len(n.text.strip())>100]
    print(f"[RAGAS] {len(chunks)} chunks available")

    qa_pairs = generate_qa_pairs(chunks, n=n_questions)
    if not qa_pairs:
        print("[RAGAS] No Q&A pairs generated"); return None

    results = run_queries(pdf_id, qa_pairs, user_id=user_id)
    scores  = run_ragas(results)
    save_results(pdf_id, scores, len(results))
    print_report(pdf_id, scores, results)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_id",  required=True)
    parser.add_argument("--n",       type=int, default=5)
    parser.add_argument("--user_id", type=int, default=1)
    args = parser.parse_args()
    evaluate_pdf(args.pdf_id, args.n, args.user_id)