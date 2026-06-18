"""
adaptive_retrieval.py — Adaptive Retrieval for Bujji Babu PDF RAG

Problem with fixed top-k:
  top-k=3 → fast but misses context for complex questions
  top-k=8 → slow and noisy for simple factoid questions

  A question like "Who wrote this?" needs 2 nodes at most.
  A question like "Compare the methodology across all experiments"
  needs 8-10 nodes.

  Fixed top-k is always wrong for one of these cases.

Adaptive strategy:
  Round 1: Retrieve top-3 (fast)
  → Check confidence score
  → If confident (>= 0.55): done, use these 3
  → If borderline (0.35-0.55): expand to top-6, re-merge
  → If low (< 0.35): expand to top-10, re-merge, still may gate

  Also adapts based on query type:
  - Factoid  (who/what/when/where) → start=3,  max=5
  - Analytical (compare/analyse)   → start=5,  max=10
  - Summary  (summarize/overview)  → start=8,  max=12

Research contribution:
  Dynamic retrieval depth based on query complexity and confidence
  feedback is not standard in published RAG systems.
"""

import re
from typing import List, Dict, Tuple, Optional

# ── Query type detection ──────────────────────────────────────────────────────

FACTOID_PATTERNS = [
    r"^(who is|who are|who was|who were|who wrote|who created|who developed)",
    r"^(when (was|is|did|were)|what year|what date)",
    r"^(where (is|was|are|were|did))",
    r"^(which (one|paper|model|method|year|author))",
    r"^(name the|give me the name|what is the name|who is the author)",
    r"(how many|how much|what (number|count|total|percentage))",
]

ANALYTICAL_PATTERNS = [
    r"(compare|contrast|difference|similarities|versus|vs\.)",
    r"(analyse|analyze|evaluate|assess|examine|discuss)",
    r"(why|how does|explain|what causes|what led to)",
    r"(methodology|approach|framework|model|algorithm)",
]

SUMMARY_PATTERNS = [
    r"(summarize|summarise|overview|summary|outline)",
    r"(main (points|findings|contributions|results|ideas))",
    r"(tell me about|what is this (paper|document|book) about)",
    r"(key (ideas|concepts|takeaways|insights|contributions|findings))",
    r"(what are the (contributions|findings|results|implications))",
]


def detect_query_type(question: str) -> str:
    """
    Classify question into: factoid | analytical | summary | general

    Returns one of: "factoid", "analytical", "summary", "general"
    """
    q = question.lower().strip()

    # Check summary first (most specific)
    for pattern in SUMMARY_PATTERNS:
        if re.search(pattern, q):
            return "summary"

    # Check analytical
    for pattern in ANALYTICAL_PATTERNS:
        if re.search(pattern, q):
            return "analytical"

    # Check factoid
    for pattern in FACTOID_PATTERNS:
        if re.search(pattern, q):
            return "factoid"

    return "general"


# ── Retrieval budget per query type ──────────────────────────────────────────

RETRIEVAL_BUDGET = {
    #             initial_k, expand_k, max_k
    "factoid":   (3,         4,        4),  # tight — factoid needs few precise chunks
    "general":   (4,         6,        8),
    "analytical":(5,         8,        12),
    "summary":   (6,         10,       14),
}


def get_retrieval_budget(query_type: str) -> Tuple[int, int, int]:
    """Return (initial_k, expand_k, max_k) for a given query type."""
    return RETRIEVAL_BUDGET.get(query_type, RETRIEVAL_BUDGET["general"])


# ── Confidence check ──────────────────────────────────────────────────────────

EXPAND_THRESHOLD = 0.50   # below this → expand retrieval
GOOD_THRESHOLD   = 0.55   # above this → no expansion needed


def _should_expand(confidence: float, query_type: str) -> bool:
    """
    Decide whether to expand retrieval based on confidence + query type.
    Analytical and summary queries have a higher expansion threshold.
    """
    if query_type in ("analytical", "summary"):
        # Always expand for complex queries — need more context
        return True
    return confidence < EXPAND_THRESHOLD


# ── Deduplication ─────────────────────────────────────────────────────────────

def _dedup_nodes(nodes: List[Dict]) -> List[Dict]:
    """Remove duplicate nodes by text prefix."""
    seen = set()
    result = []
    for node in nodes:
        key = node.get("text", "")[:80].lower().strip()
        if key not in seen:
            seen.add(key)
            result.append(node)
    return result


# ── Main adaptive retrieval ───────────────────────────────────────────────────

def adaptive_retrieve(
    pdf_id: str,
    sub_queries: List[str],
    query_type: str,
    retrieval_fn,          # callable: (pdf_id, query, k) -> List[Dict]
    bm25_fn,               # callable: (pdf_id, nodes, query, k) -> List[Dict]
    rrf_fn,                # callable: (chroma, bm25, top_n) -> List[Dict]
    confidence_fn,         # callable: (question, nodes, scores) -> Dict
    all_nodes: List,
    primary_question: str,
) -> Tuple[List[Dict], List[Dict], float, str]:
    """
    Adaptive retrieval pipeline.

    Round 1: retrieve with initial_k
    → compute confidence
    → if low: expand to expand_k and re-retrieve
    → if still low and analytical/summary: expand to max_k

    Args:
        pdf_id:           PDF identifier
        sub_queries:      List of (possibly decomposed) sub-queries
        query_type:       "factoid" | "analytical" | "summary" | "general"
        retrieval_fn:     ChromaDB query function
        bm25_fn:          BM25 retrieval function
        rrf_fn:           RRF merge function
        confidence_fn:    gate_answer function
        all_nodes:        All nodes for BM25
        primary_question: Original question for confidence scoring

    Returns:
        (merged_nodes, all_chroma_results, confidence_score, expansion_log)
    """
    initial_k, expand_k, max_k = get_retrieval_budget(query_type)
    log_parts = [f"type={query_type} budget=({initial_k},{expand_k},{max_k})"]

    # ── Round 1: Initial retrieval ────────────────────────────────────────────
    all_chroma  = []
    results_r1  = []

    for sub_q in sub_queries:
        from query_rewriter import _needs_rewriting
        # HyDE expand
        try:
            from query import _hyde_expand
            # Skip HyDE for factoid — direct question finds author/title pages better
            hypo = sub_q if query_type == "factoid" else _hyde_expand(sub_q)
        except Exception:
            hypo = sub_q

        c_res = retrieval_fn(pdf_id, hypo, initial_k)
        b_res = bm25_fn(pdf_id, all_nodes, sub_q, initial_k)
        merged = rrf_fn(c_res, b_res, top_n=initial_k)
        results_r1.extend(merged)
        all_chroma.extend(c_res)

    results_r1 = _dedup_nodes(results_r1)
    chroma_scores = [n.get("score", 0.0) for n in all_chroma]
    gate = confidence_fn(primary_question, results_r1, chroma_scores)
    confidence = gate["confidence"]

    log_parts.append(f"round1_k={initial_k} conf={confidence:.3f}")
    print(f"[ADAPTIVE] Round 1: {len(results_r1)} nodes, confidence={confidence:.3f}")

    # ── Check if expansion needed ─────────────────────────────────────────────
    if not _should_expand(confidence, query_type):
        log_parts.append("no_expansion")
        print(f"[ADAPTIVE] No expansion needed")
        return results_r1, all_chroma, confidence, " | ".join(log_parts)

    # ── Round 2: Expanded retrieval ───────────────────────────────────────────
    print(f"[ADAPTIVE] Expanding to k={expand_k}...")
    all_chroma_r2 = []
    results_r2    = []

    for sub_q in sub_queries:
        try:
            from query import _hyde_expand
            hypo = _hyde_expand(sub_q)
        except Exception:
            hypo = sub_q

        c_res = retrieval_fn(pdf_id, hypo, expand_k)
        b_res = bm25_fn(pdf_id, all_nodes, sub_q, expand_k)
        merged = rrf_fn(c_res, b_res, top_n=expand_k)
        results_r2.extend(merged)
        all_chroma_r2.extend(c_res)

    results_r2    = _dedup_nodes(results_r2)
    chroma_scores2 = [n.get("score", 0.0) for n in all_chroma_r2]
    gate2 = confidence_fn(primary_question, results_r2, chroma_scores2)
    confidence2 = gate2["confidence"]

    log_parts.append(f"round2_k={expand_k} conf={confidence2:.3f}")
    print(f"[ADAPTIVE] Round 2: {len(results_r2)} nodes, confidence={confidence2:.3f}")

    # ── Round 3: Max retrieval (only for analytical/summary still low) ────────
    if confidence2 < EXPAND_THRESHOLD and query_type in ("analytical", "summary"):
        print(f"[ADAPTIVE] Max expansion to k={max_k}...")
        all_chroma_r3 = []
        results_r3    = []

        for sub_q in sub_queries:
            try:
                from query import _hyde_expand
                hypo = _hyde_expand(sub_q)
            except Exception:
                hypo = sub_q

            c_res = retrieval_fn(pdf_id, hypo, max_k)
            b_res = bm25_fn(pdf_id, all_nodes, sub_q, max_k)
            merged = rrf_fn(c_res, b_res, top_n=max_k)
            results_r3.extend(merged)
            all_chroma_r3.extend(c_res)

        results_r3     = _dedup_nodes(results_r3)
        chroma_scores3 = [n.get("score", 0.0) for n in all_chroma_r3]
        gate3 = confidence_fn(primary_question, results_r3, chroma_scores3)
        confidence3 = gate3["confidence"]

        log_parts.append(f"round3_k={max_k} conf={confidence3:.3f}")
        print(f"[ADAPTIVE] Round 3: {len(results_r3)} nodes, confidence={confidence3:.3f}")
        return results_r3, all_chroma_r3, confidence3, " | ".join(log_parts)

    return results_r2, all_chroma_r2, confidence2, " | ".join(log_parts)