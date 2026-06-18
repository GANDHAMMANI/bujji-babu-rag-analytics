"""
confidence.py — Confidence Scoring & Gating for Bujji Babu PDF RAG

Problem it solves:
  User asks something not in the document → current system hallucinates an answer
  using loosely related context chunks.

  With confidence gating:
  - Score = 0.0–0.35  → "I couldn't find this in the document" (hard block)
  - Score = 0.35–0.55 → Answer with low-confidence warning
  - Score = 0.55+     → Answer normally

How scoring works:
  1. Primary:  mean of top-3 ChromaDB cosine similarity scores
  2. Secondary: keyword overlap between question and retrieved text
  3. Combined: weighted average → single confidence float 0.0–1.0
"""

import re
from typing import List, Dict, Tuple

# ── Thresholds ────────────────────────────────────────────────────────────────

THRESHOLD_BLOCK  = 0.35   # Below this → refuse to answer
THRESHOLD_WARN   = 0.55   # Below this → answer with warning
THRESHOLD_GOOD   = 0.55   # Above this → answer normally

# Questions that should never be blocked (greetings, meta questions)
ALWAYS_ANSWER_PATTERNS = [
    r"^(hi|hello|hey|thanks|thank you)",
    r"what (is this|are you|can you)",
    r"how (do|can|should) (i|you|we)",
    r"(summarize|summarise|overview|what is the document about)",
    r"(list|show|give me).{0,20}(page|chapter|section|topic)",
]


# ── Keyword overlap scorer ────────────────────────────────────────────────────

def _keyword_overlap(question: str, nodes: List[Dict]) -> float:
    """
    Simple keyword overlap between question and retrieved text.
    Ignores stopwords. Returns 0.0–1.0.
    """
    STOPWORDS = {
        "a","an","the","is","are","was","were","be","been","being",
        "have","has","had","do","does","did","will","would","could",
        "should","may","might","shall","can","need","dare","ought",
        "in","on","at","to","for","of","and","or","but","not","with",
        "this","that","these","those","what","who","how","when","where",
        "which","why","i","you","he","she","it","we","they","me","him",
        "her","us","them","my","your","his","its","our","their",
    }

    # Extract content words from question
    q_words = set(
        w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', question)
        if w.lower() not in STOPWORDS
    )
    if not q_words:
        return 0.5  # Can't judge, give neutral score

    # Check presence in top-3 nodes
    combined_text = " ".join(
        n.get("text", "").lower() for n in nodes[:3]
    )
    node_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', combined_text))

    matched = q_words & node_words
    overlap = len(matched) / len(q_words)

    return min(overlap, 1.0)


# ── Main confidence scorer ────────────────────────────────────────────────────

def score_retrieval(
    question: str,
    nodes: List[Dict],
    chroma_scores: List[float],
) -> Tuple[float, str]:
    """
    Compute a confidence score for the retrieval result.

    Args:
        question:       Original (or rewritten) user question
        nodes:          Retrieved nodes (dicts with 'text', 'metadata', 'score')
        chroma_scores:  Raw cosine similarity scores from ChromaDB

    Returns:
        (confidence: float 0.0–1.0, reason: str)
    """
    if not nodes:
        return 0.0, "no_nodes"

    # Never block simple/meta questions
    q_lower = question.lower().strip()
    for pattern in ALWAYS_ANSWER_PATTERNS:
        if re.search(pattern, q_lower):
            return 1.0, "always_answer"

    # ── Primary: ChromaDB similarity scores ───────────────────────────────────
    top_scores = sorted(chroma_scores, reverse=True)[:3]
    if top_scores:
        chroma_conf = sum(top_scores) / len(top_scores)
    else:
        chroma_conf = 0.0

    # ── Secondary: keyword overlap ────────────────────────────────────────────
    kw_conf = _keyword_overlap(question, nodes)

    # ── Combined: weight chroma more (it's more reliable) ────────────────────
    confidence = (chroma_conf * 0.65) + (kw_conf * 0.35)

    # ── Determine reason ──────────────────────────────────────────────────────
    if confidence < THRESHOLD_BLOCK:
        reason = "low_similarity"
    elif confidence < THRESHOLD_WARN:
        reason = "borderline"
    else:
        reason = "confident"

    return round(confidence, 3), reason


# ── Decision maker ────────────────────────────────────────────────────────────

def gate_answer(
    question: str,
    nodes: List[Dict],
    chroma_scores: List[float],
) -> Dict:
    """
    Decide whether to answer, warn, or block based on confidence.

    Returns dict with:
        should_answer:  bool — if False, return the fallback message
        confidence:     float
        warning:        str or None — shown below answer if borderline
        fallback:       str — message to return if should_answer is False
    """
    confidence, reason = score_retrieval(question, nodes, chroma_scores)

    print(f"[CONFIDENCE] score={confidence:.3f} reason={reason}")

    if reason == "always_answer":
        return {
            "should_answer": True,
            "confidence":    confidence,
            "warning":       None,
            "fallback":      None,
        }

    if confidence < THRESHOLD_BLOCK:
        return {
            "should_answer": False,
            "confidence":    confidence,
            "warning":       None,
            "fallback": (
                "I couldn't find relevant information about that in this document. "
                "Try rephrasing your question, or check if this topic is covered in the PDF."
            ),
        }

    if confidence < THRESHOLD_WARN:
        return {
            "should_answer": True,
            "confidence":    confidence,
            "warning": (
                "The document may not fully cover this topic — "
                "this answer is based on loosely related content."
            ),
            "fallback":      None,
        }

    return {
        "should_answer": True,
        "confidence":    confidence,
        "warning":       None,
        "fallback":      None,
    }