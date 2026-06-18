"""
faithfulness.py — Faithfulness Scoring for Bujji Babu PDF RAG

Problem it solves:
  LLM generates a confident answer but some sentences come from its
  training data, not from the retrieved document chunks.
  User has no way to know which parts are grounded vs hallucinated.

What it does:
  1. Split answer into individual sentences
  2. For each sentence, check if it is supported by retrieved context
     using a lightweight NLI (Natural Language Inference) approach
  3. Score each sentence: SUPPORTED / UNSUPPORTED / UNCERTAIN
  4. Return overall faithfulness score + per-sentence breakdown
  5. Frontend highlights unsupported sentences in amber

Two modes:
  - Fast mode (default): keyword + overlap heuristic — no extra API call
  - Deep mode: Groq NLI check — more accurate, costs one API call

Research contribution:
  Sentence-level faithfulness grounding with visual highlighting
  is not standard in open-source RAG systems. This is publishable.
"""

import re
from typing import List, Dict, Tuple, Optional
from llm_client import chat_text

# ── Thresholds ────────────────────────────────────────────────────────────────

SUPPORTED_THRESHOLD   = 0.45   # overlap score above this → supported
UNSUPPORTED_THRESHOLD = 0.15   # overlap score below this → unsupported
# Between the two → uncertain

# Sentences shorter than this are usually connectives — skip scoring
MIN_SENTENCE_TOKENS = 6

# Stopwords for overlap calculation
STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could",
    "should","may","might","shall","can","in","on","at","to",
    "for","of","and","or","but","not","with","this","that",
    "these","those","it","its","by","from","as","so","if",
    "then","than","also","just","more","some","such","there",
}


# ── Sentence splitter ─────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """
    Split answer text into sentences.
    Handles abbreviations, decimal numbers, and bullet points.
    """
    # Protect common abbreviations
    text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Fig|et al|i\.e|e\.g|vs|approx|dept)\.',
                  lambda m: m.group().replace('.', '<DOT>'), text)

    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Restore protected dots
    sentences = [s.replace('<DOT>', '.').strip() for s in sentences]

    # Filter out very short fragments
    return [s for s in sentences if len(s.split()) >= MIN_SENTENCE_TOKENS]


# ── Keyword overlap scorer ────────────────────────────────────────────────────

def _content_words(text: str) -> set:
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return {w for w in words if w not in STOPWORDS}


def _overlap_score(sentence: str, context: str) -> float:
    """
    Fraction of sentence content words present in context.
    Returns 0.0–1.0.
    """
    s_words = _content_words(sentence)
    if not s_words:
        return 0.5  # can't judge

    c_words = _content_words(context)
    matched = s_words & c_words
    return len(matched) / len(s_words)


# ── Fast heuristic scoring ────────────────────────────────────────────────────

def _fast_score_sentence(sentence: str, context: str) -> Tuple[str, float]:
    """
    Fast keyword-overlap faithfulness check.
    Returns (label, score) where label is SUPPORTED/UNCERTAIN/UNSUPPORTED.
    """
    score = _overlap_score(sentence, context)

    if score >= SUPPORTED_THRESHOLD:
        return "supported", score
    elif score >= UNSUPPORTED_THRESHOLD:
        return "uncertain", score
    else:
        return "unsupported", score


# ── Deep NLI scoring (Groq) ───────────────────────────────────────────────────

def _deep_score_batch(
    sentences: List[str],
    context: str,
) -> List[Tuple[str, float]]:
    """
    Use Groq LLM as NLI classifier for a batch of sentences.
    More accurate than keyword overlap but costs one API call.
    """
    context_snippet = context[:2000]

    sentences_formatted = "\n".join(
        f"{i+1}. {s}" for i, s in enumerate(sentences)
    )

    prompt = f"""You are a faithfulness checker for a RAG system.

Retrieved document context:
\"\"\"
{context_snippet}
\"\"\"

For each sentence below, decide if it is:
- SUPPORTED: The sentence is directly supported by the context above
- UNCERTAIN: The sentence might be supported but it's unclear  
- UNSUPPORTED: The sentence contradicts or is not found in the context

Sentences to check:
{sentences_formatted}

Reply ONLY with a JSON array like:
[{{"label": "SUPPORTED", "score": 0.9}}, {{"label": "UNSUPPORTED", "score": 0.1}}, ...]
One entry per sentence in the same order. No other text."""

    try:
        raw = chat_text([{"role": "user", "content": prompt}], max_tokens=300, temperature=0.0)
        raw = raw.replace("```json", "").replace("```", "").strip()

        import json
        results = json.loads(raw)

        output = []
        for item in results[:len(sentences)]:
            label = item.get("label", "UNCERTAIN").upper()
            score = float(item.get("score", 0.5))
            if label not in ("SUPPORTED", "UNSUPPORTED", "UNCERTAIN"):
                label = "UNCERTAIN"
            output.append((label.lower(), score))

        # Pad if LLM returned fewer items
        while len(output) < len(sentences):
            output.append(("uncertain", 0.5))

        return output

    except Exception as e:
        print(f"[FAITHFULNESS] Deep scoring failed ({e}), falling back to fast")
        return [_fast_score_sentence(s, context) for s in sentences]


# ── Main faithfulness scorer ──────────────────────────────────────────────────

def score_faithfulness(
    answer: str,
    context: str,
    deep: bool = False,
) -> Dict:
    """
    Score the faithfulness of a generated answer against retrieved context.

    Args:
        answer:   Generated LLM answer
        context:  Retrieved context string (joined node texts)
        deep:     Use Groq NLI (more accurate) vs fast heuristic

    Returns:
        {
            "overall_score":  float 0.0–1.0
            "label":          "faithful" | "partial" | "unfaithful"
            "sentences": [
                {
                    "text":   str,
                    "label":  "supported" | "uncertain" | "unsupported",
                    "score":  float,
                }
            ]
        }
    """
    sentences = _split_sentences(answer)

    if not sentences:
        return {
            "overall_score": 1.0,
            "label":         "faithful",
            "sentences":     [],
        }

    # Score each sentence
    if deep and len(sentences) <= 8:
        # Deep mode: batch all sentences in one call (max 8 for token budget)
        scores = _deep_score_batch(sentences, context)
    else:
        # Fast mode: heuristic overlap
        scores = [_fast_score_sentence(s, context) for s in sentences]

    # Build sentence results
    sentence_results = []
    score_values = []

    for sentence, (label, score) in zip(sentences, scores):
        sentence_results.append({
            "text":  sentence,
            "label": label,
            "score": round(score, 3),
        })
        score_values.append(score)

    # Overall score = mean of sentence scores
    overall = sum(score_values) / len(score_values) if score_values else 1.0
    overall = round(overall, 3)

    # Overall label
    if overall >= 0.55:
        label = "faithful"
    elif overall >= 0.30:
        label = "partial"
    else:
        label = "unfaithful"

    supported_count   = sum(1 for s in sentence_results if s["label"] == "supported")
    unsupported_count = sum(1 for s in sentence_results if s["label"] == "unsupported")

    print(
        f"[FAITHFULNESS] score={overall:.3f} label={label} "
        f"supported={supported_count}/{len(sentences)} "
        f"unsupported={unsupported_count}/{len(sentences)}"
    )

    return {
        "overall_score":    overall,
        "label":            label,
        "sentences":        sentence_results,
        "supported_count":  supported_count,
        "unsupported_count": unsupported_count,
        "total_sentences":  len(sentences),
    }