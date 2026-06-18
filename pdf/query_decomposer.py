"""
query_decomposer.py — Multi-part Query Decomposition for Bujji Babu PDF RAG

Problem it solves:
  "Compare the methodology in section 3 with the results in section 5
   and tell me who the authors are"

  This is 3 sub-questions. A single retrieval pass finds context for
  one of them at best. The answer is always incomplete.

  With decomposition:
  1. Detect if question has multiple parts
  2. Split into focused sub-queries
  3. Retrieve independently for each
  4. Merge all context before generation → complete answer

How it works:
  - Fast heuristic first (conjunction/comparison keywords) → skip API if simple
  - If complex → single Groq call to decompose into 2-4 sub-queries
  - Each sub-query retrieved independently (ChromaDB + BM25)
  - Context merged with source labelling so LLM knows what came from where
"""

import re
from typing import List, Dict, Tuple, Optional
from llm_client import chat_text

# ── Signals that a question has multiple parts ────────────────────────────────

DECOMPOSE_TRIGGERS = [
    # Comparison
    "compare", "contrast", "difference between", "similarities between",
    "vs", "versus", "how does .{1,30} differ", "what is the difference",
    # Enumeration
    "and also", "as well as", "in addition", "furthermore",
    "both .{1,30} and", "list .{1,30} and",
    # Multi-part connectors
    "firstly", "secondly", "thirdly", "finally",
    "what are .{1,30} and (how|why|when)",
    # Explicit multi-question
    r"\?\s+and\s+", r"\?\s+also\s+", r"\?\s+what\s+",
    # Section references
    "section .{1,10} and section", "chapter .{1,10} and chapter",
    "table .{1,10} and (figure|table)",
]

# Questions that look complex but are actually single-intent
SINGLE_INTENT_PATTERNS = [
    r"^(what|who|where|when|why|how) (is|are|was|were|did|does) .{1,60}\?$",
    r"^(explain|describe|summarize|define) .{1,60}$",
    r"^(list|show|give me) .{1,60}$",
]

MAX_SUB_QUERIES = 4


# ── Heuristic detector ────────────────────────────────────────────────────────

def _needs_decomposition(question: str) -> bool:
    """
    Fast check before making an API call.
    Returns True only if question clearly has multiple distinct parts.
    """
    q = question.lower().strip()

    # Short questions are almost never multi-part
    if len(q.split()) < 8:
        return False

    # Single-intent patterns → no decomposition
    for pattern in SINGLE_INTENT_PATTERNS:
        if re.match(pattern, q):
            return False

    # Check for decomposition triggers
    for trigger in DECOMPOSE_TRIGGERS:
        if re.search(trigger, q):
            return True

    # Count question marks — multiple ? = multiple questions
    if q.count("?") >= 2:
        return True

    return False


# ── LLM decomposer ────────────────────────────────────────────────────────────

def _decompose_with_llm(question: str) -> List[str]:
    """
    Use Groq to split question into focused sub-queries.
    Returns list of sub-query strings.
    """
    prompt = f"""You are a query decomposition assistant for a document Q&A system.

Original question: "{question}"

Split this into {MAX_SUB_QUERIES} or fewer focused, self-contained sub-questions.
Each sub-question should be retrievable independently from a document.

Rules:
- Each sub-question must be specific and self-contained
- Keep original intent — don't add new questions
- If the question is actually simple, return it as-is (just one line)
- Return ONLY the sub-questions, one per line, no numbering, no bullets
- Maximum {MAX_SUB_QUERIES} sub-questions

Sub-questions:"""

    try:
        raw = chat_text([{"role": "user", "content": prompt}], max_tokens=200, temperature=0.1)

        # Parse lines into sub-queries
        lines = [
            l.strip().lstrip("0123456789.-) ").strip()
            for l in raw.splitlines()
            if l.strip() and len(l.strip()) > 5
        ]

        # Deduplicate and limit
        seen = set()
        sub_queries = []
        for line in lines:
            if line.lower() not in seen and line:
                seen.add(line.lower())
                sub_queries.append(line)
            if len(sub_queries) >= MAX_SUB_QUERIES:
                break

        if not sub_queries:
            return [question]

        print(f"[DECOMPOSER] {len(sub_queries)} sub-queries: {sub_queries}")
        return sub_queries

    except Exception as e:
        print(f"[DECOMPOSER] LLM failed ({e}), using original")
        return [question]


# ── Public API ────────────────────────────────────────────────────────────────

def decompose_query(question: str) -> Tuple[List[str], bool]:
    """
    Main entry point.

    Args:
        question: User's question

    Returns:
        (sub_queries: list of strings, was_decomposed: bool)
        If not decomposed, returns ([question], False)
    """
    if not _needs_decomposition(question):
        print(f"[DECOMPOSER] Single query: '{question[:60]}'")
        return [question], False

    print(f"[DECOMPOSER] Decomposing: '{question[:60]}'")
    sub_queries = _decompose_with_llm(question)

    was_decomposed = len(sub_queries) > 1
    return sub_queries, was_decomposed


def merge_contexts(
    sub_queries: List[str],
    results_per_query: List[List[Dict]],
    was_decomposed: bool,
) -> List[Dict]:
    """
    Merge retrieved nodes from multiple sub-queries.
    Deduplicates by node text, preserves ordering by relevance.

    Args:
        sub_queries:        List of sub-query strings
        results_per_query:  List of node lists, one per sub-query
        was_decomposed:     Whether decomposition happened

    Returns:
        Merged deduplicated node list with sub-query labels in metadata
    """
    if not was_decomposed:
        return results_per_query[0] if results_per_query else []

    seen_texts = set()
    merged = []

    for sub_q, nodes in zip(sub_queries, results_per_query):
        for node in nodes:
            # Dedup by first 100 chars of text
            text_key = node.get("text", "")[:100].lower().strip()
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)

            # Tag node with which sub-query it came from
            node_copy = dict(node)
            node_copy["sub_query"] = sub_q
            merged.append(node_copy)

    print(f"[DECOMPOSER] Merged {len(merged)} unique nodes from {len(sub_queries)} sub-queries")
    return merged


def build_decomposed_context(
    sub_queries: List[str],
    results_per_query: List[List[Dict]],
) -> str:
    """
    Build a structured context string that groups retrieved text
    by sub-query, so the LLM can answer each part clearly.
    """
    parts = []

    for sub_q, nodes in zip(sub_queries, results_per_query):
        if not nodes:
            continue

        section = f"[Context for: {sub_q}]\n"
        for node in nodes[:3]:  # Top 3 per sub-query
            page = node.get("metadata", {}).get("page", "?")
            heading = node.get("metadata", {}).get("section_heading", "")
            header = f"[Page {page}" + (f" | {heading}" if heading else "") + "]"
            section += f"{header}\n{node.get('text', '')}\n\n"

        parts.append(section.strip())

    return "\n\n" + ("─" * 40) + "\n\n".join(parts)