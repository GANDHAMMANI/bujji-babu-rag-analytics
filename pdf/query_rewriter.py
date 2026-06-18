"""
query_rewriter.py — Contextual Query Rewriter for Bujji Babu PDF RAG

Problem it solves:
  User: "Who wrote this paper?"
  User: "What is his background?"   ← retriever searches "his background" — finds nothing
  
  With rewriting:
  "What is his background?" + history → "What is the background of [Author Name]?"
  Now retrieval works correctly.

How it works:
  1. Check if rewriting is needed (pronouns, references, follow-ups)
  2. If yes → single fast Groq call to rewrite using history
  3. If no  → return original question unchanged (saves API call)
"""

import re
from typing import List, Dict, Optional
from llm_client import chat_text, get_model

# ── Signals that a question references prior context ─────────────────────────

REFERENCE_PRONOUNS = [
    "he", "she", "they", "it", "his", "her", "their", "its",
    "him", "them", "this", "that", "these", "those",
]

REFERENCE_PHRASES = [
    "the same", "the above", "the previous", "the latter", "the former",
    "mentioned", "said", "discussed", "that one", "the one",
    "first one", "second one", "third one", "last one",
    "what about", "how about", "and the", "what else",
    "more about", "tell me more", "explain more", "elaborate",
    "go on", "continue", "and also", "also tell",
]

FOLLOWUP_STARTERS = [
    "what about", "how about", "and ", "but ", "also ",
    "so ", "then ", "now ", "why ", "when ", "where ",
    "who ", "which ", "can you", "could you",
]

# Questions that are self-contained and should never be rewritten
SELF_CONTAINED = [
    r"who (is|are|was) the author",
    r"who wrote",
    r"what is the title",
    r"when was (it|this|the book) published",
    r"what year",
    r"isbn",
    r"publisher",
    r"copyright",
]


def _needs_rewriting(question: str, history: List[Dict]) -> bool:
    if not history:
        return False

    q = question.lower().strip()

    # Check pronouns FIRST — these always need rewriting if history exists
    first_word = q.split()[0] if q.split() else ""
    if first_word in REFERENCE_PRONOUNS:
        return True
    if re.search(r"\b(it|this|that|he|she|they|his|her|their|him|them)\b", q):
        return True

    # Only THEN check if it's self-contained (no pronouns = no rewrite needed)
    for pattern in SELF_CONTAINED:
        if re.search(pattern, q):
            return False
    """
    Fast heuristic check — avoids unnecessary API call when question
    is clearly self-contained.
    Returns True if question likely references prior context.
    """
    if not history:
        return False

    q = question.lower().strip()

    # Very short questions almost always reference context
    if len(q.split()) <= 4:
        return True

    # Check for reference pronouns at start or anywhere
    first_word = q.split()[0]
    if first_word in REFERENCE_PRONOUNS:
        return True

    # Check for reference phrases
    if any(phrase in q for phrase in REFERENCE_PHRASES):
        return True

    # Check for follow-up starters
    if any(q.startswith(starter) for starter in FOLLOWUP_STARTERS):
        return True

    # Check for missing subject — question has no noun but has a verb
    # e.g. "Was it published in 2023?" — "it" refers to prior subject
    if re.search(r'\b(it|this|that|he|she|they)\b', q):
        return True

    return False


# Term expansions for ambiguous single-word queries
TERM_EXPANSIONS = {
    "reviewer":  "technical reviewer",
    "editor":    "technical editor",
    "author":    "author",
    "foreword":  "foreword writer",
    "preface":   "preface author",
    "publisher": "publisher",
}


def _expand_terms(question: str) -> str:
    """
    Expand ambiguous single terms to their full common forms.
    'who is the reviewer' → 'who is the reviewer or technical reviewer'
    """
    q_lower = question.lower()
    for short_term, full_term in TERM_EXPANSIONS.items():
        if short_term in q_lower and full_term not in q_lower:
            # Append expanded form so both are searched
            return question.rstrip("?").rstrip() + f" (also known as {full_term})?"
    return question


def rewrite_query(
    question: str,
    history: List[Dict],
    max_history_turns: int = 3,
) -> str:
    """
    Rewrite the question to be self-contained using conversation history.

    Args:
        question:           Current user question
        history:            List of {"role": "user"/"assistant", "content": "..."}
        max_history_turns:  How many prior turns to use (default 3)

    Returns:
        Rewritten question string, or original if no rewriting needed.
    """
    if not _needs_rewriting(question, history):
        expanded = _expand_terms(question)
        if expanded != question:
            print(f"[REWRITER] Term expanded: '{question}' → '{expanded}'")
        else:
            print(f"[REWRITER] No rewrite needed: '{question}'")
        return expanded

    # Take last N turns (user + assistant pairs)
    recent = history[-(max_history_turns * 2):]

    # Build compact history string
    history_str = ""
    for turn in recent:
        role = "User" if turn["role"] == "user" else "Assistant"
        # Truncate long assistant answers — we only need key entities
        content = turn["content"]
        if turn["role"] == "assistant" and len(content) > 300:
            content = content[:300] + "…"
        history_str += f"{role}: {content}\n"

    prompt = f"""You are a query rewriter for a document Q&A system.

Conversation so far:
{history_str.strip()}

New question: "{question}"

Task: Rewrite the new question to be fully self-contained and specific,
replacing all pronouns and vague references with the actual entities from
the conversation history.

Rules:
- If the question is already self-contained, return it unchanged
- Replace pronouns (he/she/it/they/this/that) with actual names/terms
- Keep the question concise — do not add unnecessary words
- Do not answer the question — only rewrite it
- Return ONLY the rewritten question, nothing else, no quotes

Rewritten question:"""

    try:
        resp = _groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.1,
        )
        rewritten = resp.choices[0].message.content.strip()

        # Clean up — remove quotes if model added them
        rewritten = rewritten.strip('"\'')

        # Sanity check — if rewritten is too different or too long, use original
        if not rewritten or len(rewritten) > len(question) * 3:
            print(f"[REWRITER] Rewrite rejected (too long/empty), using original")
            return question

        print(f"[REWRITER] '{question}' → '{rewritten}'")
        return rewritten

    except Exception as e:
        print(f"[REWRITER] Failed ({e}), using original question")
        return question