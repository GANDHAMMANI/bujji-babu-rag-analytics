"""
query_classifier.py — Query Intent Classifier for Bujji Babu PDF RAG

Purpose:
  Different question types need different answer strategies.

  "Who wrote this?" → short direct answer, cite page
  "Compare X and Y" → structured comparison, use headers
  "Summarize this"  → bullet points, broad coverage
  "What does X mean?" → definition + explanation + example

  Without classification, every question gets the same generic prompt
  → answers are often the wrong format/length/style.

Classes:
  FACTOID      → direct short answer, cite page number
  COMPARATIVE  → structured side-by-side, use sections
  CAUSAL       → explain cause/effect chain
  DEFINITIONAL → define term, explain context
  ANALYTICAL   → deep reasoning, multi-paragraph
  SUMMARY      → bullet-point overview, broad coverage
  PROCEDURAL   → step-by-step explanation
  CONVERSATIONAL → casual, short, no citation needed

Each class gets:
  - A tailored system prompt
  - Expected answer length
  - Citation requirement
  - Format guidance
"""

import re
from typing import Dict, Tuple

# ── Pattern sets per class ────────────────────────────────────────────────────

_PATTERNS: Dict[str, list] = {
    "FACTOID": [
        r"^(who (is|are|was|were|wrote|created|developed|invented|founded))",
        r"^(when (was|is|did|were|did))",
        r"^(where (is|was|are|were|did))",
        r"^(what (year|date|number|name|title|version))",
        r"(how many|how much|what is the (number|count|total|size|value|accuracy|score|rate|percentage|ratio))",
        r"^(which (one|paper|model|method|year|author|chapter|section))",
        r"^what is the (accuracy|score|f1|precision|recall|loss|error|result|output|performance)",
    ],
    "COMPARATIVE": [
        r"(compare|contrast|difference between|similarities between)",
        r"(vs\.?|versus)",
        r"(how does .{1,40} differ from)",
        r"(better|worse|advantage|disadvantage|pros and cons)",
        r"(which is (better|more|less|faster|slower|stronger|weaker))",
    ],
    "CAUSAL": [
        r"^(why (did|does|is|are|was|were))",
        r"(what (caused|led to|resulted in|triggered|influenced))",
        r"(reason(s)? (for|behind|why))",
        r"(impact|effect|consequence|implication|outcome) of",
        r"(how did .{1,40} (lead to|cause|result in|affect))",
    ],
    "DEFINITIONAL": [
        r"^(what is|what are|define|definition of|meaning of)",
        r"^(explain (what|the concept|the term|the idea))",
        r"(what does .{1,30} mean)",
        r"(what (is meant|do you mean) by)",
        r"^(describe (the|a|an))",
        # "can you explain what X is" / "please explain what X means"
        r"(explain (what|how) .{1,40} (is|are|works?|means?|does))",
        r"^(can you (explain|describe|define|tell me what))",
    ],
    "ANALYTICAL": [
        r"(analyse|analyze|evaluate|assess|critique|examine|investigate)",
        r"(how (effective|efficient|accurate|reliable|valid|significant))",
        r"(what are the (implications|limitations|strengths|weaknesses))",
        r"(discuss|elaborate|explain (how|why|the relationship))",
        r"(methodology|approach|framework|architecture|design)",
    ],
    "SUMMARY": [
        r"(summarize|summarise|summary|overview|outline|synopsis)",
        r"(main (points|findings|contributions|ideas|results|themes))",
        r"(key (ideas|concepts|takeaways|insights|contributions|findings))",
        r"(tell me about|what is this (paper|document|book|chapter) about)",
        r"(what (happened|is discussed|is covered|is presented))",
        # "what algorithms/topics/chapters are covered in this book"
        r"(what .{0,40} (are|is) (covered|discussed|included|presented) in (this|the))",
        r"((covered|discussed|included|presented|addressed) in (this|the) (book|document|paper|chapter))",
        # "list all algorithms / give me all topics"
        r"(list (all|every|the) .{0,30}|give me (a list|all) .{0,30})",
    ],
    "PROCEDURAL": [
        r"(how (do|does|can|to|did) .{1,40} (work|function|operate|run|implement))",
        r"(steps? (to|for|in|of)|process of|procedure for)",
        r"(how (is|was|are|were) .{1,40} (done|implemented|built|trained|created))",
        r"(walk me through|guide me through|explain the process)",
    ],
    "CONVERSATIONAL": [
        # Only true greetings / one-word acks — NOT "can you explain X"
        r"^(hi|hello|hey|thanks|thank you|ok|okay|sure|yes|no|great|awesome|nice|cool|got it|understood)(\s|$|!|\?)",
        r"^(good (morning|afternoon|evening|day))",
        r"^(what do you think|do you have an opinion|your opinion)",
        r"^(is (it|this|that) (true|correct|right|accurate|possible))(\?|$)",
    ],
}

# ── Priority order (checked top to bottom, first match wins) ─────────────────

_PRIORITY = [
    "COMPARATIVE",   # "compare X and Y" — very specific, check first
    "CAUSAL",        # "why did / what caused"
    "PROCEDURAL",    # "how does X work / steps to"
    "SUMMARY",       # "tell me about / what is covered / overview"
    "FACTOID",       # "who/when/where/how many" — before DEFINITIONAL
    "ANALYTICAL",    # "analyse / evaluate / discuss"
    "DEFINITIONAL",  # "what is / define / explain what"
    "CONVERSATIONAL",# LAST — only true greetings/acks; prevents "can you explain..." misfire
]


# ── Classifier ────────────────────────────────────────────────────────────────

def _llm_classify(question: str) -> str:
    """LLM fallback when no regex pattern matches. One fast API call."""
    try:
        from llm_client import get_client, get_model
        classes = ", ".join(_PRIORITY)
        resp = get_client().chat.completions.create(
            model=get_model("fast"),
            messages=[
                {"role": "system", "content": (
                    f"Classify the user's question into exactly ONE of these categories: {classes}.\n"
                    "Rules:\n"
                    "FACTOID = specific fact, number, name, date\n"
                    "COMPARATIVE = comparing two or more things\n"
                    "CAUSAL = why something happened or its effects\n"
                    "DEFINITIONAL = what something means or is\n"
                    "ANALYTICAL = deep evaluation, critique, implications\n"
                    "SUMMARY = overview, main points, list of topics\n"
                    "PROCEDURAL = how something works, steps, process\n"
                    "CONVERSATIONAL = greeting, thanks, small talk\n"
                    "Reply with ONLY the category name, nothing else."
                )},
                {"role": "user", "content": question},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        cls = resp.choices[0].message.content.strip().upper()
        if cls in _PRIORITY:
            print(f"[CLASSIFIER] '{question[:50]}' → {cls} (LLM)")
            return cls
    except Exception as e:
        print(f"[CLASSIFIER] LLM fallback failed: {e}")
    return "ANALYTICAL"


def classify_query(question: str) -> str:
    """
    Classify question into one of 8 intent classes.
    Tries regex patterns first; falls back to a single LLM call for ambiguous queries.
    """
    q = question.lower().strip()

    for cls in _PRIORITY:
        for pattern in _PATTERNS[cls]:
            if re.search(pattern, q):
                print(f"[CLASSIFIER] '{question[:50]}' → {cls}")
                return cls

    # No regex matched — use LLM for accurate classification
    print(f"[CLASSIFIER] '{question[:50]}' → LLM fallback")
    return _llm_classify(question)


# ── Answer strategy per class ─────────────────────────────────────────────────

_STRATEGIES: Dict[str, Dict] = {
    "FACTOID": {
        "max_tokens":   600,
        "temperature":  0.1,
        "cite_pages":   True,
        "system": (
            "You are a precise document assistant. "
            "CRITICAL: Answer EXCLUSIVELY from the provided document context below. "
            "Do NOT use any external knowledge or training data. "
            "If the answer is not in the context, say 'The document does not mention this.' "
            "RULES:\n"
            "1. If the user asks for 'reviewer' and context has 'technical reviewer', answer directly.\n"
            "2. Never say 'not explicitly mentioned' if a related term exists in context.\n"
            "3. Always add context from the document: role, expertise, background.\n"
            "4. Cite the actual content page, not the Table of Contents page.\n"
            "5. Never truncate names. Never add facts not in the context."
        ),
    },
    "COMPARATIVE": {
        "max_tokens":   800,
        "temperature":  0.1,
        "cite_pages":   True,
        "system": (
            "You are a precise document assistant. "
            "CRITICAL: Answer EXCLUSIVELY from the provided document context. "
            "Do NOT use external knowledge or training data. "
            "Structure your comparison clearly — address each aspect for both subjects. "
            "Only compare what the document explicitly states. "
            "Cite page numbers where relevant."
        ),
    },
    "CAUSAL": {
        "max_tokens":   600,
        "temperature":  0.1,
        "cite_pages":   True,
        "system": (
            "You are a precise document assistant. "
            "CRITICAL: Answer EXCLUSIVELY from the provided document context. "
            "Do NOT use external knowledge or training data. "
            "Explain the cause-effect relationship using ONLY what the document states. "
            "Cite pages naturally. Do not speculate beyond the document."
        ),
    },
    "DEFINITIONAL": {
        "max_tokens":   400,
        "temperature":  0.1,
        "cite_pages":   True,
        "system": (
            "You are a precise document assistant. "
            "CRITICAL: Answer EXCLUSIVELY from the provided document context. "
            "Do NOT use external knowledge or training data. "
            "Provide the definition as it appears in the document. "
            "If the document gives an example, include it. "
            "Cite the page where the definition appears."
        ),
    },
    "ANALYTICAL": {
        "max_tokens":   1100,
        "temperature":  0.2,
        "cite_pages":   True,
        "system": (
            "You are a precise document analyst. "
            "CRITICAL: Base your analysis EXCLUSIVELY on the provided document context. "
            "Do NOT use external knowledge or training data. "
            "Every claim must be traceable to the document context. "
            "Cite pages naturally. Do not mention 'context', 'chunks', or 'retrieval'."
        ),
    },
    "SUMMARY": {
        "max_tokens":   1200,
        "temperature":  0.2,
        "cite_pages":   False,
        "system": (
            "You are a thorough document summariser. "
            "CRITICAL: Answer EXCLUSIVELY from the provided document context. "
            "Do NOT add information from external knowledge or training data. "
            "Be as comprehensive as possible — if the user asks for a list (algorithms, topics, "
            "chapters, methods, etc.), enumerate EVERY item found in the context, not just a few. "
            "Use bullet points for lists and short paragraphs for overviews. "
            "Group related items under clear headings where it helps readability. "
            "Do not mention 'context', 'chunks', or 'retrieval'."
        ),
    },
    "PROCEDURAL": {
        "max_tokens":   600,
        "temperature":  0.1,
        "cite_pages":   True,
        "system": (
            "You are a precise document assistant. "
            "CRITICAL: Answer EXCLUSIVELY from the provided document context. "
            "Do NOT use external knowledge or training data. "
            "Explain the process step by step using ONLY what the document states. "
            "Cite page numbers where relevant."
        ),
    },
    "CONVERSATIONAL": {
        "max_tokens":   200,
        "temperature":  0.3,
        "cite_pages":   False,
        "system": (
            "You are a friendly document assistant. "
            "Answer based on the document context provided. "
            "Keep it brief and direct. "
            "Do not add information beyond what is in the document."
        ),
    },
}


def get_answer_strategy(query_class: str) -> Dict:
    """
    Return the answer strategy dict for a given query class.
    Always returns a valid dict (falls back to ANALYTICAL).
    """
    return _STRATEGIES.get(query_class, _STRATEGIES["ANALYTICAL"])


def classify_and_strategise(question: str) -> Tuple[str, Dict]:
    """
    Convenience function — classify then return strategy.

    Returns:
        (query_class: str, strategy: Dict)
    """
    cls      = classify_query(question)
    strategy = get_answer_strategy(cls)
    return cls, strategy