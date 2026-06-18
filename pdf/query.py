"""
query.py — Bujji Babu PDF Query Pipeline (v3)

Upgrades:
- ChromaDB replaces VectorStoreIndex for retrieval (persistent)
- PDF conversation memory: last N Q&A pairs included in LLM context
- Parallel vision scoring via ThreadPoolExecutor (3-4x faster)
- BM25 corpus cached per pdf_id (not rebuilt every query)
- Groq client instantiated once at module load
- Conversation history stored in SQLite
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Single shared Groq client
# LLM client managed by llm_client.py

from ingestion import get_metadata, get_all_nodes, query_chroma
from vision import rerank_images, should_show_table
from query_rewriter import rewrite_query
from confidence import gate_answer
from query_decomposer import decompose_query, merge_contexts, build_decomposed_context
from chunker import expand_to_parents
from faithfulness import score_faithfulness
from llm_client import chat, get_model, stream_chat, get_client
from adaptive_retrieval import adaptive_retrieve, detect_query_type
from query_classifier import classify_query, get_answer_strategy

# Appended to every non-CONVERSATIONAL system prompt so the LLM naturally
# invites the user to continue the conversation after each answer.
_CONV_TAIL = (
    "\n\nAfter answering, add one short natural sentence that invites the user "
    "to keep exploring — offer to go deeper on a specific item from your answer, "
    "or suggest a related aspect they might want to know about. "
    "Make it specific to what you just said (e.g. 'Want me to walk through how "
    "CNNs work in more detail?' not 'Let me know if you have more questions.'). "
    "Do NOT prefix it with 'Follow-up:' or any label — just write it naturally."
)

DB_PATH = Path("bujji.db")

# ── BM25 cache (rebuilt only when nodes change) ───────────────────────────────
_bm25_cache: Dict[str, Any] = {}


# ── Conversation history (SQLite-backed) ──────────────────────────────────────

def _init_conversation_table():
    if not DB_PATH.exists():
        return
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS pdf_conversations (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_id     TEXT NOT NULL,
            user_id    INTEGER NOT NULL,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    con.commit()
    con.close()

_init_conversation_table()


def _init_feedback_table():
    """Create pdf_feedback table for thumbs-up/down ratings."""
    if not DB_PATH.exists():
        return
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS pdf_feedback (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_id     TEXT NOT NULL,
            user_id    INTEGER NOT NULL,
            question   TEXT NOT NULL,
            answer     TEXT NOT NULL,
            rating     INTEGER NOT NULL,
            comment    TEXT DEFAULT '',
            created_at TEXT NOT NULL
        )
    """)
    con.commit()
    con.close()

_init_feedback_table()


def save_feedback(
    pdf_id: str,
    user_id: int,
    question: str,
    answer: str,
    rating: int,
    comment: str = "",
):
    """Persist a user thumbs-up (1) or thumbs-down (-1) rating."""
    if not DB_PATH.exists():
        return
    from datetime import datetime
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """INSERT INTO pdf_feedback
           (pdf_id, user_id, question, answer, rating, comment, created_at)
           VALUES (?,?,?,?,?,?,?)""",
        (pdf_id, user_id, question, answer, rating, comment,
         datetime.utcnow().isoformat()),
    )
    con.commit()
    con.close()
    print(f"[FEEDBACK] pdf={pdf_id} user={user_id} rating={rating}")


def _save_turn(pdf_id: str, user_id: int, question: str, answer: str):
    if not DB_PATH.exists():
        return
    from datetime import datetime
    con = sqlite3.connect(DB_PATH)
    now = datetime.utcnow().isoformat()
    con.executemany(
        "INSERT INTO pdf_conversations (pdf_id, user_id, role, content, created_at) VALUES (?,?,?,?,?)",
        [
            (pdf_id, user_id, "user",      question, now),
            (pdf_id, user_id, "assistant", answer,   now),
        ]
    )
    con.commit()
    con.close()


def get_conversation_history(pdf_id: str, user_id: int, last_n: int = 6) -> List[Dict]:
    """Return last_n turns as [{"role": ..., "content": ...}, ...]"""
    if not DB_PATH.exists():
        return []
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.execute(
            """SELECT role, content FROM pdf_conversations
               WHERE pdf_id=? AND user_id=?
               ORDER BY id DESC LIMIT ?""",
            (pdf_id, user_id, last_n * 2)
        )
        rows = cur.fetchall()
        con.close()
        # Rows are newest-first; reverse so oldest first
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
    except Exception:
        return []


def clear_conversation(pdf_id: str, user_id: int):
    if not DB_PATH.exists():
        return
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "DELETE FROM pdf_conversations WHERE pdf_id=? AND user_id=?",
        (pdf_id, user_id)
    )
    con.commit()
    con.close()


# ── HyDE ─────────────────────────────────────────────────────────────────────

def _hyde_expand(question: str, doc_context: str = "") -> str:
    context_hint = f"Document context: {doc_context}\n\n" if doc_context else ""
    try:
        resp = chat(
            messages=[
                {"role": "system", "content": ("You are a document excerpt generator. Write a short factual paragraph (3-5 sentences) that would appear in the specific document and directly answer the question. Use the document context provided. Write as if it IS the document excerpt — no preamble, no names invented.")},
                {"role": "user", "content": f"{context_hint}{question}"},
            ],
            max_tokens=200, temperature=0.3,
        )
        hypo = resp.choices[0].message.content.strip()
        print(f"[HyDE] {hypo[:80]}…")
        return hypo
    except Exception as e:
        print(f"[HyDE] Failed ({e}), using original question")
        return question


# ── BM25 (cached per pdf_id) ──────────────────────────────────────────────────

def _bm25_retrieve(pdf_id: str, nodes: List, query: str, top_k: int = 6) -> List[Dict]:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("[BM25] rank_bm25 not installed, skipping")
        return []

    try:
        text_nodes = [n for n in nodes if n.metadata.get("source_type", "text") == "text"]
        if not text_nodes:
            return []

        # Rebuild corpus only if not cached
        if pdf_id not in _bm25_cache:
            print(f"[BM25] Building corpus for {pdf_id} ({len(text_nodes)} nodes)…")
            corpus = [n.text.lower().split() for n in text_nodes]
            _bm25_cache[pdf_id] = {"bm25": BM25Okapi(corpus), "nodes": text_nodes}

        cached    = _bm25_cache[pdf_id]
        bm25_obj  = cached["bm25"]
        c_nodes   = cached["nodes"]

        scores = bm25_obj.get_scores(query.lower().split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = [{"node": c_nodes[i], "score": float(s)} for i, s in ranked if s > 0]
        print(f"[BM25] {len(results)} nodes")
        return results
    except Exception as e:
        print(f"[BM25] Failed: {e}")
        return []


# ── RRF merge ─────────────────────────────────────────────────────────────────

def _rrf_merge(chroma_nodes: List[Dict], bm25_nodes: List[Dict], k: int = 60, top_n: int = 8) -> List[Dict]:
    scores: Dict[str, float] = {}
    node_map: Dict[str, Dict] = {}

    for rank, item in enumerate(chroma_nodes):
        nid = item.get("id") or item.get("text", "")[:40]
        scores[nid]   = scores.get(nid, 0.0) + 1.0 / (k + rank + 1)
        node_map[nid] = item

    for rank, item in enumerate(bm25_nodes):
        node  = item["node"]
        nid   = node.node_id
        score = item["score"]
        if nid not in node_map:
            node_map[nid] = {
                "text":     node.text,
                "metadata": node.metadata,
                "score":    score,
            }
        scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank + 1)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    result = [node_map[nid] for nid, _ in fused if nid in node_map]
    print(f"[RRF] {len(result)} merged nodes")
    return result


# ── Cross-encoder rerank ──────────────────────────────────────────────────────

def _cross_encoder_rerank(query: str, nodes: List[Dict], top_n: int = 5) -> List[Dict]:
    try:
        from sentence_transformers import CrossEncoder
        model  = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        pairs  = [(query, n["text"][:500]) for n in nodes]
        scores = model.predict(pairs)
        ranked = sorted(zip(nodes, scores), key=lambda x: x[1], reverse=True)[:top_n]
        # Drop nodes scoring below 30th percentile to improve context precision
        if len(ranked) > 3:
            score_vals = [s for _, s in ranked]
            p30 = sorted(score_vals)[max(0, len(score_vals)//3)]
            ranked = [(n, s) for n, s in ranked if s >= p30]
        result = [n for n, _ in ranked]
        print(f"[CrossEncoder] Top {len(result)} nodes (precision cutoff applied)")
        return result
    except Exception as e:
        print(f"[CrossEncoder] Skipped ({e})")
        return nodes[:top_n]


# ── Parallel vision scoring ───────────────────────────────────────────────────

def _parallel_rerank_images(
    candidate_images: List,
    question: str,
    answer: str,
    threshold: float = 0.40,
) -> List:
    """Score all candidate images in parallel using ThreadPoolExecutor."""
    if not candidate_images:
        return []

    from vision import _describe_and_score_image

    # Lower threshold for person/author queries — headshots may score 0.45-0.6
    q_lower = question.lower()
    if any(kw in q_lower for kw in ["author","wrote","reviewer","who is","person","photo"]):
        threshold = 0.25
    THRESHOLD = threshold
    scored    = []

    def _score_one(img):
        img_path = img.get("path", "")
        if not img_path or not os.path.exists(img_path):
            return None

        # Build enriched context: caption + page heading + answer
        image_caption = img.get("combined") or img.get("caption") or ""

        # Penalty: if caption mentions a DIFFERENT role than asked
        # e.g. question asks about author but caption says "Technical Reviewer"
        q_lower = question.lower()
        cap_lower = image_caption.lower()
        role_mismatch = False
        if "reviewer" in cap_lower and "author" in q_lower and "reviewer" not in q_lower:
            role_mismatch = True
        if "author" in cap_lower and "reviewer" in q_lower and "author" not in q_lower:
            role_mismatch = True

        result = _describe_and_score_image(img_path, question, answer, image_caption=image_caption)
        score = result["score"]

        # Apply penalty for role mismatch
        if role_mismatch:
            score = score * 0.3
            print(f"[VISION] Role mismatch penalty applied: {img.get('filename')} score {result['score']:.2f} → {score:.2f}")

        # For person queries: only keep images whose description mentions a person
        q_lower = question.lower()
        is_person_q = any(kw in q_lower for kw in ["author","who is","reviewer","wrote","editor"])
        if is_person_q:
            desc_lower = result["description"].lower()
            has_person = any(w in desc_lower for w in ["headshot","portrait","photo","person","author","reviewer","man","woman","face"])
            if not has_person:
                print(f"[RERANK] Filtered non-person image: {img.get('filename')}")
                return None

        if score >= THRESHOLD:
            return {**img, "score": score, "description": result["description"]}
        return None

    # Run all scoring calls in parallel (I/O-bound: base64 encode + HTTP)
    max_workers = min(len(candidate_images), 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_score_one, img): img for img in candidate_images}
        for future in as_completed(futures):
            try:
                res = future.result()
                if res:
                    scored.append(res)
            except Exception as e:
                print(f"[VISION] Worker failed: {e}")

    scored.sort(key=lambda x: x["score"], reverse=True)
    print(f"[VISION] Parallel rerank: {len(scored)}/{len(candidate_images)} passed threshold")
    return scored


# ── Main query function ───────────────────────────────────────────────────────

def query_pdf(
    pdf_id: str,
    question: str,
    top_k: int = 6,
    user_id: int = 0,
) -> Dict[str, Any]:
    """
    Full RAG pipeline:
    ConversationHistory → QueryRewrite → HyDE → ChromaDB + BM25 → RRF → CrossEncoder
    → Generate answer (with memory) → Parallel vision rerank → Assemble
    """
    index = get_index_shim(pdf_id)
    if index is None:
        return {"error": "PDF not found. Please upload and ingest a PDF first."}

    meta      = get_metadata(pdf_id)
    all_nodes = get_all_nodes(pdf_id)
    print(f"[QUERY] Metadata keys: {list(meta.keys())}")
    print(f"[QUERY] Images in metadata: {len(meta.get('images', []))}")

    print(f"\n[QUERY] '{question}' (pdf={pdf_id}, user={user_id})")

    # ── 1. Load conversation history ──────────────────────────────────────────
    history = get_conversation_history(pdf_id, user_id, last_n=4)
    print(f"[QUERY] Loaded {len(history)} history turns")

    # ── 2. Contextual query rewriting ─────────────────────────────────────────
    # Rewrites vague follow-ups like "what about him?" → "what about [Author]?"
    # Uses original question for LLM answer generation (more natural)
    # Uses rewritten question for retrieval (more accurate)
    rewritten = rewrite_query(question, history, max_history_turns=3)

    # ── 3. Query decomposition ───────────────────────────────────────────────
    sub_queries, was_decomposed = decompose_query(rewritten)

    # ── 4. Query type detection for adaptive budget ───────────────────────────
    query_type = detect_query_type(rewritten)
    print(f"[QUERY] type={query_type} decomposed={was_decomposed}")

    # ── 5. Adaptive retrieval ─────────────────────────────────────────────────
    # Starts with small k, expands if confidence is low
    # Budget: factoid(3→5), general(4→6), analytical(5→12), summary(6→14)
    merged, all_chroma_results, adaptive_confidence, retrieval_log = adaptive_retrieve(
        pdf_id        = pdf_id,
        sub_queries   = sub_queries,
        query_type    = query_type,
        retrieval_fn  = query_chroma,
        bm25_fn       = _bm25_retrieve,
        rrf_fn        = _rrf_merge,
        confidence_fn = gate_answer,
        all_nodes     = all_nodes,
        primary_question = rewritten,
    )
    print(f"[ADAPTIVE] {retrieval_log}")

    # For decomposed queries, rebuild per-query results for context building
    if was_decomposed:
        results_per_query = []
        chunk = max(1, len(merged) // len(sub_queries))
        for i in range(len(sub_queries)):
            results_per_query.append(merged[i*chunk:(i+1)*chunk])
    else:
        results_per_query = [merged]

    if not merged:
        return {
            "answer":  "I couldn't find relevant information in this document.",
            "sources": [], "images": [], "tables": [],
        }

    # ── 6. Confidence gate ────────────────────────────────────────────────────
    chroma_scores = [n.get("score", 0.0) for n in all_chroma_results]
    gate = gate_answer(rewritten, merged, chroma_scores)

    if not gate["should_answer"]:
        return {
            "answer":     gate["fallback"],
            "sources":    [],
            "images":     [],
            "tables":     [],
            "confidence": gate["confidence"],
        }

    # ── 6b. Suppress low-confidence warning for sparse-chunk query types ──────
    # FACTOID questions (who/when/where/how many) often hit copyright pages,
    # title pages, or figure captions — chunks with very few words that score
    # low in embedding space even when they contain the exact right answer.
    # Showing the "loosely related content" warning here is a false positive.
    # Also suppress for CONVERSATIONAL (greetings never need a confidence caveat).
    _early_class = classify_query(question)
    _NO_WARN_CLASSES = {"FACTOID", "CONVERSATIONAL", "DEFINITIONAL"}
    if _early_class in _NO_WARN_CLASSES and gate.get("warning"):
        gate["warning"] = None
        print(f"[GATE] Warning suppressed for {_early_class} query (sparse-chunk false positive)")

    # ── 7. Expand child nodes to parents for richer generation context ─────────
    # Retrieval found precise child chunks — now fetch their parent chunks
    # which have 4x more surrounding context for better answer generation
    meta_cache  = get_metadata(pdf_id)
    c2p         = meta_cache.get("child_to_parent", {})
    merged      = expand_to_parents(pdf_id, merged, child_to_parent=c2p)

    # ── 8. Separate by source type ────────────────────────────────────────────
    text_nodes_raw = []
    table_nodes_raw = []
    retrieved_pages = set()

    for item in merged:
        src  = item.get("metadata", {}).get("source_type", "text")
        page = item.get("metadata", {}).get("page")
        if page:
            retrieved_pages.add(int(page))
        if src == "table":
            table_nodes_raw.append(item)
        elif src != "image":
            text_nodes_raw.append(item)

    # ── 9. Cross-encoder rerank ───────────────────────────────────────────────
    text_nodes_raw = _cross_encoder_rerank(question, text_nodes_raw, top_n=5)

    # ── 10. Build context ──────────────────────────────────────────────────────
    if was_decomposed:
        # Use structured context grouped by sub-query for multi-part questions
        context = build_decomposed_context(sub_queries, results_per_query)
    else:
        context_parts = []
        for item in text_nodes_raw:
            page    = item.get("metadata", {}).get("page", "?")
            heading = item.get("metadata", {}).get("section_heading", "")
            header  = f"[Page {page}" + (f" | {heading}" if heading else "") + "]"
            context_parts.append(f"{header}\n{item['text']}")

        for item in table_nodes_raw:
            page = item.get("metadata", {}).get("page", "?")
            md   = item.get("metadata", {}).get("table_markdown", item["text"])
            context_parts.append(f"[Table on Page {page}]\n{md}")

        context = "\n\n---\n\n".join(context_parts) if context_parts else "No context found."

    # ── 11. Classify query + get answer strategy ─────────────────────────────
    # Each query class gets tailored: system prompt, max_tokens, temperature
    # Reuse the early classification from step 6b (avoids a second regex pass)
    query_class = _early_class
    strategy    = get_answer_strategy(query_class)
    print(f"[CLASSIFIER] class={query_class} max_tokens={strategy['max_tokens']}")

    # ── 12. Build messages with conversation memory ────────────────────────
    if was_decomposed:
        system_prompt = (
            "You are a helpful assistant answering multi-part questions based on PDF document content. "
            "The context is grouped by sub-question — answer each part clearly and completely. "
            "Structure your answer to address all parts of the question. "
            "Mention page numbers naturally where relevant. "
            "Do not mention 'context', 'chunks', or 'retrieval'."
        )
    else:
        system_prompt = strategy["system"]

    # Append conversational follow-up invitation to all substantive query types.
    # CONVERSATIONAL queries (greetings/acks) are kept short and don't need it.
    if query_class != "CONVERSATIONAL":
        system_prompt += _CONV_TAIL

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({
        "role": "user",
        "content": f"Document context:\n\n{context}\n\nQuestion: {question}",
    })

    # ── 13. Generate answer with class-specific settings ──────────────────
    llm_resp = chat(
        messages=messages,
        max_tokens=strategy["max_tokens"],
        temperature=strategy["temperature"],
    )
    answer = llm_resp.choices[0].message.content.strip()

    # ── Post-process: remove TOC page citations ───────────────────────────────
    # If answer cites multiple pages and one is from TOC (usually page 4-6),
    # the LLM is mixing TOC reference with actual content page
    # Remove "Page 4" style citations when better pages are also cited
    import re as _re
    cited_pages = _re.findall(r'[Pp]age\s+(\d+|[ivxlcdmIVXLCDM]+)', answer)
    if len(cited_pages) > 1:
        # If page 4 cited alongside other pages, likely a TOC artifact
        toc_pages = {"4", "5", "6", "3"}
        non_toc = [p for p in cited_pages if p not in toc_pages]
        if non_toc and any(p in toc_pages for p in cited_pages):
            # Remove TOC page references from answer
            for tp in toc_pages:
                answer = _re.sub(
                    rf'\(?\s*[Pp]age\s+{tp}\s*(?:and\s+)?\.?\s*\)?',
                    '', answer
                ).strip()
            answer = _re.sub(r'\s{2,}', ' ', answer).strip()
            answer = _re.sub(r'\(\s*\)', '', answer).strip()
            print(f"[QUERY] Removed TOC page citations, kept: {non_toc}")

    print(f"[QUERY] Answer: {len(answer)} chars (class={query_class})")

    # ── 14. Faithfulness scoring ──────────────────────────────────────────────
    # Fast heuristic by default — scores each sentence against retrieved context
    # deep=True uses Groq NLI (more accurate, costs one extra API call)
    # Enable deep scoring for query classes where hallucination risk is highest:
    #   ANALYTICAL (multi-paragraph reasoning), SUMMARY (broad coverage),
    #   COMPARATIVE (side-by-side claims) — these benefit most from NLI verification
    _deep_scoring_classes = {"ANALYTICAL", "SUMMARY", "COMPARATIVE"}
    _use_deep = query_class in _deep_scoring_classes
    faithfulness = score_faithfulness(answer, context, deep=_use_deep)
    if _use_deep:
        print(f"[FAITHFULNESS] Deep NLI mode for query_class={query_class}")

    # ── 15. Persist this turn to SQLite ───────────────────────────────────────
    if user_id:
        _save_turn(pdf_id, user_id, question, answer)

   # ── 16. Parallel vision rerank ────────────────────────────────────────────
    import os as _os
    from llm_client import get_mode as _get_mode
    if _os.environ.get("SKIP_VISION") == "1":
        print("[VISION] Skipped (SKIP_VISION=1 eval mode)")
        reranked_images, images_out = [], []
        all_images = []
    else:
        all_images = meta.get("images", [])
    total_pages = max((int(img.get("page", 0)) for img in all_images), default=0)
    expanded_pages = set()
    for p in retrieved_pages:
        p = int(p)
        expanded_pages.update([p - 2, p - 1, p, p + 1, p + 2])

    # For author/person queries: search front matter + last pages + wide window
    if query_class in ("FACTOID",) and any(
        kw in question.lower() for kw in ["author","wrote","reviewer","editor","who is","about"]
    ):
        expanded_pages.update(range(1, 21))   # first 20 pages (covers all front matter)
        if total_pages > 0:
            expanded_pages.update(range(max(1, total_pages - 10), total_pages + 1))  # last 10 pages
        # Also widen the retrieved page window to ±8 instead of ±2
        for p in list(retrieved_pages):
            p = int(p)
            expanded_pages.update(range(max(1, p - 8), p + 9))
        print(f"[QUERY] Person query: expanded to front/back + wide window (total={total_pages})")

    # Cast page to int — stored metadata may have string page numbers
    candidate_images = []
    for img in all_images:
        if int(img.get("page", -1)) not in expanded_pages:
            continue
        # For person queries, skip images with non-portrait captions
        if any(kw in question.lower() for kw in ["author","reviewer","who is","wrote"]):
            caption = (img.get("combined") or img.get("caption") or "").lower()
            if any(skip in caption for skip in ["figure","table","diagram","topics","chapter","contents"]):
                continue
        candidate_images.append(img)
        # Limit to 3 in offline mode (VLM is slow on CPU), 5 in online
        from llm_client import get_mode
        _limit = 3 if get_mode() == "offline" else 5
        if len(candidate_images) >= _limit:
            break
    print(f"[QUERY] Vision candidates: {len(candidate_images)} from {len(all_images)} total images")
    print(f"[QUERY] Retrieved pages: {sorted(retrieved_pages)}, expanded: {sorted(list(expanded_pages))[:15]}")

    # Sort candidates: pages closest to retrieved content first
    # This ensures author photo (p12) scores before reviewer photo (p13)
    # when query is about the author
    if retrieved_pages:
        min_retrieved = min(int(p) for p in retrieved_pages)
        candidate_images.sort(key=lambda img: abs(int(img.get("page", 999)) - min_retrieved))

    reranked_images = _parallel_rerank_images(candidate_images, question, answer)[:3]

    # ── 17. Table decision ────────────────────────────────────────────────────
    tables_out = []
    for item in table_nodes_raw:
        table_md = item.get("metadata", {}).get("table_markdown", item["text"])
        decision = should_show_table(table_md, question)
        tables_out.append({
            "markdown": table_md,
            "summary":  decision.get("summary"),
            "action":   decision.get("action", "show"),
            "page":     item.get("metadata", {}).get("page", "?"),
        })

    # ── 18. Assemble ──────────────────────────────────────────────────────────
    images_out = [{
        "path":        img.get("path", "").replace("\\", "/"),
        "page":        img.get("page", "?"),
        "description": img.get("description", ""),
        "filename":    img.get("filename", ""),
        "score":       round(img.get("score", 0.0), 2),
    } for img in reranked_images]

    return {
        "answer":        answer,
        "sources":       sorted(retrieved_pages),
        "images":        images_out,
        "tables":        tables_out,
        "confidence":    gate["confidence"],
        "warning":       gate.get("warning"),
        "faithfulness":  faithfulness,
        "query_type":    query_type,
        "query_class":   query_class,
        "retrieval_log": retrieval_log,
    }


# ── Shim for get_index (used by main.py) ──────────────────────────────────────

def get_index_shim(pdf_id: str):
    from ingestion import get_index
    return get_index(pdf_id)