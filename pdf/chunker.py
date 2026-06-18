"""
chunker.py — Hierarchical Chunking for Bujji Babu PDF RAG

Problem with flat chunking:
  SemanticSplitter creates chunks of ~300-500 tokens.
  Retrieve top-5 → you get 5 isolated paragraphs with no surrounding context.
  The answer is often in the context AROUND the retrieved chunk, not in it.

Hierarchical strategy (parent/child):
  - CHILD chunks: small (128 tokens) — used for precise retrieval
  - PARENT chunks: large (512 tokens) — returned as context for generation

  Retrieval:  embed + search child chunks (more precise matches)
  Generation: expand matched children to their parent (more context)

  Result: precision of small chunks + context richness of large chunks.

Three levels:
  Level 0 — Document section  (~1500 tokens) — broad context
  Level 1 — Parent chunk      (~512 tokens)  — generation context
  Level 2 — Child chunk       (~128 tokens)  — retrieval unit

Storage in ChromaDB:
  Each child node stores its parent_id in metadata.
  query.py retrieves children, then fetches their parents for generation.
"""

from typing import List, Dict, Optional, Tuple
from llama_index.core import Document
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
import hashlib
import re


# ── Chunk sizes ───────────────────────────────────────────────────────────────

CHILD_CHUNK_SIZE   = 128    # tokens — retrieval unit
CHILD_CHUNK_OVERLAP = 16
PARENT_CHUNK_SIZE  = 512    # tokens — generation context
PARENT_CHUNK_OVERLAP = 64
SECTION_CHUNK_SIZE = 1500   # tokens — broad document section


# ── Splitters ─────────────────────────────────────────────────────────────────

def _make_splitters():
    child_splitter = SentenceSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
    )
    parent_splitter = SentenceSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
    )
    section_splitter = SentenceSplitter(
        chunk_size=SECTION_CHUNK_SIZE,
        chunk_overlap=128,
    )
    return child_splitter, parent_splitter, section_splitter


# ── Node ID helpers ───────────────────────────────────────────────────────────

def _make_id(text: str, prefix: str = "") -> str:
    h = hashlib.md5(text[:200].encode()).hexdigest()[:12]
    return f"{prefix}_{h}" if prefix else h


# ── Main hierarchical builder ─────────────────────────────────────────────────

def build_hierarchical_nodes(
    text_docs: List[Document],
) -> Tuple[List[TextNode], List[TextNode], Dict[str, str]]:
    """
    Build parent and child nodes from documents.

    Returns:
        child_nodes:      List[TextNode]  — embed these in ChromaDB
        parent_nodes:     List[TextNode]  — store separately, fetch by ID
        child_to_parent:  Dict[str, str]  — child_id → parent_id mapping
    """
    child_splitter, parent_splitter, _ = _make_splitters()

    child_nodes:     List[TextNode] = []
    parent_nodes:    List[TextNode] = []
    child_to_parent: Dict[str, str] = {}

    for doc in text_docs:
        doc_text = doc.text
        doc_meta = doc.metadata or {}
        page     = doc_meta.get("page", "?")

        # ── Split into parents first ──────────────────────────────────────────
        parent_docs = parent_splitter.get_nodes_from_documents([doc])

        for p_idx, p_node in enumerate(parent_docs):
            p_text = p_node.text
            if not p_text.strip():
                continue

            p_id = _make_id(p_text, f"parent_p{page}_{p_idx}")
            parent_meta = {
                **doc_meta,
                "node_level":    "parent",
                "parent_index":  p_idx,
                "source_type":   "text",
            }

            parent_node          = TextNode(text=p_text, metadata=parent_meta)
            parent_node.id_  = p_id
            parent_nodes.append(parent_node)

            # ── Split each parent into children ───────────────────────────────
            p_doc        = Document(text=p_text, metadata=parent_meta)
            child_docs   = child_splitter.get_nodes_from_documents([p_doc])

            for c_idx, c_node in enumerate(child_docs):
                c_text = c_node.text
                if not c_text.strip():
                    continue

                c_id = _make_id(c_text, f"child_p{page}_{p_idx}_{c_idx}")
                child_meta = {
                    **doc_meta,
                    "node_level":   "child",
                    "parent_id":    p_id,
                    "child_index":  c_idx,
                    "source_type":  "text",
                }

                child_node          = TextNode(text=c_text, metadata=child_meta)
                child_node.id_   = c_id
                child_nodes.append(child_node)
                child_to_parent[c_id] = p_id

    print(
        f"[CHUNKER] Built {len(child_nodes)} child nodes "
        f"from {len(parent_nodes)} parents "
        f"across {len(text_docs)} pages"
    )
    return child_nodes, parent_nodes, child_to_parent


# ── Parent store (in-memory per pdf_id, also persisted via ChromaDB) ─────────

_parent_store: Dict[str, Dict[str, TextNode]] = {}
# Format: { pdf_id: { parent_id: TextNode } }


def store_parents(pdf_id: str, parent_nodes: List[TextNode]) -> None:
    """Store parent nodes in memory for fast lookup during query time."""
    _parent_store[pdf_id] = {n.node_id: n for n in parent_nodes}
    print(f"[CHUNKER] Stored {len(parent_nodes)} parent nodes for {pdf_id}")


def get_parent(pdf_id: str, parent_id: str) -> Optional[TextNode]:
    """Retrieve a parent node by ID."""
    return _parent_store.get(pdf_id, {}).get(parent_id)


def expand_to_parents(
    pdf_id: str,
    child_nodes: List[Dict],
    child_to_parent: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """
    Given retrieved child nodes, expand each to its parent node.
    Deduplicates parents (multiple children from same parent → one parent).

    Args:
        pdf_id:           PDF identifier
        child_nodes:      List of retrieved node dicts (from ChromaDB)
        child_to_parent:  Optional mapping override

    Returns:
        List of parent node dicts with richer context text.
        Falls back to child if parent not found.
    """
    seen_parents = set()
    expanded     = []

    for node in child_nodes:
        node_id   = node.get("id", "")
        meta      = node.get("metadata", {})
        parent_id = meta.get("parent_id") or (
            child_to_parent.get(node_id) if child_to_parent else None
        )

        if not parent_id:
            # No parent — node is already a flat chunk, use as-is
            expanded.append(node)
            continue

        if parent_id in seen_parents:
            # Already included this parent from another child
            continue
        seen_parents.add(parent_id)

        parent_node = get_parent(pdf_id, parent_id)
        if parent_node:
            expanded.append({
                "text":     parent_node.text,
                "metadata": parent_node.metadata,
                "score":    node.get("score", 0.0),
                "id":       parent_id,
            })
        else:
            # Parent not in memory (e.g. after restart) — use child
            expanded.append(node)

    print(f"[CHUNKER] Expanded {len(child_nodes)} children → {len(expanded)} parents")
    return expanded


# ── Persist parent store to SQLite for restart recovery ──────────────────────

def persist_parents(pdf_id: str, parent_nodes: List[TextNode]) -> None:
    """Save parent node texts to SQLite so they survive restarts."""
    import sqlite3, json
    from pathlib import Path

    db_path = Path("bujji.db")
    if not db_path.exists():
        return

    try:
        con = sqlite3.connect(db_path)
        con.execute("""
            CREATE TABLE IF NOT EXISTS parent_nodes (
                pdf_id    TEXT NOT NULL,
                node_id   TEXT NOT NULL,
                text      TEXT NOT NULL,
                metadata  TEXT NOT NULL,
                PRIMARY KEY (pdf_id, node_id)
            )
        """)
        rows = [
            (pdf_id, n.node_id, n.text, json.dumps(n.metadata or {}))
            for n in parent_nodes
        ]
        con.executemany(
            "INSERT OR REPLACE INTO parent_nodes (pdf_id, node_id, text, metadata) VALUES (?,?,?,?)",
            rows
        )
        con.commit()
        con.close()
        print(f"[CHUNKER] Persisted {len(rows)} parent nodes to SQLite")
    except Exception as e:
        print(f"[CHUNKER] Persist failed: {e}")


def load_parents_from_db(pdf_id: str) -> None:
    """Reload parent nodes from SQLite into memory after a restart."""
    import sqlite3, json
    from pathlib import Path

    db_path = Path("bujji.db")
    if not db_path.exists():
        return

    try:
        con = sqlite3.connect(db_path)
        cur = con.execute(
            "SELECT node_id, text, metadata FROM parent_nodes WHERE pdf_id = ?",
            (pdf_id,)
        )
        rows = cur.fetchall()
        con.close()

        if not rows:
            return

        nodes = {}
        for node_id, text, meta_str in rows:
            try:
                node = TextNode(text=text, id_=str(node_id), metadata=json.loads(meta_str))
            except TypeError:
                node = TextNode(text=text, metadata=json.loads(meta_str))
                object.__setattr__(node, "id_", str(node_id))
            nodes[str(node_id)] = node

        _parent_store[pdf_id] = nodes
        print(f"[CHUNKER] Loaded {len(nodes)} parent nodes from SQLite for {pdf_id}")

    except Exception as e:
        print(f"[CHUNKER] Load from DB failed: {e}")