"""
ingestion.py — Bujji Babu PDF Ingestion (v3)

Upgrades:
- ChromaDB replaces in-memory VectorStoreIndex → survives server restarts
- Nodes + metadata persisted to SQLite via auth.py helpers
- Image deduplication by file hash
- Parallel vision scoring via ThreadPoolExecutor
- Upload size limit enforced here (50MB)
- Background-task friendly: ingest_pdf() is now a pure function, called from FastAPI BackgroundTask
"""

import os
import json
import hashlib
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
from image_caption import extract_all_captions
from chunker import build_hierarchical_nodes, store_parents, persist_parents, load_parents_from_db, expand_to_parents
from dotenv import load_dotenv

load_dotenv()

# ── ChromaDB ──────────────────────────────────────────────────────────────────
import chromadb
from chromadb.config import Settings as ChromaSettings

CHROMA_DIR = Path("chroma_db")
CHROMA_DIR.mkdir(exist_ok=True)

_chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=ChromaSettings(anonymized_telemetry=False),
)

# ── Embeddings ────────────────────────────────────────────────────────────────
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

EMBED_DIM = 384  # bge-small-en-v1.5 output dimension

# ── Groq LLM ──────────────────────────────────────────────────────────────────
from llama_index.llms.groq import Groq as GroqLLM
Settings.llm = GroqLLM(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
)

EXTRACTED_DIR = Path("extracted")
EXTRACTED_DIR.mkdir(exist_ok=True)

MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB hard limit

# ── In-memory cache (for fast repeated queries within a session) ──────────────
# Format: { pdf_id: { "nodes": [...], "metadata": {...} } }
_cache: Dict[str, Any] = {}


# ── ID ────────────────────────────────────────────────────────────────────────

def _pdf_id(pdf_path: str) -> str:
    return hashlib.md5(pdf_path.encode()).hexdigest()[:10]


# ── ChromaDB collection per PDF ───────────────────────────────────────────────

def _get_or_create_collection(pdf_id: str):
    """Each PDF gets its own ChromaDB collection named pdf_{pdf_id}."""
    return _chroma_client.get_or_create_collection(
        name=f"pdf_{pdf_id}",
        metadata={"hnsw:space": "cosine"},
    )


def _collection_exists(pdf_id: str) -> bool:
    try:
        col = _chroma_client.get_collection(f"pdf_{pdf_id}")
        return col.count() > 0
    except Exception:
        return False


# ── OCR ───────────────────────────────────────────────────────────────────────

def _ocr_page(page) -> str:
    try:
        import pytesseract
        from PIL import Image
        import io
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img)
    except Exception as e:
        print(f"[OCR] Failed: {e}")
        return ""


def _needs_ocr(text: str) -> bool:
    if not text:
        return True
    return (text.count("\ufffd") / max(len(text), 1)) > 0.40


# ── Image extraction with deduplication ───────────────────────────────────────

def _image_hash(img_bytes: bytes) -> str:
    return hashlib.md5(img_bytes).hexdigest()


def _extract_images_and_tables(pdf_path: str, pdf_id: str) -> Dict[str, List]:
    doc = fitz.open(pdf_path)
    images = []
    tables = []
    seen_hashes = set()  # deduplication

    img_dir = EXTRACTED_DIR / pdf_id / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for page_num, page in enumerate(doc):
        # ── Images ──
        for img_index, img_ref in enumerate(page.get_images(full=True)):
            xref = img_ref[0]
            try:
                base_img  = doc.extract_image(xref)
                img_bytes = base_img["image"]

                # Skip tiny images (likely decorative icons, bullets)
                if len(img_bytes) < 4096:  # < 4KB
                    continue

                # Deduplication by content hash
                img_hash = _image_hash(img_bytes)
                if img_hash in seen_hashes:
                    print(f"[IMG] Skipping duplicate image on page {page_num+1}")
                    continue
                seen_hashes.add(img_hash)

                ext = base_img["ext"]
                # Skip SVG and other non-raster formats
                if ext not in ("png", "jpg", "jpeg", "webp", "bmp"):
                    continue

                img_filename = f"page{page_num+1}_img{img_index+1}.{ext}"
                img_path = str(img_dir / img_filename)
                with open(img_path, "wb") as f:
                    f.write(img_bytes)

                images.append({
                    "path":     img_path.replace("\\", "/"),
                    "page":     int(page_num + 1),  # always int
                    "filename": img_filename,
                    "hash":     img_hash,
                    "size":     len(img_bytes),
                })
            except Exception:
                pass

        # ── Tables ──
        try:
            table_finder = page.find_tables()
            for t_index, table in enumerate(table_finder.tables):
                df = table.to_pandas()
                if df.empty:
                    continue
                md = df.to_markdown(index=False)
                tables.append({
                    "markdown": md,
                    "page":     page_num + 1,
                    "index":    t_index,
                })
        except Exception:
            pass

    doc.close()
    print(f"[IMG] Extracted {len(images)} unique images, {len(tables)} tables")
    return {"images": images, "tables": tables}


# ── Text loading ──────────────────────────────────────────────────────────────

def _load_text(pdf_path: str) -> List[Document]:
    # Try OpenDataLoader first — #1 ranked PDF parser, handles tables + reading order
    odl_docs = {}
    try:
        import opendataloader_pdf, tempfile, os, json
        with tempfile.TemporaryDirectory() as tmpdir:
            opendataloader_pdf.convert(
                input_path=[pdf_path],
                output_dir=tmpdir,
                format="markdown",
            )
            # Find output markdown file
            md_files = [f for f in os.listdir(tmpdir) if f.endswith('.md')]
            if md_files:
                md_path = os.path.join(tmpdir, md_files[0])
                with open(md_path, encoding='utf-8') as f:
                    md_content = f.read()
                # Split by page markers if present, else treat as one chunk per ~3000 chars
                if '<!-- page' in md_content.lower():
                    import re
                    pages = re.split(r'<!--\s*[Pp]age\s*\d+\s*-->', md_content)
                    odl_docs = {i: p.strip() for i, p in enumerate(pages) if p.strip()}
                else:
                    # Split into ~3000 char pages
                    chunk_size = 3000
                    for i in range(0, len(md_content), chunk_size):
                        odl_docs[len(odl_docs)] = md_content[i:i+chunk_size].strip()
                print(f"[TEXT] OpenDataLoader: {len(odl_docs)} sections extracted")
    except Exception as e:
        odl_docs = {}
        print(f"[TEXT] OpenDataLoader unavailable ({e}), using pdfplumber/PyMuPDF")

    doc        = fitz.open(pdf_path)
    final_docs = []

    # Try pdfplumber for better table/structured PDF extraction
    plumber_texts = {}
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as plumb:
            for i, pg in enumerate(plumb.pages):
                # Extract tables as text first
                tables = pg.extract_tables()
                table_text = ""
                for tbl in tables:
                    for row in tbl:
                        row_clean = [str(c).strip() if c else "" for c in row]
                        if any(row_clean):
                            table_text += " | ".join(row_clean) + "\n"
                # Extract remaining text
                words = pg.extract_text(x_tolerance=3, y_tolerance=3) or ""
                combined = (words + "\n" + table_text).strip()
                if combined:
                    plumber_texts[i] = combined
        print(f"[TEXT] pdfplumber extracted {len(plumber_texts)} pages")
    except Exception as e:
        print(f"[TEXT] pdfplumber unavailable: {e}, using PyMuPDF")

    for page_num, page in enumerate(doc):
        page_label = page_num + 1
        # Priority: OpenDataLoader > pdfplumber > PyMuPDF
        text = (odl_docs.get(page_num, "")
                or plumber_texts.get(page_num, "")
                or page.get_text().strip())
        method = ("opendata" if odl_docs.get(page_num)
                  else "pdfplumber" if plumber_texts.get(page_num)
                  else "pymupdf")

        if _needs_ocr(text):
            print(f"[OCR] Page {page_label} needs OCR")
            ocr_text = _ocr_page(page)
            if ocr_text.strip():
                text   = ocr_text
                method = "ocr"

        if text.strip():
            first_line   = next((l.strip() for l in text.splitlines() if l.strip()), "")
            heading_hint = first_line[:80] if len(first_line) < 80 else ""
            final_docs.append(Document(
                text=text,
                metadata={
                    "page":               page_label,
                    "source_type":        "text",
                    "extraction_method":  method,
                    "section_heading":    heading_hint,
                    "pdf_path":           pdf_path,
                },
            ))

    doc.close()
    print(f"[TEXT] {len(final_docs)} pages loaded")
    return final_docs


# ── Chunking ──────────────────────────────────────────────────────────────────

def _build_nodes(text_docs: List[Document]) -> List:
    """Semantic chunking with fallback to sentence splitter."""
    print("[CHUNK] Semantic chunking…")
    try:
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        from llama_index.core.extractors import TitleExtractor, KeywordExtractor
        from llama_index.core.ingestion import IngestionPipeline

        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=85,
            embed_model=embed_model,
        )
        pipeline = IngestionPipeline(transformations=[
            splitter,
            TitleExtractor(nodes=3),
            KeywordExtractor(keywords=8),
        ])
        nodes = pipeline.run(documents=text_docs, show_progress=True)
        print(f"[CHUNK] {len(nodes)} semantic nodes")
        return nodes
    except Exception as e:
        print(f"[CHUNK] Semantic failed ({e}), using SentenceSplitter fallback")
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        nodes    = splitter.get_nodes_from_documents(text_docs)
        print(f"[CHUNK] {len(nodes)} sentence nodes")
        return nodes


# ── ChromaDB upsert ───────────────────────────────────────────────────────────

def _upsert_to_chroma(pdf_id: str, all_nodes: List) -> None:
    """Embed all nodes and store in ChromaDB."""
    collection = _get_or_create_collection(pdf_id)

    BATCH = 64
    total = 0

    for i in range(0, len(all_nodes), BATCH):
        batch = all_nodes[i: i + BATCH]

        ids        = []
        embeddings = []
        documents  = []
        metadatas  = []

        for node in batch:
            text = node.text or ""
            if not text.strip():
                continue

            try:
                emb = embed_model.get_text_embedding(text)
            except Exception as e:
                print(f"[CHROMA] Embed failed for node: {e}")
                continue

            # ChromaDB metadata values must be str/int/float/bool
            meta = {}
            for k, v in (node.metadata or {}).items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = str(v)

            ids.append(node.node_id)
            embeddings.append(emb)
            documents.append(text[:2000])  # Chroma stores doc text too
            metadatas.append(meta)

        if ids:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            total += len(ids)

    print(f"[CHROMA] Upserted {total} nodes for pdf_id={pdf_id}")


# ── ChromaDB query ────────────────────────────────────────────────────────────

def query_chroma(pdf_id: str, query_text: str, top_k: int = 8) -> List[Dict]:
    """Return top_k most similar nodes from ChromaDB."""
    try:
        collection = _chroma_client.get_collection(f"pdf_{pdf_id}")
    except Exception:
        return []

    try:
        emb = embed_model.get_text_embedding(query_text)
        results = collection.query(
            query_embeddings=[emb],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        nodes = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            nodes.append({
                "text":     doc,
                "metadata": meta,
                "score":    1.0 - dist,  # cosine distance → similarity
            })
        return nodes
    except Exception as e:
        print(f"[CHROMA] Query failed: {e}")
        return []


# ── Load from ChromaDB into memory ────────────────────────────────────────────

def _load_nodes_from_chroma(pdf_id: str) -> List:
    """Reconstruct TextNode list from ChromaDB for BM25 use."""
    try:
        collection = _chroma_client.get_collection(f"pdf_{pdf_id}")
        count      = collection.count()
        if count == 0:
            return []

        results = collection.get(include=["documents", "metadatas"])
        nodes   = []
        for doc, meta, nid in zip(
            results["documents"],
            results["metadatas"],
            results["ids"],
        ):
            try:
                n = TextNode(text=doc or "", id_=str(nid), metadata=meta or {})
            except TypeError:
                # Older LlamaIndex: set id_ after construction
                n = TextNode(text=doc or "", metadata=meta or {})
                object.__setattr__(n, "id_", str(nid))
            nodes.append(n)
        print(f"[CHROMA] Loaded {len(nodes)} nodes from disk for pdf_id={pdf_id}")
        return nodes
    except Exception as e:
        print(f"[CHROMA] Load failed: {e}")
        return []


# ── Metadata persistence (SQLite via auth.py) ─────────────────────────────────

def _save_pdf_meta(pdf_id: str, metadata: dict) -> None:
    """Persist full metadata to SQLite via auth.py (single source of truth)."""
    try:
        from auth import get_full_pdf_metadata
        import sqlite3, json
        db_path = Path("bujji.db")
        if not db_path.exists():
            return
        # Write to pdf_sessions metadata column directly (user_id=0 for system)
        # The proper save happens in _run_ingest via save_pdf_session with real user_id
        # This is a fallback for the ingestion pipeline to store full metadata
        con = sqlite3.connect(db_path)
        con.execute("""
            CREATE TABLE IF NOT EXISTS pdf_metadata_cache (
                pdf_id   TEXT PRIMARY KEY,
                metadata TEXT NOT NULL
            )
        """)
        con.execute(
            "INSERT OR REPLACE INTO pdf_metadata_cache (pdf_id, metadata) VALUES (?, ?)",
            (pdf_id, json.dumps(metadata))
        )
        con.commit()
        con.close()
    except Exception as e:
        print(f"[META] Save failed: {e}")


def _load_pdf_meta(pdf_id: str) -> Optional[dict]:
    """Load metadata — checks pdf_sessions first (has user data), then cache table."""
    import sqlite3, json
    db_path = Path("bujji.db")
    if not db_path.exists():
        return None
    try:
        con = sqlite3.connect(db_path)
        # Try pdf_sessions first (has normalised stats + images list)
        cur = con.execute(
            "SELECT metadata FROM pdf_sessions WHERE pdf_id = ? LIMIT 1", (pdf_id,)
        )
        row = cur.fetchone()
        if row:
            con.close()
            return json.loads(row[0])
        # Fallback: pdf_metadata_cache (written during ingest before user session saved)
        cur = con.execute(
            "SELECT metadata FROM pdf_metadata_cache WHERE pdf_id = ? LIMIT 1", (pdf_id,)
        )
        row = cur.fetchone()
        con.close()
        return json.loads(row[0]) if row else None
    except Exception:
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def check_size(pdf_path: str) -> None:
    """Raise ValueError if file exceeds MAX_PDF_BYTES."""
    size = os.path.getsize(pdf_path)
    if size > MAX_PDF_BYTES:
        mb = size / (1024 * 1024)
        raise ValueError(f"PDF is {mb:.1f} MB — maximum allowed is 50 MB.")


def ingest_pdf(pdf_path: str, force_reingest: bool = False) -> str:
    """
    Full ingestion pipeline. Returns pdf_id.
    If ChromaDB already has data for this pdf_id and force_reingest=False,
    skips re-embedding (fast path).
    """
    pdf_id = _pdf_id(pdf_path)

    # ── Fast path: already in ChromaDB ───────────────────────────────────────
    if not force_reingest and _collection_exists(pdf_id):
        print(f"[INGEST] pdf_id={pdf_id} already in ChromaDB, loading from disk…")
        meta = _load_pdf_meta(pdf_id)
        if meta:
            # Restore in-memory cache for BM25 use
            if pdf_id not in _cache:
                nodes = _load_nodes_from_chroma(pdf_id)
                load_parents_from_db(pdf_id)
                c2p = meta.get("child_to_parent", {})
                _cache[pdf_id] = {"nodes": nodes, "child_to_parent": c2p, "metadata": meta}
            print(f"[INGEST] ✓ Fast-loaded from ChromaDB (no re-embedding)")
            return pdf_id
        # metadata missing → fall through to full ingest

    print(f"[INGEST] Starting full ingest for {pdf_path}")

    # ── Step 1: Size check ────────────────────────────────────────────────────
    check_size(pdf_path)

    # ── Step 2: Text extraction ───────────────────────────────────────────────
    print("[STEP 1] Loading text…")
    text_docs = _load_text(pdf_path)
    print(f"[STEP 1] ✓ {len(text_docs)} pages")

    # ── Step 3: Image + table extraction ─────────────────────────────────────
    print("[STEP 2] Extracting images and tables…")
    extracted = _extract_images_and_tables(pdf_path, pdf_id)
    print(f"[STEP 2] ✓ {len(extracted['images'])} images, {len(extracted['tables'])} tables")

    if extracted["images"]:
        extracted["images"] = extract_all_captions(pdf_path, extracted["images"])

    # ── Step 4: Hierarchical chunking ────────────────────────────────────────
    # Child nodes (128 tokens) → embedded in ChromaDB for precise retrieval
    # Parent nodes (512 tokens) → stored separately, fetched at generation time
    print("[STEP 3] Hierarchical chunking...")
    child_nodes, parent_nodes, child_to_parent = build_hierarchical_nodes(text_docs)

    # ── Step 5: Table nodes ───────────────────────────────────────────────────
    table_nodes = [
        TextNode(
            text=f"[TABLE on page {t['page']}]\n{t['markdown']}",
            metadata={
                "source_type":    "table",
                "page":           t["page"],
                "table_index":    t["index"],
                "table_markdown": t["markdown"],
            },
        )
        for t in extracted["tables"]
    ]

    # ── Step 6: Image nodes (lightweight text refs) ───────────────────────────
    image_nodes = [
        TextNode(
            text=f"[IMAGE on page {img['page']}] Filename: {img['filename']}",
            metadata={
                "source_type": "image",
                "page":        img["page"],
                "image_path":  img["path"],
                "filename":    img["filename"],
            },
        )
        for img in extracted["images"]
    ]

    # Embed child nodes (small, precise) + table + image nodes
    all_nodes = child_nodes + table_nodes + image_nodes
    print(f"[STEP 4] Total nodes: {len(all_nodes)} ({len(child_nodes)} children, {len(table_nodes)} tables, {len(image_nodes)} images)")

    # ── Step 7: Upsert children to ChromaDB ──────────────────────────────────
    print("[STEP 5] Upserting to ChromaDB...")
    _upsert_to_chroma(pdf_id, all_nodes)

    # ── Step 7b: Store + persist parent nodes ────────────────────────────────
    store_parents(pdf_id, parent_nodes)
    persist_parents(pdf_id, parent_nodes)

    # ── Step 8: Persist metadata ──────────────────────────────────────────────
    metadata = {
        "images":        extracted["images"],
        "tables":        extracted["tables"],
        "pdf_path":      pdf_path,
        "doc_count":     len(all_nodes),
        "text_count":    len(child_nodes),
        "parent_count":  len(parent_nodes),
        "image_count":   len(extracted["images"]),
        "table_count":   len(extracted["tables"]),
        "child_to_parent": child_to_parent,
    }
    _save_pdf_meta(pdf_id, metadata)

    # ── Step 9: Cache in memory ───────────────────────────────────────────────
    _cache[pdf_id] = {
        "nodes":           all_nodes,
        "parent_nodes":    parent_nodes,
        "child_to_parent": child_to_parent,
        "metadata":        metadata,
    }

    print(f"[INGEST] ✓ Done. pdf_id={pdf_id}")
    return pdf_id


def get_metadata(pdf_id: str) -> dict:
    # Try memory cache first, then SQLite
    if pdf_id in _cache:
        return _cache[pdf_id]["metadata"]
    meta = _load_pdf_meta(pdf_id)
    return meta or {}


def get_all_nodes(pdf_id: str) -> List:
    """Return all nodes (from memory cache or reconstructed from ChromaDB)."""
    if pdf_id in _cache:
        return _cache[pdf_id]["nodes"]
    # Reconstruct from ChromaDB
    nodes = _load_nodes_from_chroma(pdf_id)
    meta  = _load_pdf_meta(pdf_id) or {}
    if nodes:
        load_parents_from_db(pdf_id)
        c2p = meta.get("child_to_parent", {})
        _cache[pdf_id] = {"nodes": nodes, "child_to_parent": c2p, "metadata": meta}
    return nodes


def get_index(pdf_id: str):
    """
    Legacy compatibility shim.
    Returns a lightweight wrapper that query.py can use.
    The real retrieval now goes through query_chroma() directly.
    """
    if _collection_exists(pdf_id) or pdf_id in _cache:
        return {"pdf_id": pdf_id, "ready": True}
    return None