"""
main.py — Bujji Babu API (v3)

Upgrades:
- PDF ingestion runs as BackgroundTask → returns job_id immediately
- /upload/status/{job_id} for polling progress
- 200MB upload size enforced
- Conversation history routes: GET /conversation, DELETE /conversation
- user_id passed to query_pdf for memory
- CSV save preferences route
"""

# ── Package path injection ─────────────────────────────────────────────────────
# Adds core/, pdf/, csv/, utils/ to sys.path so all existing flat imports
# (e.g. "from ingestion import ...") continue to work after folder reorganisation.
import sys as _sys, os as _os
_base = _os.path.dirname(_os.path.abspath(__file__))
for _pkg in ("core", "pdf", "csv_pipeline", "utils"):
    _p = _os.path.join(_base, _pkg)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ──────────────────────────────────────────────────────────────────────────────

import os
import uuid
import shutil
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from auth import (
    init_db, login, verify_token, logout,
    save_pdf_session, get_pdf_session, list_user_pdfs,
    find_pdf_by_filename, get_full_pdf_metadata,
)
from ingestion import ingest_pdf, get_metadata, get_index, check_size
from query import query_pdf, get_conversation_history, clear_conversation, save_feedback
from csv_agent import ingest_csv, query_csv, get_model_path, update_save_preference, build_nl_messages
from csv_session import get_csv_chat_history, delete_csv_session, list_csv_sessions, load_csv_session, update_last_turn_answer
from chat_exporter import export_csv_chat_to_pdf
import llm_client
from exporter import get_export_path as get_csv_export_path

# ── Init ──────────────────────────────────────────────────────────────────────
init_db()

app = FastAPI(title="Bujji Babu API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR     = Path("uploads");     UPLOAD_DIR.mkdir(exist_ok=True)
CSV_UPLOAD_DIR = Path("csv_uploads"); CSV_UPLOAD_DIR.mkdir(exist_ok=True)
Path("extracted").mkdir(exist_ok=True)

MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB

app.mount("/extracted", StaticFiles(directory="extracted"), name="extracted")
app.mount("/static",    StaticFiles(directory="static"),    name="static")

# ── In-memory job tracker ─────────────────────────────────────────────────────
# Format: { job_id: { "status": "pending"|"running"|"done"|"error",
#                     "pdf_id": ..., "filename": ..., "message": ..., "stats": ... } }
_jobs: dict = {}


# ── Auth dependency ───────────────────────────────────────────────────────────

def get_user(authorization: Optional[str] = Header(None)) -> int:
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
    user_id = verify_token(token) if token else None
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized. Please log in.")
    return user_id


# ── Pydantic models ───────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str

class QueryRequest(BaseModel):
    pdf_id:   str
    question: str
    top_k:    Optional[int] = 5

class CSVQueryRequest(BaseModel):
    csv_id:   str
    question: str

class SavePrefsRequest(BaseModel):
    csv_id:       str
    save_charts:  bool = True
    save_models:  bool = True
    save_history: bool = True

class FeedbackRequest(BaseModel):
    pdf_id:   str
    question: str
    answer:   str
    rating:   int          # 1 = thumbs-up, -1 = thumbs-down
    comment:  Optional[str] = ""


# ── Static / root ─────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.post("/auth/login")
async def auth_login(body: LoginRequest):
    token = login(body.username, body.password)
    if not token:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    return {"token": token}


@app.post("/auth/logout")
async def auth_logout(authorization: Optional[str] = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        logout(authorization[7:])
    return {"ok": True}


@app.get("/auth/me")
async def auth_me(user_id: int = Depends(get_user)):
    return {"user_id": user_id, "ok": True}


# ── PDF list ──────────────────────────────────────────────────────────────────

@app.get("/pdfs")
async def list_pdfs(user_id: int = Depends(get_user)):
    return {"pdfs": list_user_pdfs(user_id)}


# ── Background ingestion task ─────────────────────────────────────────────────

def _run_ingest(job_id: str, pdf_path: str, filename: str, user_id: int):
    """Runs in background thread. Updates _jobs dict with progress."""
    _jobs[job_id]["status"] = "running"
    _jobs[job_id]["message"] = "Extracting text and images…"
    try:
        pdf_id = ingest_pdf(pdf_path)
        meta   = get_metadata(pdf_id)

        # Save full metadata — includes images/tables lists for post-restart loading
        save_pdf_session(
            pdf_id=pdf_id,
            user_id=user_id,
            filename=filename,
            pdf_path=pdf_path,
            metadata={
                "text_chunks":  meta.get("text_count",  0),
                "images_found": meta.get("image_count", 0),
                "tables_found": meta.get("table_count", 0),
                "images":       meta.get("images",      []),
                "tables":       meta.get("tables",      []),
            }
        )

        stats = {
            "text_chunks":   meta.get("text_count",  0),
            "images_found":  meta.get("image_count", 0),
            "tables_found":  meta.get("table_count", 0),
        }
        _jobs[job_id].update({
            "status":   "done",
            "pdf_id":   pdf_id,
            "filename": filename,
            "cached":   False,
            "message":  "PDF indexed successfully.",
            "stats":    stats,
        })
        print(f"[JOB {job_id}] ✓ Done. pdf_id={pdf_id}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        _jobs[job_id].update({
            "status":  "error",
            "message": str(e),
        })
        print(f"[JOB {job_id}] ✗ Failed: {e}")


# ── Upload PDF ────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: int = Depends(get_user),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # ── Size check (streaming, before writing) ────────────────────────────────
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        mb = len(content) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File is {mb:.1f} MB — maximum is 200 MB."
        )

    # ── Check if already ingested ─────────────────────────────────────────────
    existing_pdf_id = find_pdf_by_filename(file.filename, user_id)
    if existing_pdf_id:
        # Try to load from ChromaDB (fast path — no re-embedding)
        try:
            pdf_id = ingest_pdf(
                get_pdf_session(existing_pdf_id, user_id)["pdf_path"],
                force_reingest=False,
            )
            meta = get_metadata(pdf_id)
            return JSONResponse({
                "success":  True,
                "job_id":   None,
                "pdf_id":   pdf_id,
                "filename": file.filename,
                "cached":   True,
                "status":   "done",
                "message":  "Loaded from your previous session — no re-processing needed.",
                "stats": {
                    "text_chunks":  meta.get("text_count",  0),
                    "images_found": meta.get("image_count", 0),
                    "tables_found": meta.get("table_count", 0),
                },
            })
        except Exception:
            pass  # Fall through to fresh ingest

    # ── Save file ─────────────────────────────────────────────────────────────
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        f.write(content)

    # ── Create job and kick off background ingestion ──────────────────────────
    job_id = str(uuid.uuid4())[:12]
    _jobs[job_id] = {
        "status":   "pending",
        "pdf_id":   None,
        "filename": file.filename,
        "message":  "Queued for processing…",
        "stats":    None,
    }

    background_tasks.add_task(_run_ingest, job_id, str(save_path), file.filename, user_id)

    return JSONResponse({
        "success":  True,
        "job_id":   job_id,
        "pdf_id":   None,
        "filename": file.filename,
        "cached":   False,
        "status":   "pending",
        "message":  "Processing started. Poll /upload/status/{job_id} for updates.",
    })


@app.get("/upload/status/{job_id}")
async def upload_status(job_id: str, user_id: int = Depends(get_user)):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JSONResponse(job)


# ── Query PDF ─────────────────────────────────────────────────────────────────

@app.post("/pdfs/load/{pdf_id}")
async def load_pdf(pdf_id: str, user_id: int = Depends(get_user)):
    """
    Called when user clicks a saved PDF in the sidebar.
    Warms ChromaDB cache without re-uploading.
    Returns current stats from SQLite.
    """
    # Get saved session from SQLite
    saved = get_pdf_session(pdf_id, user_id)
    if not saved:
        raise HTTPException(status_code=404, detail="PDF session not found.")

    meta = saved.get("metadata", {})
    pdf_path = saved.get("pdf_path", "")

    # Warm in-memory cache if needed (ChromaDB persists, just need _cache populated)
    index = get_index(pdf_id)
    if index is None:
        # ChromaDB evicted or missing — re-ingest silently
        if pdf_path and Path(pdf_path).exists():
            try:
                ingest_pdf(pdf_path, force_reingest=False)
                fresh_meta = get_metadata(pdf_id)
                if fresh_meta:
                    meta = {
                        "text_chunks":  fresh_meta.get("text_count",  0),
                        "images_found": fresh_meta.get("image_count", 0),
                        "tables_found": fresh_meta.get("table_count", 0),
                        "images":       fresh_meta.get("images",      []),
                        "tables":       fresh_meta.get("tables",      []),
                    }
            except Exception as e:
                print(f"[LOAD] Re-ingest failed: {e}")

    return JSONResponse({
        "pdf_id":   pdf_id,
        "filename": saved.get("filename", ""),
        "cached":   True,
        "stats": {
            "text_chunks":  meta.get("text_chunks",  meta.get("text_count",  0)),
            "images_found": meta.get("images_found", meta.get("image_count", 0)),
            "tables_found": meta.get("tables_found", meta.get("table_count", 0)),
        },
    })


@app.post("/query")
async def query(request: QueryRequest, user_id: int = Depends(get_user)):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = query_pdf(
            request.pdf_id,
            request.question,
            request.top_k,
            user_id=user_id,
        )
        # Fix image paths for serving
        for img in result.get("images", []):
            raw_path   = img.get("path", "")
            # Normalise Windows backslashes + make path relative
            normalized = raw_path.replace("\\", "/").replace("\\\\", "/")
            # Extract from 'extracted/' onwards
            if "extracted/" in normalized:
                normalized = normalized[normalized.index("extracted/"):]
            elif "extracted\\" in raw_path:
                normalized = raw_path.replace("\\", "/")
                normalized = normalized[normalized.index("extracted/"):]
            img["url"] = "/" + normalized.lstrip("/")
        return JSONResponse(result)
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# ── Conversation memory routes ────────────────────────────────────────────────

@app.delete("/pdfs/{pdf_id}")
async def delete_pdf(pdf_id: str, user_id: int = Depends(get_user)):
    """
    Delete a PDF session — removes from SQLite, ChromaDB, and memory cache.
    User can re-upload the same PDF fresh after this.
    """
    import sqlite3

    # 1. Remove from SQLite pdf_sessions
    try:
        con = sqlite3.connect("bujji.db")
        con.execute("DELETE FROM pdf_sessions WHERE pdf_id = ? AND user_id = ?", (pdf_id, user_id))
        con.execute("DELETE FROM pdf_metadata_cache WHERE pdf_id = ?", (pdf_id,))
        con.execute("DELETE FROM pdf_conversations WHERE pdf_id = ? AND user_id = ?", (pdf_id, user_id))
        con.execute("DELETE FROM parent_nodes WHERE pdf_id = ?", (pdf_id,))
        con.commit()
        con.close()
        print(f"[DELETE] SQLite cleared for pdf_id={pdf_id}")
    except Exception as e:
        print(f"[DELETE] SQLite error: {e}")

    # 2. Remove from ChromaDB
    try:
        from ingestion import _chroma_client, _cache
        _chroma_client.delete_collection(f"pdf_{pdf_id}")
        print(f"[DELETE] ChromaDB collection deleted for pdf_id={pdf_id}")
    except Exception as e:
        print(f"[DELETE] ChromaDB error: {e}")

    # 3. Remove from memory cache
    try:
        from ingestion import _cache
        _cache.pop(pdf_id, None)
        print(f"[DELETE] Memory cache cleared for pdf_id={pdf_id}")
    except Exception as e:
        print(f"[DELETE] Cache error: {e}")

    # 4. Remove from BM25 cache
    try:
        from query import _bm25_cache
        _bm25_cache.pop(pdf_id, None)
    except Exception:
        pass

    # 5. Remove from chunker parent store
    try:
        from chunker import _parent_store
        _parent_store.pop(pdf_id, None)
    except Exception:
        pass

    return {"ok": True, "message": f"PDF {pdf_id} deleted successfully."}


@app.get("/conversation/{pdf_id}")
async def get_conv(pdf_id: str, user_id: int = Depends(get_user)):
    history = get_conversation_history(pdf_id, user_id, last_n=20)
    return {"history": history}


@app.delete("/conversation/{pdf_id}")
async def clear_conv(pdf_id: str, user_id: int = Depends(get_user)):
    clear_conversation(pdf_id, user_id)
    return {"ok": True, "message": "Conversation cleared."}


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest, user_id: int = Depends(get_user)):
    """
    Store user thumbs-up (rating=1) or thumbs-down (rating=-1) for a PDF answer.
    Saved in the pdf_feedback SQLite table for future quality analysis.
    """
    if request.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="Rating must be 1 (up) or -1 (down).")
    save_feedback(
        pdf_id   = request.pdf_id,
        user_id  = user_id,
        question = request.question,
        answer   = request.answer,
        rating   = request.rating,
        comment  = request.comment or "",
    )
    return {"ok": True}


# ── CSV routes (no auth) ──────────────────────────────────────────────────────

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files supported.")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        mb = len(content) / (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"File is {mb:.1f} MB — maximum is 200 MB.")

    save_path = CSV_UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        f.write(content)

    try:
        result = ingest_csv(str(save_path))
        return JSONResponse({"success": True, **result})
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query-csv")
async def query_csv_endpoint(request: CSVQueryRequest):
    try:
        result = query_csv(request.csv_id, request.question)
        # If no export path in result but session has one, include it
        if not result.get("export_path"):
            from csv_agent import _sessions
            session = _sessions.get(request.csv_id, {})
            if session.get("last_export_path"):
                result["export_path"] = session["last_export_path"]
                result["has_export"]  = True
        return JSONResponse(result)
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/csv-prefs")
async def set_csv_prefs(req: SavePrefsRequest):
    update_save_preference(req.csv_id, req.save_charts, req.save_models, req.save_history)
    return {"ok": True}


@app.get("/download-model/{csv_id}")
async def download_model(csv_id: str):
    from fastapi.responses import FileResponse as FileResp
    model_path = get_model_path(csv_id)
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="No model trained yet.")
    fname = Path(model_path).name
    return FileResp(model_path, filename=fname, media_type="application/octet-stream")


@app.get("/download-csv/{csv_id}")
async def download_csv(csv_id: str, label: str = "", file: str = ""):
    """
    Serve an exported CSV.
    - `file` param: specific filename inside exports/ (preferred — exact file from this session)
    - `label` param: fallback label (filtered / original / cleaned)
    - default:       most-recent export for this csv_id
    """
    from fastapi.responses import FileResponse as FileResp
    from exporter import get_export_path, EXPORT_DIR
    from pathlib import Path as _Path

    exports_dir = EXPORT_DIR

    # 1. Exact filename requested (from frontend metadata)
    if file:
        exact = exports_dir / file
        if exact.exists():
            return FileResp(str(exact), filename=exact.name, media_type="text/csv")

    # 2. Fallback: most recent matching label
    for try_label in (["filtered", "original", "cleaned"] if not label else [label, "filtered", "original"]):
        if not try_label:
            continue
        export_path = get_export_path(csv_id, try_label)
        if export_path and os.path.exists(export_path):
            fname = _Path(export_path).name
            return FileResp(export_path, filename=fname, media_type="text/csv")

    raise HTTPException(status_code=404, detail="No exported CSV found.")


@app.post("/query-stream")
async def query_stream(request: QueryRequest, user_id: int = Depends(get_user)):
    """
    Streaming version of /query.
    Returns server-sent events (SSE) so the frontend can render tokens as they arrive.
    Format: data: <json_chunk>\n\n
    Final event: data: [DONE]\n\n
    """
    import json
    from groq import Groq as _Groq
    import os

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Run non-streaming parts first (retrieval, reranking, image scoring)
    # Then stream only the LLM generation step
    try:
        result = query_pdf(
            request.pdf_id,
            request.question,
            request.top_k,
            user_id=user_id,
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    # Fix image paths
    for img in result.get("images", []):
        raw_path   = img.get("path", "")
        normalized = raw_path.replace("\\", "/").replace("\\\\", "/")
        if "extracted/" in normalized:
            normalized = normalized[normalized.index("extracted/"):]
        img["url"] = "/" + normalized.lstrip("/")

    # Send metadata first (sources, images, tables, confidence)
    async def event_generator():
        import asyncio

        # 1. Send metadata chunk immediately
        meta = {
            "type":         "meta",
            "sources":      result.get("sources", []),
            "images":       result.get("images", []),
            "tables":       result.get("tables", []),
            "confidence":   result.get("confidence"),
            "warning":      result.get("warning"),
            "query_class":  result.get("query_class"),
            "faithfulness": result.get("faithfulness"),
        }
        yield f"data: {json.dumps(meta, separators=(',', ':'))}\n\n"

        # 2. Stream answer word by word with delay for typing effect
        answer = result.get("answer", "")
        words  = answer.split(" ")

        for i, word in enumerate(words):
            # Add space between words (not after last word)
            text = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'type': 'token', 'text': text})}\n\n"
            # Typing delay — shorter for common words, longer for rare
            delay = 0.025 if len(word) <= 4 else 0.04
            await asyncio.sleep(delay)

        # 3. Done signal
        yield f"data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
            "Transfer-Encoding":           "chunked",
            "X-Content-Type-Options":      "nosniff",
        },
    )


@app.get("/csv-sessions")
async def get_csv_sessions(user_id: int = Depends(get_user)):
    """List all CSV sessions for the sidebar."""
    return {"sessions": list_csv_sessions()}


@app.delete("/csv-sessions/{csv_id}")
async def delete_csv(csv_id: str, user_id: int = Depends(get_user)):
    """Delete a CSV session from SQLite + memory."""
    from csv_agent import _sessions
    _sessions.pop(csv_id, None)
    delete_csv_session(csv_id)
    return {"ok": True}


@app.get("/export-chat/{csv_id}")
def export_chat(
    csv_id: str,
    token: Optional[str] = None,          # accept token as query param for direct-link downloads
    authorization: Optional[str] = Header(None),
):
    """
    Export CSV chat history as PDF.

    Auth: Bearer token in Authorization header (fetch) OR ?token=... query param
    (direct <a href> navigation — keeps browser user-gesture window open so the
    download is not silently blocked after a long PDF generation).
    """
    from fastapi.responses import FileResponse as FileResp
    from auth import verify_token as _verify_token

    # Resolve token from either source
    _raw = token
    if not _raw and authorization and authorization.startswith("Bearer "):
        _raw = authorization[7:]
    if not _raw or not _verify_token(_raw):
        raise HTTPException(status_code=401, detail="Unauthorized. Please log in.")

    history = get_csv_chat_history(csv_id)
    session = load_csv_session(csv_id)
    filename = session["filename"] if session else csv_id
    stats = {"rows": session["shape_rows"], "cols": session["shape_cols"]} if session else None

    try:
        path = export_csv_chat_to_pdf(csv_id, filename, history, stats)
    except Exception as _e:
        print(f"[EXPORT] Unhandled error: {_e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {_e}")

    if not os.path.exists(path):
        raise HTTPException(status_code=500, detail="Export failed: file not generated")

    fname = Path(path).name
    media  = "application/pdf" if path.endswith(".pdf") else "text/html"
    return FileResp(
        path,
        filename=fname,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.post("/query-csv-stream")
async def query_csv_stream(request: CSVQueryRequest, user_id: int = Depends(get_user)):
    """
    True streaming CSV query using Groq streaming API.
    Returns SSE: metadata chunk first, then token chunks, then [DONE].
    """
    import json
    from groq import Groq as _Groq

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Run code execution + analysis synchronously (can't stream that part)
    try:
        result = query_csv(request.csv_id, request.question, skip_nl_answer=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    async def event_generator():
        import asyncio

        _pj     = result.get("plotly_json")
        _pj_str = _pj if _pj else None

        # 1. Send metadata
        meta = {
            "type":          "meta",
            "stat_artifact": result.get("stat_artifact"),
            "plotly_json":   None,
            "model_path":    result.get("model_path"),
            "has_chart":     bool(_pj_str),
            "has_model":     result.get("has_model", False),
            "raw_output":    result.get("raw_output"),
            "export_path":   result.get("export_path"),
            "has_export":    result.get("has_export", False),
            "is_filter":     result.get("is_filter", False),
        }
        yield f"data: {json.dumps(meta, separators=(',', ':'))}\n\n"

        if _pj_str:
            yield f"data: {json.dumps({'type':'chart','plotly_json':_pj_str}, separators=(',',':'))}\n\n"

        # 2. True Groq streaming for the NL answer
        raw_out    = result.get("raw_output") or result.get("output") or ""
        has_chart  = bool(_pj_str)
        has_model  = result.get("has_model", False)
        pre_answer = result.get("answer") or ""  # pre-computed by shortcut paths

        # Accumulate the final answer so we can persist it to the DB for PDF export
        _streamed_answer_parts: list = []

        if pre_answer:
            # Stat overview / correlation / column-listing paths already called the LLM
            # synchronously — stream the pre-built answer word-by-word with a small delay
            # so the typing effect is visible (mirrors the PDF streaming cadence).
            # These paths already saved the correct answer via _save_and_return, so
            # no DB update is needed here.
            words = pre_answer.split(" ")
            for i, word in enumerate(words):
                if word:
                    text = word + (" " if i < len(words) - 1 else "")
                    yield f"data: {json.dumps({'type':'token','text':text})}\n\n"
                    delay = 0.02 if len(word) <= 4 else 0.035
                    await asyncio.sleep(delay)
        elif raw_out or has_chart or has_model:
            # Code-execution paths: stream a fresh NL answer from the raw output.
            #
            # IMPORTANT: the Groq sync client blocks the event loop while waiting for
            # each HTTP chunk from Groq's server. If we iterate it directly with
            # `for chunk in stream:`, uvicorn cannot flush already-yielded SSE events
            # during those blocking waits — everything arrives at the client in one burst.
            #
            # Fix: push the sync iterator onto a background thread and feed tokens
            # into an asyncio.Queue. The async generator dequeues them one by one,
            # so the event loop (and uvicorn) stays free to flush each token as it lands.
            import threading

            try:
                messages = build_nl_messages(request.question, raw_out, has_chart, has_model=has_model)

                token_queue: asyncio.Queue = asyncio.Queue()
                loop = asyncio.get_running_loop()

                def _groq_worker():
                    """Run blocking Groq streaming in a thread; push tokens to queue."""
                    try:
                        _stream = llm_client.get_client().chat.completions.create(
                            model=llm_client.get_model("fast"),
                            messages=messages,
                            max_tokens=350,
                            temperature=0.3,
                            stream=True,
                        )
                        for _chunk in _stream:
                            _delta = _chunk.choices[0].delta.content or ""
                            if _delta:
                                loop.call_soon_threadsafe(token_queue.put_nowait, _delta)
                    except Exception as _e:
                        print(f"[CSV STREAM] groq worker error: {_e}")
                    finally:
                        loop.call_soon_threadsafe(token_queue.put_nowait, None)  # sentinel

                worker = threading.Thread(target=_groq_worker, daemon=True)
                worker.start()

                # Drain the queue with a per-token timeout so we never hang
                # forever if Ollama silently drops the connection mid-stream.
                # 300s matches the Ollama client read timeout in llm_client.py.
                _STREAM_TIMEOUT = 300
                while True:
                    try:
                        token = await asyncio.wait_for(
                            token_queue.get(), timeout=_STREAM_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        print(f"[CSV STREAM] token queue timed out after {_STREAM_TIMEOUT}s")
                        break
                    if token is None:
                        break
                    _streamed_answer_parts.append(token)
                    yield f"data: {json.dumps({'type':'token','text':token})}\n\n"
                    await asyncio.sleep(0)  # let uvicorn flush this token before fetching next

                # join on a thread executor so we don't block the event loop
                await loop.run_in_executor(None, lambda: worker.join(timeout=10))

            except Exception as e:
                print(f"[CSV STREAM] NL answer error: {e}")

        # Persist the streamed answer back to DB so PDF export shows real text
        # (the DB row was initially saved with answer="" by the code-execution path)
        if _streamed_answer_parts:
            _final_answer = "".join(_streamed_answer_parts).strip()
            if _final_answer:
                try:
                    update_last_turn_answer(request.csv_id, request.question, _final_answer)
                except Exception as _ue:
                    print(f"[CSV STREAM] answer DB update failed: {_ue}")

        # If truly nothing (no pre_answer, no raw_out, no chart, no model) — send no tokens;
        # the frontend will show an empty bubble (chart/artifact fills the space instead).

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        },
    )


@app.get("/mode")
async def get_mode():
    """Get current LLM mode and available models."""
    return {
        "mode":  llm_client.get_mode(),
        "model": llm_client.get_model("fast"),
        "vlm":   llm_client.get_vlm_model(),
    }


@app.post("/mode/{mode}")
async def set_mode(mode: str, user_id: int = Depends(get_user)):
    """Switch between online (Groq) and offline (Ollama) mode."""
    if mode not in ("online", "offline"):
        raise HTTPException(status_code=400, detail="Mode must be 'online' or 'offline'")
    llm_client.set_mode(mode)
    # Invalidate client cache so new client is created
    llm_client._client_cache.clear()
    health = llm_client.check_health()
    return {"ok": health["ok"], "mode": mode, "model": llm_client.get_model("fast"), "error": health.get("error")}


@app.get("/ragas/{pdf_id}")
async def get_ragas_scores(pdf_id: str, user_id: int = Depends(get_user)):
    """Get RAGAS evaluation scores for a PDF."""
    import sqlite3
    try:
        con = sqlite3.connect("bujji.db")
        cur = con.execute("""
            SELECT faithfulness, answer_relevancy, context_precision, context_recall,
                   n_questions, evaluated_at
            FROM ragas_results WHERE pdf_id = ?
            ORDER BY id DESC LIMIT 1
        """, (pdf_id,))
        row = cur.fetchone()
        con.close()
        if not row:
            return {"available": False}
        return {
            "available":          True,
            "faithfulness":       row[0],
            "answer_relevancy":   row[1],
            "context_precision":  row[2],
            "context_recall":     row[3],
            "n_questions":        row[4],
            "evaluated_at":       row[5],
            "overall":            round(sum([row[0],row[1],row[2],row[3]])/4, 4),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


@app.post("/ragas/{pdf_id}/run")
async def run_ragas_eval(
    pdf_id: str,
    background_tasks: BackgroundTasks,
    n: int = 5,
    user_id: int = Depends(get_user),
):
    """Manually trigger RAGAS evaluation for a PDF."""
    def _run():
        from ragas_eval import evaluate_pdf
        evaluate_pdf(pdf_id, n_questions=n, user_id=user_id)

    background_tasks.add_task(_run)
    return {"message": f"RAGAS evaluation started for {pdf_id} with {n} questions"}


@app.get("/health")
async def health():
    """
    Basic health check — also reports GPU and Ollama status so the
    frontend / ops team can see the offline-mode readiness at a glance.
    """
    gpu_info    = llm_client.check_gpu()
    ollama_info = llm_client.check_ollama_daemon()
    return {
        "status":        "ok",
        "groq_key_set":  bool(os.getenv("GROQ_API_KEY")),
        "version":       "3.0.0",
        "llm_mode":      llm_client.get_mode(),
        "gpu":           gpu_info,
        "ollama":        ollama_info,
    }