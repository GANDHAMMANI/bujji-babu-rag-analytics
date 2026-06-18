"""
auth.py — SQLite-backed auth + PDF session persistence for Bujji Babu
"""
import os
import sqlite3
import hashlib
import secrets
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

DB_PATH = Path("bujji.db")

# ── Schema ────────────────────────────────────────────────────────────────────

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT UNIQUE NOT NULL,
            password   TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sessions (
            token      TEXT PRIMARY KEY,
            user_id    INTEGER NOT NULL,
            expires_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS pdf_sessions (
            pdf_id      TEXT NOT NULL,
            user_id     INTEGER NOT NULL,
            filename    TEXT NOT NULL,
            pdf_path    TEXT NOT NULL,
            metadata    TEXT NOT NULL DEFAULT '{}',
            created_at  TEXT NOT NULL,
            PRIMARY KEY (pdf_id, user_id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)

    # Seed default user from .env (fallback to admin/bujji123)
    username = os.getenv("BUJJI_USER", "admin")
    password = os.getenv("BUJJI_PASS", "bujji123")
    _hash = _hash_password(password)
    cur.execute(
        "INSERT OR IGNORE INTO users (username, password, created_at) VALUES (?, ?, ?)",
        (username, _hash, datetime.utcnow().isoformat())
    )
    con.commit()
    con.close()
    print(f"[AUTH] DB ready. Default user: '{username}'")


# ── Password ──────────────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    salt = os.getenv("BUJJI_SALT", "bujji_salt_2025")
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


# ── Auth ──────────────────────────────────────────────────────────────────────

def login(username: str, password: str) -> Optional[str]:
    """Returns session token on success, None on failure."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT id, password FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    if not row or row[1] != _hash_password(password):
        con.close()
        return None

    user_id = row[0]
    token = secrets.token_hex(32)
    expires = (datetime.utcnow() + timedelta(hours=12)).isoformat()
    cur.execute(
        "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
        (token, user_id, expires)
    )
    con.commit()
    con.close()
    return token


def verify_token(token: str) -> Optional[int]:
    """Returns user_id if valid, None if expired/invalid."""
    if not token:
        return None
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "SELECT user_id, expires_at FROM sessions WHERE token = ?", (token,)
    )
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    if datetime.utcnow().isoformat() > row[1]:
        return None
    return row[0]


def logout(token: str):
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM sessions WHERE token = ?", (token,))
    con.commit()
    con.close()


# ── PDF Session Persistence ───────────────────────────────────────────────────

def save_pdf_session(pdf_id: str, user_id: int, filename: str, pdf_path: str, metadata: dict):
    # Normalise keys — always store with frontend-compatible names
    normalised = {
        "text_chunks":   metadata.get("text_chunks",  metadata.get("text_count",  0)),
        "images_found":  metadata.get("images_found", metadata.get("image_count", 0)),
        "tables_found":  metadata.get("tables_found", metadata.get("table_count", 0)),
        # Also preserve full image list for vision reranking after restart
        "images":        metadata.get("images", []),
        "tables":        metadata.get("tables", []),
    }
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """INSERT OR REPLACE INTO pdf_sessions
           (pdf_id, user_id, filename, pdf_path, metadata, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (pdf_id, user_id, filename, pdf_path, json.dumps(normalised), datetime.utcnow().isoformat())
    )
    con.commit()
    con.close()


def get_pdf_session(pdf_id: str, user_id: int) -> Optional[Dict[str, Any]]:
    """Returns saved metadata if this user has previously ingested this PDF."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "SELECT filename, pdf_path, metadata FROM pdf_sessions WHERE pdf_id = ? AND user_id = ?",
        (pdf_id, user_id)
    )
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    return {
        "filename": row[0],
        "pdf_path": row[1],
        "metadata": json.loads(row[2]),
    }


def list_user_pdfs(user_id: int) -> list:
    """List all PDFs previously uploaded by this user."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """SELECT pdf_id, filename, metadata, created_at
           FROM pdf_sessions WHERE user_id = ?
           ORDER BY created_at DESC""",
        (user_id,)
    )
    rows = cur.fetchall()
    con.close()
    result = []
    for r in rows:
        meta = json.loads(r[2])
        result.append({
            "pdf_id":      r[0],
            "filename":    r[1],
            "uploaded_at": r[3],
            "stats": {
                "text_chunks":  meta.get("text_chunks",  meta.get("text_count",  0)),
                "images_found": meta.get("images_found", meta.get("image_count", 0)),
                "tables_found": meta.get("tables_found", meta.get("table_count", 0)),
            },
        })
    return result


def get_full_pdf_metadata(pdf_id: str) -> Optional[dict]:
    """Return full metadata including images/tables lists — used on restart."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT metadata FROM pdf_sessions WHERE pdf_id = ? LIMIT 1", (pdf_id,))
    row = cur.fetchone()
    con.close()
    return json.loads(row[0]) if row else None


def find_pdf_by_filename(filename: str, user_id: int) -> Optional[str]:
    """Check if user already has an ingested PDF with this exact filename."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "SELECT pdf_id FROM pdf_sessions WHERE filename = ? AND user_id = ?",
        (filename, user_id)
    )
    row = cur.fetchone()
    con.close()
    return row[0] if row else None