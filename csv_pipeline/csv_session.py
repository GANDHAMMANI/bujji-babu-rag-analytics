"""
csv_session.py — SQLite-backed CSV Session Persistence for Bujji Babu

Saves:
  - CSV file path, EDA summary, column info
  - Query history (question + answer + code)
  - Last model path
  - Save preferences

On server restart: user re-uploads CSV → instantly restores history
No re-ingestion needed since CSV is cheap (just pandas read)
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

DB_PATH = Path("bujji.db")


def _init_csv_tables():
    if not DB_PATH.exists():
        return
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS csv_sessions (
            csv_id      TEXT PRIMARY KEY,
            csv_path    TEXT NOT NULL,
            filename    TEXT NOT NULL,
            eda_summary TEXT NOT NULL,
            columns     TEXT NOT NULL,
            dtypes      TEXT NOT NULL,
            shape_rows  INTEGER NOT NULL,
            shape_cols  INTEGER NOT NULL,
            prefs       TEXT NOT NULL DEFAULT '{}',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS csv_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            csv_id      TEXT NOT NULL,
            question    TEXT NOT NULL,
            answer      TEXT NOT NULL,
            code        TEXT,
            has_chart   INTEGER DEFAULT 0,
            has_model   INTEGER DEFAULT 0,
            model_path  TEXT,
            plotly_json   TEXT,
            stat_artifact TEXT,
            created_at    TEXT NOT NULL,
            FOREIGN KEY (csv_id) REFERENCES csv_sessions(csv_id)
        );
    """)
    con.commit()
    con.close()


_init_csv_tables()


def save_csv_session(
    csv_id: str,
    csv_path: str,
    filename: str,
    eda_summary: str,
    columns: list,
    dtypes: dict,
    shape: tuple,
    prefs: dict,
):
    if not DB_PATH.exists():
        return
    now = datetime.utcnow().isoformat()
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        INSERT OR REPLACE INTO csv_sessions
        (csv_id, csv_path, filename, eda_summary, columns, dtypes,
         shape_rows, shape_cols, prefs, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        csv_id, csv_path, filename, eda_summary,
        json.dumps(columns), json.dumps(dtypes),
        shape[0], shape[1],
        json.dumps(prefs), now, now,
    ))
    con.commit()
    con.close()


def load_csv_session(csv_id: str) -> Optional[Dict]:
    if not DB_PATH.exists():
        return None
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.execute(
            "SELECT * FROM csv_sessions WHERE csv_id = ?", (csv_id,)
        )
        row = cur.fetchone()
        if not row:
            con.close()
            return None
        cols = [d[0] for d in cur.description]
        data = dict(zip(cols, row))
        data["columns"] = json.loads(data["columns"])
        data["dtypes"]   = json.loads(data["dtypes"])
        data["prefs"]    = json.loads(data["prefs"])

        # Load history
        cur2 = con.execute(
            "SELECT question, answer, code, has_chart, has_model, model_path "
            "FROM csv_history WHERE csv_id = ? ORDER BY id",
            (csv_id,)
        )
        history = []
        for r in cur2.fetchall():
            history.append({
                "question":  r[0],
                "answer":    r[1],
                "code":      r[2],
                "has_chart": bool(r[3]),
                "has_model": bool(r[4]),
                "model_path": r[5],
            })
        data["history"] = history
        con.close()
        return data
    except Exception as e:
        print(f"[CSV_SESSION] Load failed: {e}")
        return None


def _migrate_plotly_column():
    """Add plotly_json column to existing DBs that predate this field."""
    if not DB_PATH.exists():
        return
    try:
        con = sqlite3.connect(DB_PATH)
        cols = [r[1] for r in con.execute("PRAGMA table_info(csv_history)").fetchall()]
        if "plotly_json" not in cols:
            con.execute("ALTER TABLE csv_history ADD COLUMN plotly_json TEXT")
            con.commit()
        con.close()
    except Exception:
        pass


_migrate_plotly_column()


def _migrate_stat_artifact_column():
    if not DB_PATH.exists():
        return
    try:
        con = sqlite3.connect(DB_PATH)
        cols = [r[1] for r in con.execute("PRAGMA table_info(csv_history)").fetchall()]
        if "stat_artifact" not in cols:
            con.execute("ALTER TABLE csv_history ADD COLUMN stat_artifact TEXT")
            con.commit()
        con.close()
    except Exception:
        pass


_migrate_stat_artifact_column()


def save_csv_turn(
    csv_id: str,
    question: str,
    answer: str,
    code: Optional[str] = None,
    has_chart: bool = False,
    has_model: bool = False,
    model_path: Optional[str] = None,
    plotly_json: Optional[str] = None,
    stat_artifact: Optional[str] = None,
):
    if not DB_PATH.exists():
        return
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute("""
            INSERT INTO csv_history
            (csv_id, question, answer, code, has_chart, has_model, model_path, plotly_json, stat_artifact, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            csv_id, question, answer, code,
            int(has_chart), int(has_model), model_path, plotly_json,
            stat_artifact, datetime.utcnow().isoformat(),
        ))
        con.execute(
            "UPDATE csv_sessions SET updated_at = ? WHERE csv_id = ?",
            (datetime.utcnow().isoformat(), csv_id)
        )
        con.commit()
        con.close()
    except Exception as e:
        print(f"[CSV_SESSION] Save turn failed: {e}")


def get_csv_chat_history(csv_id: str, since: Optional[str] = None) -> List[Dict]:
    """
    Return chat history for PDF export.

    since:  ISO-8601 timestamp string. When provided only turns created on or
            after this time are returned — used so the export shows only the
            *current* session and not turns accumulated from previous uploads
            of the same file (which share the same csv_id).
            If None, history is filtered automatically to turns after the
            session's own created_at timestamp in csv_sessions.
    """
    if not DB_PATH.exists():
        return []
    try:
        con = sqlite3.connect(DB_PATH)

        # Determine the cutoff: use the explicit `since` param, or fall back to
        # the session's created_at (set fresh on every ingest_csv call via
        # INSERT OR REPLACE).
        cutoff = since
        if not cutoff:
            row = con.execute(
                "SELECT created_at FROM csv_sessions WHERE csv_id = ?", (csv_id,)
            ).fetchone()
            cutoff = row[0] if row else None

        if cutoff:
            cur = con.execute(
                "SELECT question, answer, has_chart, has_model, model_path, "
                "plotly_json, stat_artifact, created_at "
                "FROM csv_history WHERE csv_id = ? AND created_at >= ? ORDER BY id",
                (csv_id, cutoff),
            )
        else:
            cur = con.execute(
                "SELECT question, answer, has_chart, has_model, model_path, "
                "plotly_json, stat_artifact, created_at "
                "FROM csv_history WHERE csv_id = ? ORDER BY id",
                (csv_id,),
            )

        rows = cur.fetchall()
        con.close()
        return [
            {
                "question":      r[0],
                "answer":        r[1],
                "has_chart":     bool(r[2]),
                "has_model":     bool(r[3]),
                "model_path":    r[4],
                "plotly_json":   r[5],
                "stat_artifact": r[6],
                "created_at":    r[7],
            }
            for r in rows
        ]
    except Exception as e:
        print(f"[CSV_SESSION] Get history failed: {e}")
        return []


def update_last_turn_answer(csv_id: str, question: str, answer: str):
    """
    Update the most-recent row for (csv_id, question) with the final streamed answer.
    Called by the streaming endpoint after all tokens have been sent, so that the
    PDF export shows the real NL answer instead of an empty string.
    """
    if not DB_PATH.exists() or not answer.strip():
        return
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute(
            """UPDATE csv_history SET answer = ?
               WHERE id = (
                   SELECT id FROM csv_history
                   WHERE csv_id = ? AND question = ?
                   ORDER BY id DESC LIMIT 1
               )""",
            (answer.strip(), csv_id, question),
        )
        con.commit()
        con.close()
    except Exception as e:
        print(f"[CSV_SESSION] update_last_turn_answer failed: {e}")


def delete_csv_session(csv_id: str):
    if not DB_PATH.exists():
        return
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM csv_history WHERE csv_id = ?", (csv_id,))
    con.execute("DELETE FROM csv_sessions WHERE csv_id = ?", (csv_id,))
    con.commit()
    con.close()
    print(f"[CSV_SESSION] Deleted session {csv_id}")


def list_csv_sessions() -> List[Dict]:
    """List all CSV sessions for UI display."""
    if not DB_PATH.exists():
        return []
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.execute("""
            SELECT s.csv_id, s.filename, s.shape_rows, s.shape_cols,
                   s.updated_at, COUNT(h.id) as turn_count
            FROM csv_sessions s
            LEFT JOIN csv_history h ON s.csv_id = h.csv_id
            GROUP BY s.csv_id
            ORDER BY s.updated_at DESC
        """)
        rows = cur.fetchall()
        con.close()
        return [
            {
                "csv_id":     r[0],
                "filename":   r[1],
                "rows":       r[2],
                "cols":       r[3],
                "updated_at": r[4],
                "turns":      r[5],
            }
            for r in rows
        ]
    except Exception as e:
        print(f"[CSV_SESSION] List failed: {e}")
        return []