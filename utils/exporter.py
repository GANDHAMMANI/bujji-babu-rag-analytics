"""
exporter.py — Dataset Export for Bujji Babu CSV Agent

Handles:
  - Export current (possibly filtered) dataframe as CSV
  - Export after cleaning operations
  - Route /export-csv endpoint logic

Each export gets a unique timestamp-based filename so previous exports
are never served in place of a fresh one.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd

# Use an absolute path anchored to the directory of this file so exports always
# resolve to the same location regardless of the process working directory.
EXPORT_DIR = Path(__file__).resolve().parent.parent / "exports"
EXPORT_DIR.mkdir(exist_ok=True)


def _ts() -> str:
    """Short timestamp string: YYYYMMDD_HHMMSS"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def register_for_export(csv_id: str, df: pd.DataFrame, label: str = "cleaned") -> str:
    """
    Save a dataframe to a timestamped CSV and return the export path string.
    """
    filename = f"{csv_id}_{label}_{_ts()}.csv"
    export_path = EXPORT_DIR / filename
    df.to_csv(export_path, index=False)
    print(f"[EXPORT] Saved {len(df)} rows → {export_path}")
    return str(export_path)


def get_export_path(csv_id: str, label: str = "cleaned") -> Optional[str]:
    """
    Return the MOST RECENT export matching csv_id + label, or None.
    Used as a fallback for the plain /download-csv/{csv_id} endpoint.
    """
    candidates = sorted(
        EXPORT_DIR.glob(f"{csv_id}_{label}_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return str(candidates[0])
    # Legacy: try the old non-timestamped filename
    legacy = EXPORT_DIR / f"{csv_id}_{label}.csv"
    return str(legacy) if legacy.exists() else None


def is_export_query(question: str) -> bool:
    """Detect if user wants to export/download data."""
    kw = [
        "export", "download", "save", "save as csv",
        "export as csv", "download csv", "get the data",
        "export the filtered", "export the cleaned",
        "export result", "save result",
    ]
    return any(k in question.lower() for k in kw)


def get_export_code(csv_id: str, label: str = "filtered") -> str:
    """
    Return Python code that saves the current df (or filtered_df if exists)
    using a timestamped filename and signals the export path via EXPORT_SAVED:.
    """
    export_dir = str(EXPORT_DIR).replace("\\", "/")

    return f"""
# ── Export dataset ────────────────────────────────────────────────────────────
import os, datetime
os.makedirs('{export_dir}', exist_ok=True)
_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
_export_filename = f'{csv_id}_{label}_{{_ts}}.csv'
_export_path = '{export_dir}/' + _export_filename
_export_target = filtered_df if 'filtered_df' in dir() else df
_export_target.to_csv(_export_path, index=False)
print(f'EXPORT_SAVED:{{_export_path}}')
print(f'Exported {{len(_export_target)}} rows x {{len(_export_target.columns)}} columns')
"""
