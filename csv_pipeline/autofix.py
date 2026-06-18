"""
autofix.py — Surgical Auto-Fix for Bujji Babu CSV Agent

Surgical approach:
  1. Parse the traceback to find exact error line number
  2. Extract only the broken snippet (±5 lines around error)
  3. Send ONLY that snippet + error to LLM for targeted fix
  4. Splice the fixed snippet back into original code

Rule-based fixes are tried first (zero API cost):
  - fig.show() / plt.show() leak → replace with PLOTLY_JSON print
  - Unclosed parentheses / brackets → balance them
  - NameError: counts_* → safe_value_counts injection
  - KeyError on column → guarded access
  - trendline='ols' statsmodels missing → strip trendline
  - ModuleNotFoundError → strip broken import
"""

import re
from typing import Optional, Tuple
from dotenv import load_dotenv
import os

load_dotenv()
from llm_client import get_client as _get_client, get_model as _get_model
_groq = _get_client()


# ── Rule-based fixes (no API call) ───────────────────────────────────────────

def _rule_based_fix(code: str, error: str) -> Optional[str]:
    """
    Apply deterministic fixes for common known errors.
    Returns fixed code or None if no rule matched.
    """
    # ── Always replace fig.show() — it hangs headless servers silently ──────
    # Do NOT wait for an error message (it often doesn't produce one)
    if "fig.show()" in code:
        fixed = code.replace(
            "fig.show()",
            "print('PLOTLY_JSON:' + fig.to_json())"
        )
        if fixed != code:
            print("[AUTOFIX] Rule: replaced fig.show() with PLOTLY_JSON print")
            return fixed

    # ── plt.show() ─────────────────────────────────────────────────────────
    if "plt.show()" in code:
        fixed = code.replace("plt.show()", "# plt.show() disabled")
        print("[AUTOFIX] Rule: disabled plt.show()")
        return fixed

    # ── Unclosed parenthesis — most common LLM generation error ─────────────
    if "was never closed" in error or ("SyntaxError" in error and "paren" in error.lower()):
        open_p  = code.count("(")
        close_p = code.count(")")
        diff    = open_p - close_p
        if diff > 0:
            # Insert closing parens before the last print statement, not at file end
            lines = code.rstrip().splitlines()
            fixed = "\n".join(lines) + (")" * diff)
            print(f"[AUTOFIX] Rule: added {diff} missing closing paren(s)")
            return fixed

    # ── SyntaxError — balance all bracket types ──────────────────────────────
    if "SyntaxError" in error:
        fixed = code
        diff_p = fixed.count("(") - fixed.count(")")
        diff_b = fixed.count("[") - fixed.count("]")
        diff_c = fixed.count("{") - fixed.count("}")
        lines = fixed.rstrip().splitlines()
        tail = ""
        if diff_p > 0: tail += ")" * diff_p
        if diff_b > 0: tail += "]" * diff_b
        if diff_c > 0: tail += "}" * diff_c
        if tail:
            fixed = "\n".join(lines) + tail
            print(f"[AUTOFIX] Rule: balanced brackets (p={diff_p} b={diff_b} c={diff_c})")
            return fixed

    # ── NameError: counts_* not defined ─────────────────────────────────────
    # Note: Python NameError says "name 'counts_X' is not defined" — match that
    if "NameError" in error and "counts_" in error:
        col_match = re.search(r"counts_(\w+)", error)
        if not col_match:
            col_match = re.search(r"counts_(\w+)", code)
        if col_match:
            col_var = col_match.group(0)          # e.g. counts_General_Appearance
            col_name = col_match.group(1).replace("_", " ")  # e.g. General Appearance
            injection = (
                f"# Auto-injected: recompute missing counts variable\n"
                f"_sc = [c for c in df.columns if c.lower().replace(' ','_') == '{col_match.group(1).lower()}']\n"
                f"{col_var} = safe_value_counts(df[_sc[0]] if _sc else df.iloc[:,0])\n"
            )
            fixed = injection + code
            print(f"[AUTOFIX] Rule: injected safe_value_counts for {col_var}")
            return fixed

    # ── KeyError on column access ────────────────────────────────────────────
    if "KeyError" in error:
        col_match = re.search(r"KeyError: ['\"](.+?)['\"]", error)
        if col_match:
            bad_col = col_match.group(1)
            fixed   = code.replace(
                f"df['{bad_col}']",
                f"df['{bad_col}'] if '{bad_col}' in df.columns else df.iloc[:,-1]"
            ).replace(
                f'df["{bad_col}"]',
                f"df['{bad_col}'] if '{bad_col}' in df.columns else df.iloc[:,-1]"
            )
            print(f"[AUTOFIX] Rule: guarded KeyError for column '{bad_col}'")
            return fixed

    # ── trendline='ols' requires statsmodels — strip it ─────────────────────
    if "statsmodels" in error or ("trendline" in code and "make_trace" in error):
        fixed = re.sub(r",?\s*trendline=['\"]\w+['\"]", "", code)
        print("[AUTOFIX] Rule: removed trendline parameter")
        return fixed

    # ── ModuleNotFoundError — strip the import ───────────────────────────────
    if "ModuleNotFoundError" in error:
        mod_match = re.search(r"No module named '(.+?)'", error)
        if mod_match:
            mod = mod_match.group(1)
            lines = [l for l in code.splitlines()
                     if f"import {mod}" not in l and f"from {mod}" not in l]
            print(f"[AUTOFIX] Rule: removed broken import '{mod}'")
            return "\n".join(lines)

    # ── px.bar x='index' / wrong column names ───────────────────────────────
    # This catches the "To use the index, pass it in directly" Plotly error
    if "index" in error.lower() and "px.bar" in code and "To use the index" in error:
        fixed = re.sub(r"(px\.bar\(counts_\w+[^)]*?)x=['\"]index['\"]", r"\1x='value'", code)
        fixed = re.sub(r"(px\.bar\(counts_\w+[^)]*?)y=['\"](?!count)[^'\"]+['\"]", r"\1y='count'", fixed)
        if fixed != code:
            print("[AUTOFIX] Rule: fixed x='index' → x='value', y→'count' in px.bar")
            return fixed

    # ── squared=False removed in sklearn 1.4 ────────────────────────────────
    if "squared" in error and "mean_squared_error" in code:
        fixed = re.sub(
            r"mean_squared_error\(([^)]+),\s*squared=False\)",
            r"mean_squared_error(\1)**0.5",
            code,
        )
        if fixed != code:
            print("[AUTOFIX] Rule: replaced squared=False with **0.5")
            return fixed

    return None


# ── Traceback parser ──────────────────────────────────────────────────────────

def _count_header_lines(code_before_user: str) -> int:
    """Count the lines in the script header that precede user code."""
    return len(code_before_user.splitlines())


def _extract_error_line(error: str, header_lines: int) -> Tuple[Optional[int], str]:
    """
    Parse traceback to find the line number in user code that caused the error.
    Returns (line_number_in_user_code, error_type_message).

    The header_lines count is computed dynamically from the actual script header
    so the offset is always accurate regardless of dataset column count.
    """
    # Find all "line N" references in traceback
    line_matches = re.findall(r'line (\d+)', error)
    if not line_matches:
        return None, error.strip().split('\n')[-1]

    # Use the last line number (innermost frame = actual error site)
    raw_line = int(line_matches[-1])
    user_line = max(1, raw_line - header_lines)

    # Get the actual error type + message (last line of traceback)
    error_lines = error.strip().split('\n')
    error_msg = error_lines[-1].strip() if error_lines else "Unknown error"

    return user_line, error_msg


def _extract_snippet(code: str, error_line: int, context: int = 5) -> Tuple[str, int, int]:
    """
    Extract code lines around the error line.
    Returns (snippet_with_numbers, start_line_1indexed, end_line_1indexed).
    """
    lines      = code.splitlines()
    total      = len(lines)
    error_line = max(1, min(error_line, total))
    start      = max(0, error_line - context - 1)
    end        = min(total, error_line + context)
    if start >= end:
        start = max(0, end - 3)
    snippet = "\n".join(
        f"{start+i+1}: {line}"
        for i, line in enumerate(lines[start:end])
    )
    return snippet, start + 1, max(end, start + 1)


def _splice_fix(original_code: str, fixed_snippet: str, start_line: int, end_line: int) -> str:
    """Replace lines start_line..end_line (1-indexed) in original_code with fixed_snippet."""
    lines       = original_code.splitlines()
    before      = lines[:start_line - 1]
    after       = lines[end_line:]
    fixed_lines = fixed_snippet.splitlines()
    return "\n".join(before + fixed_lines + after)


# ── LLM surgical fix ──────────────────────────────────────────────────────────

def _llm_fix_snippet(
    snippet: str,
    error_msg: str,
    df_columns: list,
    eda_summary: str,
) -> str:
    """
    Send only the broken snippet to LLM for targeted fix.
    Much cheaper and more accurate than sending the full code.
    """
    resp = _groq.chat.completions.create(
        model=_get_model("strong"),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Python debugging expert. Fix ONLY the broken lines shown.\n"
                    "Return ONLY the corrected Python lines — no line numbers, no explanation, no markdown.\n\n"
                    f"Available columns (use exact spelling): {df_columns}\n"
                    f"Dataset info: {eda_summary[:300]}\n\n"
                    "Rules:\n"
                    "- NEVER use fig.show() — always use print('PLOTLY_JSON:' + fig.to_json())\n"
                    "- For bar charts on value_counts: use pre-computed counts_ColName with x='value', y='count'\n"
                    "- NEVER use x='index' in px.bar — always x='value'\n"
                    "- Guard column access with: if 'col' in df.columns\n"
                    "- NEVER use Pipeline or ColumnTransformer\n"
                    "- RMSE: use mean_squared_error(y_test, pred)**0.5 (not squared=False)\n"
                    "- cross_val_score requires estimators, not plain dicts\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Error: {error_msg}\n\n"
                    f"Broken code (line numbers shown for reference only):\n{snippet}\n\n"
                    "Return only the fixed Python lines (no numbers, no explanation):"
                ),
            },
        ],
        max_tokens=500,
        temperature=0.05,
    )
    raw = resp.choices[0].message.content.strip()
    # Strip fenced markdown if LLM added it
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
    if fenced:
        raw = "\n".join(fenced).strip()
    # Strip any remaining line-number prefixes (e.g. "12: code here")
    lines = [re.sub(r'^\d+:\s*', '', line) for line in raw.splitlines()]
    return "\n".join(lines)


def _is_valid_python(code: str) -> bool:
    """Quick compile check — catches prose/explanation returned by LLM."""
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def surgical_fix(
    code: str,
    error: str,
    df_columns: list,
    eda_summary: str,
    header_lines: int = 0,
) -> str:
    """
    Main entry point — fix broken code surgically.

    Strategy:
    1. Try rule-based fix (free, instant)
    2. Parse traceback → find error line (using dynamic header_lines offset)
    3. Extract snippet (±5 lines)
    4. LLM fixes snippet only
    5. Splice back into original code + validate syntax
    6. Falls back to full-code LLM rewrite if splicing fails / invalid
    """
    # ── Rule-based first (no API) ─────────────────────────────────────────────
    rule_fix = _rule_based_fix(code, error)
    if rule_fix:
        return rule_fix

    # ── Surgical LLM fix ──────────────────────────────────────────────────────
    error_line, error_msg = _extract_error_line(error, header_lines)
    print(f"[AUTOFIX] Error at line ~{error_line}: {error_msg[:80]}")

    if error_line:
        snippet, start, end = _extract_snippet(code, error_line, context=5)
        print(f"[AUTOFIX] Fixing snippet lines {start}-{end} ({end-start+1} lines)")

        fixed_snippet = _llm_fix_snippet(snippet, error_msg, df_columns, eda_summary)

        try:
            spliced = _splice_fix(code, fixed_snippet, start, end)
            if spliced.strip() and len(spliced) > 20 and _is_valid_python(spliced):
                print(f"[AUTOFIX] Surgical fix applied ({len(fixed_snippet)} chars)")
                return spliced
            else:
                print("[AUTOFIX] Splice invalid, falling back to full rewrite")
        except Exception as e:
            print(f"[AUTOFIX] Splice failed ({e}), falling back to full rewrite")

    # ── Fallback: full rewrite ────────────────────────────────────────────────
    print("[AUTOFIX] Falling back to full LLM rewrite")
    resp = _groq.chat.completions.create(
        model=_get_model("strong"),
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a Python debugging expert. Fix the broken code below.\n"
                    f"Output ONLY raw Python — no markdown fences, no explanation.\n"
                    f"Available columns: {df_columns}\n"
                    f"Dataset info: {eda_summary[:300]}\n\n"
                    "Rules:\n"
                    "- NEVER use fig.show() — use print('PLOTLY_JSON:' + fig.to_json())\n"
                    "- px.bar on value_counts: x='value', y='count' (NEVER x='index')\n"
                    "- RMSE: mean_squared_error(y_test, pred)**0.5\n"
                    "- Do NOT add any import statements\n"
                ),
            },
            {
                "role": "user",
                "content": f"Broken code:\n{code}\n\nError:\n{error}\n\nFixed code:"
            },
        ],
        max_tokens=2000,
        temperature=0.05,
    )
    raw = resp.choices[0].message.content.strip()
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
    result = "\n".join(fenced).strip() if fenced else raw

    # Validate — if LLM returned prose instead of code, return original code
    # so the next retry attempt uses the original rather than garbled prose
    if not _is_valid_python(result):
        print("[AUTOFIX] Full rewrite returned invalid Python — keeping original")
        return code

    return result
