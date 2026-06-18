"""
chat_exporter.py — Export Chat History as PDF for Bujji Babu

Generates a clean, well-formatted PDF of CSV agent chat history:
  - Proper markdown rendering (bold, headers, bullet lists, numbered lists)
  - Plotly charts embedded as PNG images via kaleido
  - Clean visual design matching the app aesthetic
"""

import os
import re
import io
import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Use absolute path anchored to this file's directory so the exports folder
# is always created in the project root regardless of CWD.
EXPORTS_DIR = Path(__file__).resolve().parent.parent / "exports"
EXPORTS_DIR.mkdir(exist_ok=True)


def _fmt_num(v) -> str:
    """Format a numeric value for a table cell — max 8 chars, sensible rounding."""
    if v is None or v == "":
        return ""
    try:
        f = float(v)
        if f == int(f) and abs(f) < 1e9:
            return str(int(f))
        if abs(f) >= 1e6:
            return f"{f:.2e}"
        if abs(f) >= 100:
            return f"{f:.1f}"
        return f"{f:.3g}"
    except (TypeError, ValueError):
        return str(v)[:10]


# ── Markdown → ReportLab XML ──────────────────────────────────────────────────

def _md_to_rl(text: str) -> str:
    """Convert basic markdown to ReportLab paragraph XML markup."""
    # Escape XML special chars first
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    # Bold **text** and __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__',     r'<b>\1</b>', text)
    # Italic *text*
    text = re.sub(r'\*(.+?)\*',     r'<i>\1</i>', text)
    # Inline code
    text = re.sub(r'`(.+?)`', r'<font face="Courier">\1</font>', text)
    return text


def _parse_md_blocks(text: str):
    """
    Parse markdown text into a list of block dicts:
      {"type": "h1"|"h2"|"h3"|"bullet"|"numbered"|"para", "text": str, "num": int}
    """
    blocks = []
    lines  = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if not stripped:
            i += 1
            continue
        # Headings
        if stripped.startswith('### '):
            blocks.append({"type": "h3", "text": stripped[4:].strip()})
        elif stripped.startswith('## '):
            blocks.append({"type": "h2", "text": stripped[3:].strip()})
        elif stripped.startswith('# '):
            blocks.append({"type": "h1", "text": stripped[2:].strip()})
        # Bullet list
        elif re.match(r'^[-*•]\s+', stripped):
            blocks.append({"type": "bullet", "text": re.sub(r'^[-*•]\s+', '', stripped)})
        # Numbered list
        elif re.match(r'^\d+[.)]\s+', stripped):
            m = re.match(r'^(\d+)[.)]\s+(.*)', stripped)
            blocks.append({"type": "numbered", "text": m.group(2), "num": int(m.group(1))})
        else:
            blocks.append({"type": "para", "text": stripped})
        i += 1
    return blocks


# ── Chart → PNG bytes ─────────────────────────────────────────────────────────

# Maximum seconds to wait for a single chart render before giving up.
# kaleido cold-starts on the first call (~3-5s), subsequent calls are faster (~0.5s).
_CHART_RENDER_TIMEOUT = 12  # seconds
# Max charts to embed as images — keeps export snappy even with long sessions.
MAX_EMBEDDED_CHARTS = 6


def _plotly_to_png(plotly_json_str: str) -> Optional[bytes]:
    """
    Render Plotly JSON → PNG via kaleido with a hard timeout.
    Returns None (shows placeholder text) if rendering takes too long or fails.
    Uses lower resolution than interactive mode — export doesn't need retina scale.
    """
    import threading

    result_box: list = [None]
    error_box:  list = [None]

    def _render():
        try:
            import plotly.io as pio
            fig = pio.from_json(plotly_json_str)
            is_splom = any(getattr(t, 'type', '') == 'splom' for t in fig.data)
            if is_splom:
                fig.update_layout(height=500, width=500,
                                  margin=dict(t=60, b=60, l=60, r=30))
                result_box[0] = pio.to_image(fig, format="png",
                                             width=500, height=500, scale=1.5)
            else:
                result_box[0] = pio.to_image(fig, format="png",
                                             width=520, height=300, scale=1.5)
        except Exception as e:
            error_box[0] = str(e)

    t = threading.Thread(target=_render, daemon=True)
    t.start()
    t.join(timeout=_CHART_RENDER_TIMEOUT)

    if t.is_alive():
        print(f"[EXPORT] Chart render timed out after {_CHART_RENDER_TIMEOUT}s — skipping")
        return None
    if error_box[0]:
        print(f"[EXPORT] Chart render failed: {error_box[0]}")
        return None
    return result_box[0]


# ── Main exporter ─────────────────────────────────────────────────────────────

def export_csv_chat_to_pdf(
    csv_id: str,
    filename: str,
    history: List[Dict],
    stats: Optional[Dict] = None,
) -> str:
    output_path = str(EXPORTS_DIR / f"{csv_id}_chat.pdf")

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer,
            HRFlowable, Image as RLImage, ListFlowable, ListItem,
            Table, TableStyle,
        )
        from reportlab.lib.enums import TA_LEFT, TA_CENTER

        W, H = A4

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=2.2*cm, leftMargin=2.2*cm,
            topMargin=2.2*cm,   bottomMargin=2.2*cm,
        )

        # ── Styles ────────────────────────────────────────────────────────────
        CORAL  = colors.HexColor('#CC6B52')
        INK    = colors.HexColor('#1a1a1a')
        MUTED  = colors.HexColor('#6b6b6b')
        RULE   = colors.HexColor('#e2e0db')
        BLUE   = colors.HexColor('#2c5f8a')
        BGSOFT = colors.HexColor('#faf9f7')

        title_s = ParagraphStyle('ExTitle',
            fontName='Helvetica-Bold', fontSize=20,
            textColor=INK, spaceAfter=4, spaceBefore=0)
        sub_s = ParagraphStyle('ExSub',
            fontName='Helvetica', fontSize=9,
            textColor=MUTED, spaceAfter=3, leading=14)
        q_s = ParagraphStyle('ExQ',
            fontName='Helvetica-Bold', fontSize=11,
            textColor=BLUE, spaceBefore=14, spaceAfter=5, leading=15)
        h1_s = ParagraphStyle('ExH1',
            fontName='Helvetica-Bold', fontSize=12,
            textColor=INK, spaceBefore=8, spaceAfter=3, leading=16)
        h2_s = ParagraphStyle('ExH2',
            fontName='Helvetica-Bold', fontSize=11,
            textColor=INK, spaceBefore=6, spaceAfter=2, leading=15)
        h3_s = ParagraphStyle('ExH3',
            fontName='Helvetica-BoldOblique', fontSize=10,
            textColor=MUTED, spaceBefore=4, spaceAfter=2, leading=14)
        para_s = ParagraphStyle('ExPara',
            fontName='Helvetica', fontSize=10,
            textColor=INK, spaceAfter=4, leading=15)
        bullet_s = ParagraphStyle('ExBullet',
            fontName='Helvetica', fontSize=10,
            textColor=INK, spaceAfter=2, leading=14,
            leftIndent=16, bulletIndent=4)
        tag_s = ParagraphStyle('ExTag',
            fontName='Helvetica', fontSize=8,
            textColor=colors.HexColor('#a0a0a0'), spaceAfter=2)
        ts_s = ParagraphStyle('ExTS',
            fontName='Helvetica', fontSize=8,
            textColor=colors.HexColor('#c0bdb8'), spaceAfter=0)

        def _style_map(btype):
            return {'h1': h1_s, 'h2': h2_s, 'h3': h3_s}.get(btype, para_s)

        story = []

        # ── Header ────────────────────────────────────────────────────────────
        story.append(Paragraph("Bujji Babu", title_s))
        story.append(Spacer(1, 4))
        story.append(Paragraph("CSV Analysis Chat", sub_s))
        story.append(Paragraph(f"Dataset: {filename}", sub_s))
        story.append(Paragraph(
            f"Exported: {datetime.now().strftime('%B %d, %Y at %H:%M')}",
            sub_s))
        if stats:
            story.append(Paragraph(
                f"Shape: {stats.get('rows','?'):,} rows x {stats.get('cols','?')} columns",
                sub_s))
        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=1.5,
                                color=CORAL, spaceAfter=14))

        if not history:
            story.append(Paragraph("No conversation history found.", para_s))
        else:
            _charts_rendered = 0   # track how many charts we've embedded
            for i, turn in enumerate(history, 1):
                # ── Question ─────────────────────────────────────────────────
                q_text = _md_to_rl(turn.get("question", ""))
                story.append(Paragraph(f"Q{i} &nbsp; {q_text}", q_s))

                # Timestamp
                if turn.get("created_at"):
                    try:
                        ts = datetime.fromisoformat(turn["created_at"])
                        story.append(Paragraph(
                            ts.strftime("%b %d, %Y  %H:%M UTC"), ts_s))
                    except Exception:
                        pass

                story.append(Spacer(1, 4))

                # ── Answer (markdown parsed) ──────────────────────────────────
                answer_raw = turn.get("answer", "")
                if answer_raw:
                    blocks = _parse_md_blocks(answer_raw)
                    bullet_group = []

                    def _flush_bullets():
                        if bullet_group:
                            story.append(ListFlowable(
                                [ListItem(Paragraph(_md_to_rl(b), para_s),
                                          bulletColor=CORAL, leftIndent=18,
                                          bulletOffsetY=-1)
                                 for b in bullet_group],
                                bulletType='bullet', start='bulletchar',
                                bulletChar=u'•',
                                leftIndent=12, spaceBefore=2, spaceAfter=4,
                            ))
                            bullet_group.clear()

                    num_group = []

                    def _flush_numbered():
                        if num_group:
                            story.append(ListFlowable(
                                [ListItem(Paragraph(_md_to_rl(txt), para_s),
                                          bulletColor=CORAL, leftIndent=24)
                                 for txt in num_group],
                                bulletType='1', leftIndent=12,
                                spaceBefore=2, spaceAfter=4,
                            ))
                            num_group.clear()

                    for blk in blocks:
                        btype = blk["type"]
                        if btype == "bullet":
                            _flush_numbered()
                            bullet_group.append(blk["text"])
                        elif btype == "numbered":
                            _flush_bullets()
                            num_group.append(blk["text"])
                        else:
                            _flush_bullets()
                            _flush_numbered()
                            sty = _style_map(btype)
                            story.append(Paragraph(_md_to_rl(blk["text"]), sty))

                    _flush_bullets()
                    _flush_numbered()

                # ── Embedded chart ────────────────────────────────────────────
                pj = turn.get("plotly_json")
                if pj:
                    if _charts_rendered < MAX_EMBEDDED_CHARTS:
                        png_bytes = _plotly_to_png(pj)
                        if png_bytes:
                            story.append(Spacer(1, 6))
                            img_buf = io.BytesIO(png_bytes)
                            # Scale to page width minus margins
                            usable_w = W - 4.4*cm
                            rl_img = RLImage(img_buf, width=usable_w,
                                             height=usable_w * 0.57)
                            story.append(rl_img)
                            story.append(Spacer(1, 4))
                            _charts_rendered += 1
                        else:
                            story.append(Paragraph("[Chart — render unavailable]", tag_s))
                    else:
                        story.append(Paragraph(
                            "[Chart — not embedded: export limit reached]", tag_s
                        ))

                # ── Stat artifact table ───────────────────────────────────────
                sa_raw = turn.get("stat_artifact")
                if sa_raw:
                    try:
                        sa = json.loads(sa_raw) if isinstance(sa_raw, str) else sa_raw
                        numeric = sa.get("numeric", {})
                        if numeric:
                            story.append(Spacer(1, 6))
                            story.append(Paragraph("Column Statistics", h3_s))
                            header = ["Column", "Mean", "Std", "Min", "Median", "Max", "Nulls"]
                            tdata = [header]
                            for col, v in list(numeric.items())[:12]:
                                tdata.append([
                                    col[:20],
                                    _fmt_num(v.get("mean")),
                                    _fmt_num(v.get("std")),
                                    _fmt_num(v.get("min")),
                                    _fmt_num(v.get("median")),
                                    _fmt_num(v.get("max")),
                                    str(v.get("nulls", "")),
                                ])
                            # Give the Column name column more space; numeric cols share the rest
                            usable = W - 4.4*cm
                            col_w = [usable * 0.30] + [usable * 0.70 / 6] * 6
                            tbl = Table(tdata, colWidths=col_w, repeatRows=1)
                            tbl.setStyle(TableStyle([
                                ('BACKGROUND',  (0,0), (-1,0), CORAL),
                                ('TEXTCOLOR',   (0,0), (-1,0), colors.white),
                                ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
                                ('FONTSIZE',    (0,0), (-1,-1), 7.5),
                                ('FONTNAME',    (0,1), (-1,-1), 'Helvetica'),
                                ('ROWBACKGROUNDS', (0,1),(-1,-1), [colors.white, colors.HexColor('#faf9f7')]),
                                ('GRID',        (0,0), (-1,-1), 0.4, colors.HexColor('#e2e0db')),
                                ('TOPPADDING',  (0,0), (-1,-1), 3),
                                ('BOTTOMPADDING',(0,0),(-1,-1), 3),
                                ('LEFTPADDING', (0,0), (-1,-1), 4),
                                ('RIGHTPADDING',(0,0), (-1,-1), 4),
                                ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
                            ]))
                            story.append(tbl)
                            story.append(Spacer(1, 4))
                    except Exception as e:
                        print(f"[EXPORT] Stat artifact render failed: {e}")

                # ── Tags ──────────────────────────────────────────────────────
                tags = []
                if turn.get("has_model") and turn.get("model_path"):
                    model_name = Path(turn["model_path"]).stem.replace("_", " ").title()
                    tags.append(f"Model: {model_name}")
                if turn.get("has_chart") and not pj:
                    tags.append("Chart generated")
                if tags:
                    story.append(Paragraph(" · ".join(tags), tag_s))

                story.append(Spacer(1, 6))

                if i < len(history):
                    story.append(HRFlowable(width="100%", thickness=0.5,
                                            color=RULE, spaceAfter=10))

        doc.build(story)
        print(f"[EXPORT] PDF written: {output_path}")
        return output_path

    except Exception as exc:
        print(f"[EXPORT] PDF generation failed: {exc}")
        import traceback; traceback.print_exc()
        return export_csv_chat_to_html(csv_id, filename, history, stats)


# ── HTML fallback ─────────────────────────────────────────────────────────────

def export_csv_chat_to_html(
    csv_id: str,
    filename: str,
    history: List[Dict],
    stats: Optional[Dict] = None,
) -> str:
    output_path = str(EXPORTS_DIR / f"{csv_id}_chat.html")

    def _esc(s): return s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

    rows_html = ""
    for i, turn in enumerate(history, 1):
        q = _esc(turn.get("question", ""))
        a = _esc(turn.get("answer", ""))
        tags = []
        if turn.get("has_model") and turn.get("model_path"):
            tags.append(Path(turn["model_path"]).stem.replace("_"," ").title())
        if turn.get("has_chart"):
            tags.append("Chart generated")
        tag_html = (f'<div style="font-size:11px;color:#a8a8a8;margin-top:6px">'
                    + " · ".join(tags) + '</div>') if tags else ""
        ts = turn.get("created_at","")
        rows_html += f"""
        <div style="margin-bottom:24px;padding-bottom:20px;border-bottom:1px solid #f0ede8">
          <div style="font-size:10px;color:#b0aca6;margin-bottom:4px">Q{i}{' &nbsp;|&nbsp; ' + ts[:16] if ts else ''}</div>
          <div style="font-weight:700;color:#2c5f8a;font-size:13px;margin-bottom:10px">{q}</div>
          <div style="color:#1a1a1a;line-height:1.7;white-space:pre-wrap;font-size:12px">{a}</div>
          {tag_html}
        </div>"""

    stats_html = (f'<div style="color:#6b6b6b;font-size:12px">'
                  f'{stats.get("rows","?"):,} rows &times; {stats.get("cols","?")} cols</div>'
                  ) if stats else ""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Bujji Babu — {_esc(filename)}</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 48px auto;
         padding: 0 24px; color: #1a1a1a; background: #faf9f7; }}
  h1   {{ font-size: 26px; font-weight: 800; margin-bottom: 4px; }}
  hr   {{ border: none; border-top: 2px solid #CC6B52; margin: 16px 0 24px; }}
</style></head><body>
<h1>Bujji Babu</h1>
<div style="color:#6b6b6b;margin-bottom:2px">CSV Analysis Chat &mdash; {_esc(filename)}</div>
{stats_html}
<div style="color:#a0a0a0;margin-bottom:20px;font-size:12px">
  Exported {datetime.now().strftime('%B %d, %Y at %H:%M')}</div>
<hr>
{rows_html}
</body></html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[EXPORT] HTML written: {output_path}")
    return output_path
