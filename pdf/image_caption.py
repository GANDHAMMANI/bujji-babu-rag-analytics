"""
image_caption.py — Image Caption & Context Extraction for Bujji Babu

Problem it solves:
  VLM sees an image alone and guesses: "This appears to be a professional photo"
  With caption context it knows: "Figure 2: Dr. Navin Kumar Manaswi, Author"

How it works:
  PyMuPDF gives us:
  - Exact image bounding box (x0, y0, x1, y1) on the page
  - All text blocks on the same page with their bounding boxes

  We find text blocks that are:
  1. Directly below the image  (caption — most reliable)
  2. Directly above the image  (title/label)
  3. On the same line          (inline label)
  4. Nearest paragraph         (surrounding context)

  Priority: caption below > label above > nearest paragraph
"""

import fitz  # PyMuPDF
import re
from typing import Optional, Dict, List, Tuple


# ── Proximity thresholds (in PDF points, 1pt ≈ 0.35mm) ───────────────────────

CAPTION_BELOW_GAP  = 60   # max vertical gap below image for a caption
CAPTION_ABOVE_GAP  = 40   # max vertical gap above image for a title/label
SIDE_GAP           = 80   # max horizontal gap for side labels
NEARBY_PARA_GAP    = 120  # fallback — nearest text block within this range
MIN_CAPTION_LEN    = 8    # ignore very short text fragments
MAX_CAPTION_LEN    = 400  # truncate very long surrounding text


# ── Caption patterns (figure labels, table labels, etc.) ─────────────────────

CAPTION_PATTERNS = [
    r"^(fig(ure)?\.?\s*\d+)",          # Figure 1, Fig. 2
    r"^(table\.?\s*\d+)",              # Table 1
    r"^(photo|image|picture|plate)",   # Photo 1
    r"^(author|about the author)",     # Author bio
    r"^(source|courtesy)",             # Source credit
]


def _is_caption_text(text: str) -> bool:
    """Check if text looks like a figure/image caption."""
    t = text.lower().strip()
    return any(re.match(p, t) for p in CAPTION_PATTERNS)


def _clean_text(text: str) -> str:
    """Normalise whitespace and remove junk characters."""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\x20-\x7E\u00C0-\u024F]', '', text)
    return text[:MAX_CAPTION_LEN]


def _rect_vertical_gap(img_rect: fitz.Rect, text_rect: fitz.Rect) -> float:
    """Vertical gap between image bottom and text top (positive = below image)."""
    return text_rect.y0 - img_rect.y1


def _rect_above_gap(img_rect: fitz.Rect, text_rect: fitz.Rect) -> float:
    """Vertical gap between text bottom and image top (positive = above image)."""
    return img_rect.y0 - text_rect.y1


def _horizontal_overlap(img_rect: fitz.Rect, text_rect: fitz.Rect) -> float:
    """Fraction of text block that horizontally overlaps with image."""
    overlap_x = min(img_rect.x1, text_rect.x1) - max(img_rect.x0, text_rect.x0)
    if overlap_x <= 0:
        return 0.0
    text_width = text_rect.x1 - text_rect.x0
    return overlap_x / max(text_width, 1)


# ── Main extraction function ──────────────────────────────────────────────────

def extract_image_context(
    pdf_path: str,
    page_num: int,       # 1-indexed
    img_xref: int,       # PyMuPDF xref for the image
) -> Dict[str, Optional[str]]:
    """
    Extract text context surrounding an image on a page.

    Returns:
        {
            "caption":   str or None  — text directly below/above image
            "context":   str or None  — nearest paragraph text
            "combined":  str or None  — best single string for VLM prompt
        }
    """
    result = {"caption": None, "context": None, "combined": None}

    try:
        doc  = fitz.open(pdf_path)
        page = doc[page_num - 1]  # convert to 0-indexed

        # ── Get image bounding box ────────────────────────────────────────────
        img_rect = None
        for img_info in page.get_image_info(xrefs=True):
            if img_info.get("xref") == img_xref:
                bbox     = img_info.get("bbox")
                img_rect = fitz.Rect(bbox) if bbox else None
                break

        if img_rect is None or img_rect.is_empty:
            doc.close()
            return result

        # ── Get full page text + all blocks ──────────────────────────────────
        full_page_text = _clean_text(page.get_text())
        blocks = page.get_text("blocks")

        text_blocks = []
        for b in blocks:
            if b[6] != 0:  # skip image blocks
                continue
            text = _clean_text(b[4])
            if len(text) < MIN_CAPTION_LEN:
                continue
            rect = fitz.Rect(b[0], b[1], b[2], b[3])
            text_blocks.append({"rect": rect, "text": text})

        if not text_blocks:
            # Fall back to full page text as context
            if full_page_text:
                result["context"] = full_page_text[:400]
                result["combined"] = f"Page text: {full_page_text[:400]}"
            doc.close()
            return result

        img_center_y = (img_rect.y0 + img_rect.y1) / 2
        img_center_x = (img_rect.x0 + img_rect.x1) / 2

        # ── Priority 1: Caption directly below image ──────────────────────────
        below_candidates = []
        for tb in text_blocks:
            v_gap     = _rect_vertical_gap(img_rect, tb["rect"])
            h_overlap = _horizontal_overlap(img_rect, tb["rect"])
            if 0 <= v_gap <= CAPTION_BELOW_GAP and h_overlap > 0.2:
                below_candidates.append((v_gap, tb["text"]))

        if below_candidates:
            below_candidates.sort(key=lambda x: x[0])
            result["caption"] = below_candidates[0][1]

        # ── Priority 2: Label directly above image ────────────────────────────
        if not result["caption"]:
            above_candidates = []
            for tb in text_blocks:
                v_gap     = _rect_above_gap(img_rect, tb["rect"])
                h_overlap = _horizontal_overlap(img_rect, tb["rect"])
                if 0 <= v_gap <= CAPTION_ABOVE_GAP and h_overlap > 0.2:
                    above_candidates.append((v_gap, tb["text"]))

            if above_candidates:
                above_candidates.sort(key=lambda x: x[0])
                result["caption"] = above_candidates[0][1]

        # ── Priority 3: Text BESIDE image (same vertical band) ───────────────
        # Critical for "About the Author" pages where bio text is next to photo
        beside_candidates = []
        for tb in text_blocks:
            tb_center_y = (tb["rect"].y0 + tb["rect"].y1) / 2
            # Vertically overlapping with image
            v_overlap = (min(img_rect.y1, tb["rect"].y1) - max(img_rect.y0, tb["rect"].y0))
            if v_overlap > 20:  # at least 20pt vertical overlap
                # Text is to the right or left of image
                h_gap_right = tb["rect"].x0 - img_rect.x1
                h_gap_left  = img_rect.x0 - tb["rect"].x1
                if 0 <= h_gap_right <= 150 or 0 <= h_gap_left <= 150:
                    beside_candidates.append((abs(tb_center_y - img_center_y), tb["text"]))

        if beside_candidates:
            beside_candidates.sort(key=lambda x: x[0])
            # Take up to first 2 beside blocks (may span multiple paragraphs)
            beside_text = " ".join(b[1] for b in beside_candidates[:2])
            if not result["caption"]:
                result["caption"] = beside_text[:300]
            else:
                result["context"] = beside_text[:300]

        # ── Priority 4: Nearest paragraph by vertical center ─────────────────
        if not result["context"]:
            nearest = None
            nearest_dist = float("inf")
            for tb in text_blocks:
                text_center_y = (tb["rect"].y0 + tb["rect"].y1) / 2
                dist = abs(text_center_y - img_center_y)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = tb["text"]

            if nearest and nearest_dist < NEARBY_PARA_GAP:
                result["context"] = nearest

        # ── Priority 5: Full page text fallback ───────────────────────────────
        # For pages like "About the Author" — grab everything
        if not result["caption"] and not result["context"]:
            result["context"] = full_page_text[:400]

        # ── Build combined string for VLM ─────────────────────────────────────
        parts = []
        if result["caption"]:
            parts.append(f"Caption/bio: {result['caption'][:300]}")
        if result["context"] and result["context"] != result["caption"]:
            parts.append(f"Nearby text: {result['context'][:200]}")
        # Always include page heading if it mentions "author" or "reviewer"
        first_line = full_page_text.split("\n")[0].strip() if full_page_text else ""
        if any(kw in first_line.lower() for kw in ["author","reviewer","about","editor"]):
            parts.insert(0, f"Page heading: {first_line[:80]}")

        result["combined"] = " | ".join(parts) if parts else None

        doc.close()

    except Exception as e:
        print(f"[CAPTION] Failed for page {page_num} xref {img_xref}: {e}")

    return result


# ── Batch extraction for all images in a PDF ─────────────────────────────────

def extract_all_captions(
    pdf_path: str,
    images: List[Dict],
) -> List[Dict]:
    """
    Add caption/context to each image dict in the images list.
    Modifies in place and returns the updated list.

    Each image dict gains:
        "caption":  str or None
        "context":  str or None
        "combined": str or None  ← use this in vision.py
    """
    try:
        doc = fitz.open(pdf_path)

        # Build xref → image mapping for fast lookup
        page_xrefs: Dict[int, List[int]] = {}
        for page_num in range(len(doc)):
            page = doc[page_num]
            xrefs = [img[0] for img in page.get_images(full=True)]
            page_xrefs[page_num + 1] = xrefs

        doc.close()

    except Exception as e:
        print(f"[CAPTION] Could not open PDF: {e}")
        return images

    for img in images:
        page_num = img.get("page", 1)
        filename = img.get("filename", "")

        # Recover xref from page image list
        try:
            doc  = fitz.open(pdf_path)
            page = doc[page_num - 1]
            page_imgs = page.get_images(full=True)
            doc.close()

            # Match by filename index
            img_index = int(re.search(r'img(\d+)', filename).group(1)) - 1
            if img_index < len(page_imgs):
                xref = page_imgs[img_index][0]
                ctx  = extract_image_context(pdf_path, page_num, xref)
                img["caption"]  = ctx["caption"]
                img["context"]  = ctx["context"]
                img["combined"] = ctx["combined"]
            else:
                img["caption"]  = None
                img["context"]  = None
                img["combined"] = None

        except Exception as e:
            print(f"[CAPTION] Skipped {filename}: {e}")
            img["caption"]  = None
            img["context"]  = None
            img["combined"] = None

    extracted = sum(1 for img in images if img.get("combined"))
    print(f"[CAPTION] Extracted context for {extracted}/{len(images)} images")
    return images