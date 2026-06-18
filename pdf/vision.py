"""
vision.py — Image Scoring & Reranking for Bujji Babu (v2)

Key upgrade:
  Images now carry a "combined" field extracted by image_caption.py
  containing the actual caption text and nearby paragraph from the PDF.

  Old approach:  VLM sees image + generated answer → guesses who/what it is
  New approach:  VLM sees image + PDF caption + nearby text + answer context
                 → correctly identifies person/figure from document's own text

  This fixes:
  - Two photos getting the same wrong name
  - Author photos being described as "professional headshot of unknown person"
  - Figures being described as "a diagram" without knowing what it shows
"""

import base64
import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

from llm_client import vision_chat as _vision_chat, get_vlm_model
client = None  # managed by llm_client
# NOTE: do NOT cache get_vlm_model() here — mode can switch at runtime
# and the module-level value would become stale. Call get_vlm_model()
# inside functions that need it, or use _vision_chat() which resolves the
# current mode dynamically.


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _build_context_block(
    image_caption: str | None,
    answer_context: str,
) -> str:
    """
    Build the context block for the VLM prompt.
    Prioritises PDF-extracted caption over generated answer context.
    Extracts person names from answer context for explicit name matching.
    """
    import re
    parts = []

    if image_caption:
        parts.append(f'PDF caption/nearby text: "{image_caption}"')

    if answer_context:
        ctx = answer_context[:300]
        parts.append(f'Document answer context: "{ctx}"')

        # Extract person names (Title Case sequences of 2-4 words) for VLM hint
        names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', answer_context)
        # Filter out common non-name title case phrases
        stopwords = {'The', 'This', 'Deep', 'Learning', 'Python', 'Applications'}
        person_names = [n for n in names if not any(w in stopwords for w in n.split())]
        if person_names:
            unique_names = list(dict.fromkeys(person_names))[:2]
            parts.append(f'Person name(s) to identify: {", ".join(unique_names)}')

    return "\n".join(parts) if parts else ""


def _describe_and_score_image(
    image_path: str,
    question: str,
    answer_context: str = "",
    image_caption: str | None = None,
) -> dict:
    """
    Single vision call that scores relevance and generates a description.

    Priority for identification:
    1. PDF caption (extracted from surrounding text) — ground truth
    2. Answer context (from generated answer) — secondary
    3. Visual content alone — last resort
    """
    try:
        b64  = encode_image_to_base64(image_path)
        ext  = image_path.rsplit(".", 1)[-1].lower()
        mime = f"image/{ext}" if ext in ("png", "jpg", "jpeg", "webp") else "image/png"

        context_block = _build_context_block(image_caption, answer_context)

        # Detect if this is a person/author query
        is_person_query = any(kw in question.lower() for kw in [
            "author", "who is", "reviewer", "wrote", "editor", "person", "photo", "headshot"
        ])

        person_rule = (
            "   - If this is a headshot/portrait/person photo and the question asks about a person: score 0.8+\n"
            "   - Even if you cannot identify the person by face, score HIGH if it looks like an author photo\n"
        ) if is_person_query else (
            "   - Person images: score HIGH only if person is directly relevant to question\n"
        )

        prompt = (
            f"{context_block}\n\n" if context_block else ""
        ) + (
            f"User question: '{question}'\n\n"
            "Look at this image carefully. Answer these TWO things:\n\n"
            "1. SCORE (0.0-1.0): How relevant is this image to the question?\n"
            "   Rules:\n"
            "   - Score based on what you ACTUALLY SEE\n"
            "   - If PDF caption is provided, use it to verify what/who is shown\n"
            f"{person_rule}"
            "   - Charts/graphs: score HIGH if the chart topic matches the question\n"
            "   - Code snippets, tables, logos, decorative borders: score 0.0-0.1\n"
            "   - If caption identifies this as relevant to the question: score HIGH\n\n"
            "2. DESCRIPTION (1-2 sentences): Describe what you see.\n"
            "   - If context or caption contains a person's name, you MUST use it\n"
            "   - Format: 'This is a headshot of [Full Name], [role from context].'\n"
            "   - Never say 'identified in nearby text' — just use the name directly\n"
            "   - For charts/diagrams: explain what is shown\n"
            "   - Do NOT invent names not present in the context\n\n"
            "Reply ONLY in this exact format:\n"
            "SCORE: 0.8\n"
            "DESCRIPTION: ..."
        )

        raw = _vision_chat(
            messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}, {"type": "text", "text": prompt}]}],
            max_tokens=200, temperature=0.1,
        )
        print(f"[VISION RAW] {os.path.basename(image_path)}: {raw[:120]}")

        score       = 0.0
        description = ""

        for line in raw.splitlines():
            line = line.strip()
            if line.upper().startswith("SCORE:"):
                try:
                    score = float(line.split(":", 1)[1].strip())
                    score = max(0.0, min(1.0, score))
                except ValueError:
                    pass
            elif line.upper().startswith("DESCRIPTION:"):
                description = line.split(":", 1)[1].strip()

        # Fallback score parse
        if score == 0.0 and raw:
            nums = re.findall(r'\b(0\.\d+|1\.0|1)\b', raw)
            if nums:
                try:
                    score = float(nums[0])
                except ValueError:
                    pass

        # Fallback description
        if not description and raw:
            lines = [l.strip() for l in raw.splitlines()
                     if l.strip() and not l.upper().startswith("SCORE:")]
            description = " ".join(lines)[:200]

        return {"score": score, "description": description}

    except Exception as e:
        print(f"[VISION] Error on {image_path}: {e}")
        return {"score": 0.0, "description": ""}


def rerank_images(
    image_paths: list,
    question: str,
    answer_context: str = "",
    threshold: float = 0.45,
) -> list:
    """
    Score and rank images. Now passes PDF-extracted caption to VLM.
    Images with a strong caption get much better identification.
    """
    scored = []

    for img in image_paths:
        img_path = img.get("path", "")
        if not img_path or not os.path.exists(img_path):
            print(f"[RERANK] Skipping missing: {img_path}")
            continue

        filename     = img.get("filename", os.path.basename(img_path))
        # Use PDF-extracted caption if available — this is the key upgrade
        image_caption = img.get("combined") or img.get("caption")

        print(f"[RERANK] Scoring: {filename} (page {img.get('page')}) "
              f"caption={'yes' if image_caption else 'no'}")

        result = _describe_and_score_image(
            img_path,
            question,
            answer_context,
            image_caption=image_caption,
        )
        score       = result["score"]
        description = result["description"]

        print(f"[RERANK] {filename} → score={score:.2f}")

        if score >= threshold:
            scored.append({**img, "score": score, "description": description})
            print(f"[RERANK] ✓ Kept {filename}")
        else:
            print(f"[RERANK] ✗ Filtered {filename} (score={score:.2f} < {threshold})")

    scored.sort(key=lambda x: x["score"], reverse=True)
    print(f"[RERANK] Final: {len(scored)} images passed")
    return scored


def should_show_table(table_markdown: str, question: str) -> dict:
    """Ask Groq to decide: show table, summarize, or both."""
    try:
        import json
        from llm_client import chat as _chat
        response = _chat(
            model_tier="fast",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Given a table and a user question, "
                        "decide the best presentation. "
                        "Reply ONLY with JSON: "
                        '{"action": "show"} or {"action": "summarize", "summary": "..."} '
                        'or {"action": "both", "summary": "..."}'
                    ),
                },
                {"role": "user", "content": f"Question: {question}\n\nTable:\n{table_markdown}"},
            ],
            max_tokens=300,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception:
        return {"action": "show"}