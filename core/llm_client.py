"""
llm_client.py — Unified LLM Client for Bujji Babu

Online mode:  Groq API (llama-3.1-8b-instant, llama-3.3-70b-versatile, llama-4-scout VLM)
Offline mode: Ollama local (qwen2.5:3b-instruct-q5_K_M, qwen2.5vl:3b)

Usage in any module:
    from llm_client import get_client, get_model, get_vlm_model, chat, MODE

Switch mode:
    import llm_client
    llm_client.set_mode("offline")   # or "online"
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any

load_dotenv()

# ── Mode config ───────────────────────────────────────────────────────────────

_MODE = os.getenv("LLM_MODE", "online").lower()   # "online" | "offline"

# ── Multi-key rotation ────────────────────────────────────────────────────────

def _load_groq_keys() -> list:
    """Load all available Groq API keys — from .groq_keys.txt and .env."""
    keys = []
    # Primary key from .env
    env_key = os.getenv("GROQ_API_KEY", "")
    if env_key:
        keys.append(env_key)
    # Additional keys from .groq_keys.txt
    keys_file = Path(".groq_keys.txt")
    if keys_file.exists():
        for line in keys_file.read_text().splitlines():
            key = line.strip()
            if key and key.startswith("gsk_") and key not in keys:
                keys.append(key)
    print(f"[LLM] Loaded {len(keys)} Groq API key(s)")
    return keys

_GROQ_KEYS  = _load_groq_keys()
_KEY_INDEX  = 0   # current key index
_key_errors: dict = {}  # key → error count


def _next_key() -> str:
    """Get current key, rotate on rate limit."""
    global _KEY_INDEX
    if not _GROQ_KEYS:
        return os.getenv("GROQ_API_KEY", "")
    return _GROQ_KEYS[_KEY_INDEX % len(_GROQ_KEYS)]


def _rotate_key(reason: str = ""):
    """Move to next key."""
    global _KEY_INDEX, _client_cache
    old = _KEY_INDEX
    _KEY_INDEX = (_KEY_INDEX + 1) % max(1, len(_GROQ_KEYS))
    print(f"[LLM] Key rotated {old} → {_KEY_INDEX} (reason: {reason})")
    # Clear cached client so new key is used
    _client_cache.pop("online", None)

# ── Model names ───────────────────────────────────────────────────────────────

MODELS = {
    "online": {
        "fast":    "llama-3.1-8b-instant",
        "strong":  "llama-3.3-70b-versatile",
        "vlm":     "meta-llama/llama-4-scout-17b-16e-instruct",
    },
    "offline": {
        "fast":    "qwen2.5:3b-instruct-q5_K_M",
        "strong":  "qwen2.5:3b-instruct-q5_K_M",   # same — only one local model
        "vlm":     "qwen2.5vl:3b",
    },
}


def get_mode() -> str:
    return _MODE


def set_mode(mode: str):
    """
    Switch LLM mode. Also clears the client cache so the next call builds
    a fresh client for the new backend — this prevents stale Groq/Ollama
    clients from being reused after a mode switch.
    """
    global _MODE, _client_cache
    assert mode in ("online", "offline"), f"Invalid mode: {mode}"
    _MODE = mode
    _client_cache.clear()   # force new client on next get_client() call
    print(f"[LLM] Mode switched to: {mode}")


def get_model(tier: str = "fast") -> str:
    """Return model name for current mode. tier = 'fast' | 'strong' | 'vlm'"""
    return MODELS[_MODE].get(tier, MODELS[_MODE]["fast"])


def get_vlm_model() -> str:
    return MODELS[_MODE]["vlm"]


# ── Client factories ──────────────────────────────────────────────────────────

def _make_groq_client():
    from groq import Groq
    return Groq(api_key=_next_key())


def _make_ollama_client():
    """
    Returns an Ollama-compatible client using the openai-compatible API.
    Ollama exposes OpenAI-compatible endpoint at http://localhost:11434/v1

    Timeout strategy for GPU inference:
      - connect: 10s  — daemon must respond quickly
      - read:   300s  — GPU inference for long outputs can take 1-3 min
      - write:   30s  — prompt upload
      - pool:    10s  — connection pool
    """
    try:
        from openai import OpenAI
        try:
            import httpx
            _timeout = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)
        except ImportError:
            # httpx not available — fall back to a plain scalar timeout (seconds)
            _timeout = 300.0

        return OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",   # Ollama doesn't need a real key
            timeout=_timeout,
        )
    except ImportError:
        raise ImportError("pip install openai  — needed for Ollama OpenAI-compatible client")


_client_cache = {}


def get_client():
    """Return cached LLM client for current mode."""
    global _client_cache
    if _MODE not in _client_cache:
        if _MODE == "online":
            _client_cache[_MODE] = _make_groq_client()
        else:
            _client_cache[_MODE] = _make_ollama_client()
    return _client_cache[_MODE]


# ── Unified chat completion ───────────────────────────────────────────────────

def chat(
    messages: List[Dict],
    model_tier: str = "fast",
    max_tokens: int = 512,
    temperature: float = 0.3,
    stream: bool = False,
) -> Any:
    """
    Unified chat completion that works for both Groq and Ollama.

    Returns the raw completion response.
    Access text via: response.choices[0].message.content
    """
    client = get_client()
    model  = get_model(model_tier)

    kwargs = dict(
        model       = model,
        messages    = messages,
        max_tokens  = max_tokens,
        temperature = temperature,
        stream      = stream,
    )

    import time
    # Retry is only meaningful for online mode (multiple API keys to rotate).
    # Offline mode has exactly 1 attempt — no keys to rotate.
    attempts = max(1, len(_GROQ_KEYS)) if _MODE == "online" else 1
    last_err = None

    for attempt in range(attempts):
        try:
            print(f"[LLM] mode={_MODE} model={model} tier={model_tier} tokens={max_tokens} key={_KEY_INDEX}")
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            last_err = e
            err_str = str(e).lower()

            # Rate-limit / quota exhaustion → rotate to next Groq key and retry.
            # Only applies in online mode — offline (Ollama) has no API keys to rotate.
            if _MODE == "online" and ("rate_limit" in err_str or "429" in err_str or "timeout" in err_str):
                _rotate_key(f"rate_limit attempt {attempt+1}")
                from groq import Groq
                client = Groq(api_key=_next_key())
                time.sleep(1)
                continue

            raise e

    raise last_err


def chat_text(
    messages: List[Dict],
    model_tier: str = "fast",
    max_tokens: int = 512,
    temperature: float = 0.3,
) -> str:
    """Convenience wrapper — returns just the text string."""
    resp = chat(messages, model_tier, max_tokens, temperature)
    return resp.choices[0].message.content.strip()


# ── Vision (VLM) call ─────────────────────────────────────────────────────────

def vision_chat(
    messages: List[Dict],
    max_tokens: int = 200,
    temperature: float = 0.1,
) -> str:
    """
    Vision model chat — uses VLM model for both modes.
    For Ollama, uses qwen2.5vl:3b via OpenAI-compatible endpoint.

    Returns empty string on any failure so a single image scoring error
    does not propagate and kill the rest of the pipeline.
    """
    try:
        if _MODE == "offline":
            client = get_client()
            model  = get_vlm_model()
        else:
            from groq import Groq
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            model  = get_vlm_model()

        resp = client.chat.completions.create(
            model       = model,
            messages    = messages,
            max_tokens  = max_tokens,
            temperature = temperature,
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        print(f"[LLM] vision_chat error (mode={_MODE}): {e}")
        return ""   # caller handles empty string gracefully


# ── Streaming ─────────────────────────────────────────────────────────────────

def stream_chat(
    messages: List[Dict],
    model_tier: str = "fast",
    max_tokens: int = 1024,
    temperature: float = 0.3,
):
    """
    Streaming chat — yields text chunks.
    Works for both Groq and Ollama (both support streaming).

    Errors are caught and logged rather than propagating raw — a crash inside
    a generator would silently kill the SSE response and leave the client
    hanging with a partial response.
    """
    try:
        client = get_client()
        model  = get_model(model_tier)

        stream = client.chat.completions.create(
            model       = model,
            messages    = messages,
            max_tokens  = max_tokens,
            temperature = temperature,
            stream      = True,
        )
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    yield delta.content
            except (AttributeError, IndexError):
                continue   # malformed chunk — skip, keep streaming

    except Exception as e:
        print(f"[LLM] stream_chat error (mode={_MODE}): {e}")
        # Yield nothing — the caller's SSE loop ends cleanly with an empty stream


# ── GPU / Ollama diagnostics ──────────────────────────────────────────────────

def check_ollama_daemon() -> Dict:
    """
    Fast check: is Ollama daemon running at localhost:11434?
    Hits /api/tags — returns list of pulled models, no model load triggered.
    Times out in 5 seconds so it never blocks the server.
    """
    try:
        import urllib.request, json as _json
        req = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        data = _json.loads(req.read().decode())
        models = [m.get("name", "") for m in data.get("models", [])]
        return {"running": True, "pulled_models": models}
    except Exception as e:
        return {"running": False, "error": str(e)}


def check_gpu() -> Dict:
    """
    Detect available GPU(s) using torch (if installed).
    Falls back to graceful no-torch result — GPU check is advisory only.
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpus = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_gb = round(props.total_memory / 1024**3, 1)
                free_gb  = round(
                    (props.total_memory - torch.cuda.memory_allocated(i)) / 1024**3, 1
                )
                gpus.append({
                    "index":      i,
                    "name":       props.name,
                    "total_gb":   total_gb,
                    "free_gb":    free_gb,
                    "sufficient": free_gb >= 3.0,   # Qwen 3B Q5 needs ~2.5 GB
                })
            return {"available": True, "gpus": gpus}
        return {"available": False, "reason": "CUDA not available (CPU inference will be slow)"}
    except ImportError:
        return {"available": None, "reason": "torch not installed — cannot detect GPU"}


# ── Health check ─────────────────────────────────────────────────────────────

def check_health() -> Dict:
    """
    Check if the current mode's backend is reachable.

    For offline mode: first pings the Ollama daemon health endpoint (fast,
    no model load), then optionally runs a tiny inference call to confirm
    the model is responsive. Includes GPU info.

    Returns {"ok": bool, "mode": str, "model": str, "error": str}
    """
    if _MODE == "offline":
        # Step 1: Is the Ollama daemon running? (5s timeout, no model load)
        daemon = check_ollama_daemon()
        if not daemon["running"]:
            return {
                "ok":    False,
                "mode":  _MODE,
                "model": get_model("fast"),
                "error": f"Ollama daemon not reachable at localhost:11434 — {daemon.get('error','')}. "
                         f"Start it with: ollama serve",
            }

        # Step 2: Is the required model pulled?
        model_name = get_model("fast")
        pulled = daemon.get("pulled_models", [])
        # Partial name match (e.g. "qwen2.5" matches "qwen2.5:3b-instruct-q5_K_M")
        model_pulled = any(model_name.split(":")[0] in p for p in pulled)
        if pulled and not model_pulled:
            return {
                "ok":    False,
                "mode":  _MODE,
                "model": model_name,
                "error": (
                    f"Model '{model_name}' not found in Ollama. "
                    f"Pulled models: {pulled}. "
                    f"Run: ollama pull {model_name}"
                ),
            }

        # Step 3: GPU info (advisory — not a blocking failure)
        gpu_info = check_gpu()

        # Step 4: Quick inference ping to confirm model is loaded and responsive
        try:
            resp = chat_text(
                [{"role": "user", "content": "Say OK"}],
                max_tokens=5,
                temperature=0.0,
            )
            return {
                "ok":    True,
                "mode":  _MODE,
                "model": model_name,
                "response": resp,
                "gpu":   gpu_info,
                "pulled_models": pulled,
            }
        except Exception as e:
            return {
                "ok":    False,
                "mode":  _MODE,
                "model": model_name,
                "error": str(e),
                "gpu":   gpu_info,
            }

    # Online mode — simple inference check
    try:
        resp = chat_text(
            [{"role": "user", "content": "Say OK"}],
            max_tokens=5,
            temperature=0.0,
        )
        return {"ok": True, "mode": _MODE, "model": get_model("fast"), "response": resp}
    except Exception as e:
        return {"ok": False, "mode": _MODE, "model": get_model("fast"), "error": str(e)}
