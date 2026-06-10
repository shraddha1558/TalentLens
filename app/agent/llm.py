"""
Thin async wrapper around an LLM provider.

Reads LLM_PROVIDER env var (default: "gemini").
Supported: "gemini" | "groq" | "openai" | "anthropic"

Each call goes through call_llm(system, messages, **kwargs) → str
"""

import json
import logging
import os
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "")            # override model name
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))


async def call_llm(
    system: str,
    messages: List[Dict[str, str]],
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
    json_mode: bool = False,
) -> str:
    """
    messages: list of {"role": "user"|"assistant", "content": "..."}
    Returns: the assistant text response (stripped).
    """
    if LLM_PROVIDER == "gemini":
        return await _gemini(system, messages, temperature, max_tokens, json_mode)
    elif LLM_PROVIDER == "groq":
        return await _groq(system, messages, temperature, max_tokens, json_mode)
    elif LLM_PROVIDER in ("openai", "openrouter"):
        return await _openai_compat(system, messages, temperature, max_tokens, json_mode)
    elif LLM_PROVIDER == "anthropic":
        return await _anthropic(system, messages, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


# ── Gemini ──────────────────────────────────────────────────────────────────

async def _gemini(system, messages, temperature, max_tokens, json_mode) -> str:
    import httpx

    api_key = _require_env("GEMINI_API_KEY")
    model = LLM_MODEL or "gemini-1.5-flash"

    # Gemini uses "parts" + optional systemInstruction
    contents = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})

    payload: Dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if json_mode:
        payload["generationConfig"]["responseMimeType"] = "application/json"

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    async with httpx.AsyncClient(timeout=25) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()

    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


# ── Groq ─────────────────────────────────────────────────────────────────────

async def _groq(system, messages, temperature, max_tokens, json_mode) -> str:
    import httpx

    api_key = _require_env("GROQ_API_KEY")
    model = LLM_MODEL or "llama-3.1-8b-instant"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    async with httpx.AsyncClient(timeout=25) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        r.raise_for_status()
        data = r.json()

    return data["choices"][0]["message"]["content"].strip()


# ── OpenAI / OpenRouter ───────────────────────────────────────────────────────

async def _openai_compat(system, messages, temperature, max_tokens, json_mode) -> str:
    import httpx

    if LLM_PROVIDER == "openrouter":
        api_key = _require_env("OPENROUTER_API_KEY")
        base_url = "https://openrouter.ai/api/v1/chat/completions"
        model = LLM_MODEL or "meta-llama/llama-3.1-8b-instruct:free"
    else:
        api_key = _require_env("OPENAI_API_KEY")
        base_url = "https://api.openai.com/v1/chat/completions"
        model = LLM_MODEL or "gpt-4o-mini"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    async with httpx.AsyncClient(timeout=25) as client:
        r = await client.post(
            base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        r.raise_for_status()
        data = r.json()

    return data["choices"][0]["message"]["content"].strip()


# ── Anthropic ─────────────────────────────────────────────────────────────────

async def _anthropic(system, messages, temperature, max_tokens) -> str:
    import httpx

    api_key = _require_env("ANTHROPIC_API_KEY")
    model = LLM_MODEL or "claude-3-haiku-20240307"

    payload = {
        "model": model,
        "system": system,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    async with httpx.AsyncClient(timeout=25) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
        )
        r.raise_for_status()
        data = r.json()

    return data["content"][0]["text"].strip()


# ── Util ──────────────────────────────────────────────────────────────────────

def _require_env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(
            f"Environment variable {key} is not set. "
            "Set it in your .env file or deployment environment."
        )
    return val


def safe_json(text: str) -> Optional[dict]:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON from LLM: %s", text[:200])
        return None