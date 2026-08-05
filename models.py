"""
models.py - the free-model registry and a provider-agnostic generate() call.

KEY IDEA: Groq and OpenRouter both expose an *OpenAI-compatible* API. So one client
library (the `openai` SDK) talks to both - you only change three things:
    base_url  +  api_key  +  model slug
This is the most portable pattern in LLM engineering: swap providers without
rewriting your code.

Models under test are FREE (Groq free tier + OpenRouter ":free" variants).
Model slugs change over time - if one 404s, verify the current name at:
    Groq:        https://console.groq.com/docs/models
    OpenRouter:  https://openrouter.ai/models?q=free
The eval records a bad slug as an error and moves on, so one dead model never
crashes the run.
"""

import os
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str        # friendly name for the leaderboard
    provider: str    # "groq" | "openrouter"
    slug: str        # the provider's model id


# ---- The models under test (all free) ----
MODELS = [
    # Groq free tier - fast, reliable
    ModelSpec("Llama 3.3 70B (Groq)",  "groq",       "llama-3.3-70b-versatile"),
    ModelSpec("Llama 3.1 8B (Groq)",   "groq",       "llama-3.1-8b-instant"),
    ModelSpec("GPT-OSS 20B (Groq)",    "groq",       "openai/gpt-oss-20b"),
    # OpenRouter ":free" - extra variety (rate-limited)
    ModelSpec("Qwen 2.5 72B (OR)",     "openrouter", "qwen/qwen-2.5-72b-instruct:free"),
    ModelSpec("Mistral 7B (OR)",       "openrouter", "mistralai/mistral-7b-instruct:free"),
    ModelSpec("DeepSeek V3 (OR)",      "openrouter", "deepseek/deepseek-chat:free"),
]

# ---- Per-provider connection details ----
PROVIDERS = {
    "groq":       {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",   "key_env": "OPENROUTER_API_KEY"},
}


def provider_available(provider):
    """True if we have an API key for this provider in the environment/.env."""
    return bool(os.getenv(PROVIDERS[provider]["key_env"]))


def available_models():
    """Only the models whose provider key is actually set - so a partial setup
    (e.g. Groq key only) still runs, testing just those models."""
    return [m for m in MODELS if provider_available(m.provider)]


# Cache one client per provider (creating them is cheap but pointless to repeat).
_clients = {}

def _client(provider):
    if provider not in _clients:
        from openai import OpenAI          # lazy import: mock runs need no openai/key
        cfg = PROVIDERS[provider]
        _clients[provider] = OpenAI(
            base_url=cfg["base_url"],
            api_key=os.getenv(cfg["key_env"]),
        )
    return _clients[provider]


def generate(spec, system, user, temperature=0, max_tokens=700, retries=3):
    """Send one chat request and return the raw text. Retries transient errors
    (rate limits, timeouts) with exponential backoff."""
    client = _client(spec.provider)
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=spec.slug,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            status = getattr(e, "status_code", None)
            if status in (400, 401, 403, 404):   # permanent (dead/invalid model, bad key) - don't retry
                raise
            time.sleep(2 ** attempt)             # transient (429/5xx/timeout): 1s, 2s, 4s
    raise last_err


if __name__ == "__main__":
    print("Registry:")
    for m in MODELS:
        mark = "yes" if provider_available(m.provider) else "no key"
        print(f"  [{mark:6}] {m.name:24s} {m.provider}/{m.slug}")
