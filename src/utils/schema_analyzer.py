import hashlib
import json
import logging
import os

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT = 30

logger = logging.getLogger(__name__)

_cache: dict[str, dict] = {}

PROMPT = """You are analyzing a JSON structure to decide how to index it for search.

JSON sample:
{snippet}

Return a JSON object with:
- "skip_keys": list of key names that are boilerplate, config, UUIDs, IDs, or otherwise not worth searching — these will be removed before indexing

Example: {{"skip_keys": ["parameters", "policyDefinitions", "policyDefinitionGroups", "versions"]}}

Respond with valid JSON only."""


def _make_snippet(data, max_chars: int = 2000) -> str:
    """
    Creates a representative snippet for the LLM.
    For dicts: shows all keys with truncated values; arrays shown as length + first item.
    For lists: shows first item only.
    """
    def truncate(obj, depth=0):
        if depth > 3:
            return "..."
        if isinstance(obj, dict):
            result = {}
            for k, v in list(obj.items())[:10]:
                if isinstance(v, list):
                    result[k] = f"[{len(v)} items, first: {json.dumps(truncate(v[0], depth + 1)) if v else 'empty'}]"
                else:
                    result[k] = truncate(v, depth + 1)
            return result
        if isinstance(obj, str) and len(obj) > 120:
            return obj[:120] + "..."
        return obj

    return json.dumps(truncate(data), indent=2)[:max_chars]


def _fingerprint(keys: list[str]) -> str:
    """Cache key based on the sorted set of top-level keys."""
    return hashlib.md5(json.dumps(sorted(keys)).encode()).hexdigest()


def consistent_structure(r1: dict, r2: dict) -> bool:
    """Returns True if two records share the same top-level keys."""
    return set(r1.keys()) == set(r2.keys())


def analyze(data: dict) -> dict | None:
    """
    Sends a compact snippet of a JSON object to a local Ollama model.
    Returns {"skip_keys": [...]} or None if Ollama is unavailable or fails.
    Results are cached by the object's top-level keys.
    """
    fp = _fingerprint(list(data.keys()))
    if fp in _cache:
        logger.info("Schema analysis: cache hit")
        return _cache[fp]

    snippet = _make_snippet(data)
    prompt = PROMPT.format(snippet=snippet)

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "format": "json", "stream": False},
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        schema = json.loads(resp.json().get("response", "{}"))
        schema.setdefault("skip_keys", [])

        _cache[fp] = schema
        logger.info(f"Schema analysis result: {schema}")
        return schema

    except requests.exceptions.ConnectionError:
        logger.info("Ollama not available, falling back to generic JSON processing")
        return None
    except Exception as e:
        logger.warning(f"Schema analysis failed ({e}), falling back to generic JSON processing")
        return None
