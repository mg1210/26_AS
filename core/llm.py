"""
core/llm.py
───────────
Thin wrapper around the Anthropic API.
Agents call ask() to get LLM-generated reasoning, narratives, and recommendations.
"""

import os
import json
import anthropic


_client = None

def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

        if not api_key:
            raise RuntimeError(
                "No Anthropic API key found. "
                "Set the ANTHROPIC_API_KEY environment variable, "
                "or enter it on the Home page of the UI."
            )

        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _empty_usage() -> dict:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _usage_from(response) -> dict:
    """Extract token usage from an Anthropic response (tokens only, no cost)."""
    u = getattr(response, "usage", None)
    it = int(getattr(u, "input_tokens", 0) or 0)
    ot = int(getattr(u, "output_tokens", 0) or 0)
    return {"input_tokens": it, "output_tokens": ot, "total_tokens": it + ot}


def ask_with_usage(prompt: str, system: str = "", max_tokens: int = 1024):
    """Send a prompt to Claude; return (text, usage_dict). usage_dict carries
    input_tokens/output_tokens/total_tokens. Returns ('', zero-usage) if no API key."""
    try:
        client = _get_client()
    except RuntimeError as e:
        print(f"  LLM skipped — {e.args[0].splitlines()[0]}")
        return "", _empty_usage()
    messages = [{"role": "user", "content": prompt}]
    kwargs = dict(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        messages=messages,
    )
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    return response.content[0].text, _usage_from(response)


def ask(prompt: str, system: str = "", max_tokens: int = 1024) -> str:
    """Send a prompt to Claude and return the text response. Returns '' if no API key."""
    text, _ = ask_with_usage(prompt, system=system, max_tokens=max_tokens)
    return text


def ask_json_with_usage(prompt: str, system: str = "", max_tokens: int = 1024):
    """Ask Claude for a JSON response; return (parsed_dict_or_None, usage_dict).

    Token usage is captured from the API response BEFORE the JSON parse is attempted, so it is
    never lost even if the response is empty or not valid JSON. On a parse failure this returns
    (None, usage) instead of raising — callers get the usage regardless and handle None."""
    raw, usage = ask_with_usage(prompt, system=system, max_tokens=max_tokens)
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1]
        clean = clean.rsplit("```", 1)[0]
    try:
        return json.loads(clean.strip()), usage
    except (json.JSONDecodeError, ValueError):
        return None, usage


def ask_json(prompt: str, system: str = "", max_tokens: int = 1024) -> dict:
    """Ask Claude for a JSON response. Strips markdown fences before parsing."""
    data, _ = ask_json_with_usage(prompt, system=system, max_tokens=max_tokens)
    return data


CREDIT_RISK_SYSTEM = """
You are an expert credit risk data scientist and model validator with 15+ years of
experience in consumer lending at major banks and NBFCs. You understand regulatory
requirements, model governance, IFRS 9, Basel frameworks, and best practices in
scorecard development. You communicate clearly to both technical and business audiences.
Always be precise, concise, and practical.
"""
