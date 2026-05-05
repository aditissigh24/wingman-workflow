"""
Unified LLM client via OpenRouter.

All text/LLM calls in the pipeline go through here.
Supports "gemini" and "claude" as logical model aliases.
OpenRouter provides an OpenAI-compatible endpoint, so we use the openai SDK.
"""

from openai import OpenAI
from config import OPENROUTER_API_KEY, OR_CLAUDE_MODEL, OR_GEMINI_MODEL

_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


def complete(
    system: str,
    user: str,
    model: str = "gemini",
    max_tokens: int = 4096,
) -> str:
    """
    Call an LLM via OpenRouter and return the response text.

    model: "gemini" → OR_GEMINI_MODEL, "claude" → OR_CLAUDE_MODEL
    """
    model_id = OR_GEMINI_MODEL if model == "gemini" else OR_CLAUDE_MODEL
    response = _client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()
