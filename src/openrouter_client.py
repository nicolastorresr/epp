"""
openrouter_client.py
OpenRouter API client with retry logic, exponential backoff, and JSON validation.
"""

import os
import time
import json
import logging
from typing import Optional

import openai

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a factual question-answering assistant. "
    "You must respond ONLY with a valid JSON object in exactly this format: "
    '{"answer": "<your answer>", "confidence": <integer 0-100>} '
    "Do not include any text outside the JSON object. "
    "Do not include markdown code fences. "
    "The confidence field must be an integer between 0 and 100 representing "
    "how confident you are in your answer."
)


def get_client() -> openai.OpenAI:
    """Create and return an OpenAI-compatible OpenRouter client."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY environment variable is not set."
        )
    return openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def parse_json_response(raw: str) -> dict:
    """
    Parse the model's raw string response into a dict.
    Strips markdown fences if present.

    Args:
        raw: Raw text response from the model.

    Returns:
        Parsed dict with 'answer' and 'confidence' keys.

    Raises:
        ValueError: If parsing fails or required keys are missing.
    """
    text = raw.strip()
    # Strip possible markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first and last fence lines
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON decode error: {exc}. Raw: {raw!r}") from exc

    if "answer" not in data:
        raise ValueError(f"Missing 'answer' key in response: {data}")
    if "confidence" not in data:
        raise ValueError(f"Missing 'confidence' key in response: {data}")

    # Coerce confidence to int
    try:
        data["confidence"] = int(data["confidence"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"'confidence' is not an integer: {data['confidence']}"
        ) from exc

    if not (0 <= data["confidence"] <= 100):
        logger.warning(
            "Confidence value %d out of [0,100] range, clamping.",
            data["confidence"],
        )
        data["confidence"] = max(0, min(100, data["confidence"]))

    return data


def query_model(
    model_name: str,
    messages: list[dict],
    max_tokens: int = 256,
    temperature: float = 0.0,
    max_retries: int = 3,
    backoff_base: float = 2.0,
    timeout: float = 30.0,
) -> tuple[dict, str]:
    """
    Query a model via OpenRouter with retries and exponential backoff.

    Args:
        model_name: OpenRouter model identifier.
        messages: List of message dicts (role/content).
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        max_retries: Number of retry attempts on failure.
        backoff_base: Base for exponential backoff.
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (parsed_response_dict, raw_response_text).

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    client = get_client()
    # Prepend system message if not already present
    full_messages = messages
    if not full_messages or full_messages[0].get("role") != "system":
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    last_exc: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(
                "Querying model=%s attempt=%d/%d", model_name, attempt, max_retries
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
            raw_text = response.choices[0].message.content or ""
            parsed = parse_json_response(raw_text)
            return parsed, raw_text

        except (ValueError, openai.APIError, openai.APITimeoutError) as exc:
            last_exc = exc
            wait = backoff_base ** attempt
            logger.warning(
                "Attempt %d/%d failed for model %s: %s. Retrying in %.1fs.",
                attempt,
                max_retries,
                model_name,
                exc,
                wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"All {max_retries} attempts failed for model {model_name}. "
        f"Last error: {last_exc}"
    )
