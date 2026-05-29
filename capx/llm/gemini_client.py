"""Native Gemini API wrapper using the `google-genai` SDK.

Required because we only have an `AQ.`-prefixed Google API key, which works
against the native Gemini API but NOT against the OpenAI-compat endpoint
(that one needs an `AIza` key). This module therefore owns every
`google/gemini-*` call; OpenRouter-routed Gemini (`openrouter/google/...`) and
all other providers stay on the existing OpenAI-style path in `client.py`.
"""

from __future__ import annotations

import base64
import random
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from capx.envs.launch import LaunchArgs
    from capx.llm.client import ModelQueryArgs

KEYFILE = ".geminikey"
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
DEFAULT_THINKING_BUDGET = -1  # -1 = dynamic; matches "thinking enabled" on Claude branch
INCLUDE_THOUGHTS = True

_client_cache: dict[str, Any] = {}


def is_gemini_native_model(model: str) -> bool:
    """Route to the native Gemini API (no AIza key available for OpenAI-compat).

    Matches both the prefixed form (`google/gemini-*`) used elsewhere in the
    project and the bare SDK-style name (`gemini-*`, including
    `gemini-robotics-er-*`). OpenRouter-routed Gemini stays on the standard
    path — those models start with `openrouter/`.
    """
    return model.startswith(("google/", "gemini-"))


def _strip_provider_prefix(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def _get_client() -> Any:
    keyfile = Path(KEYFILE)
    if not keyfile.exists():
        raise FileNotFoundError(
            f"Gemini API key file not found: {keyfile.resolve()}. "
            "Place an AQ-prefixed key in `.geminikey` at the working directory root."
        )
    api_key = keyfile.read_text().strip()
    if api_key not in _client_cache:
        from google import genai
        _client_cache[api_key] = genai.Client(api_key=api_key)
    return _client_cache[api_key]


def _decode_data_uri(url: str) -> tuple[bytes, str]:
    if not url.startswith("data:"):
        raise ValueError(f"Expected data URI for image_url, got: {url[:60]}...")
    header, _, b64 = url.partition(",")
    mime_type = header[5:].split(";")[0]
    return base64.b64decode(b64), mime_type


def _messages_to_gemini(messages: list[dict]) -> tuple[list[Any], str | None]:
    """OpenAI chat-completions messages → (Gemini contents, system_instruction).

    - role=system → joined into system_instruction
    - role=user/assistant → Content(role="user"|"model", parts=[...])
    - text part → Part.from_text
    - image_url part with data URI → Part.from_bytes
    """
    from google.genai import types

    system_chunks: list[str] = []
    contents: list[Any] = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            text = (
                content
                if isinstance(content, str)
                else "".join(p.get("text", "") for p in content if isinstance(p, dict))
            )
            if text:
                system_chunks.append(text)
            continue

        gemini_role = "model" if role == "assistant" else "user"
        parts: list[Any] = []

        if isinstance(content, str):
            parts.append(types.Part.from_text(text=content))
        else:
            for part in content:
                ptype = part.get("type")
                if ptype == "text":
                    parts.append(types.Part.from_text(text=part["text"]))
                elif ptype == "image_url":
                    img = part["image_url"]
                    url = img["url"] if isinstance(img, dict) else img
                    data, mime = _decode_data_uri(url)
                    parts.append(types.Part.from_bytes(data=data, mime_type=mime))
                else:
                    raise ValueError(f"Unsupported content part type: {ptype}")

        if parts:
            contents.append(types.Content(role=gemini_role, parts=parts))

    return contents, ("\n\n".join(system_chunks) if system_chunks else None)


def _build_config(args: "LaunchArgs | ModelQueryArgs", system_instruction: str | None) -> Any:
    from google.genai import types
    return types.GenerateContentConfig(
        temperature=getattr(args, "temperature", 0.2),
        max_output_tokens=getattr(args, "max_tokens", 4096),
        thinking_config=types.ThinkingConfig(
            thinking_budget=DEFAULT_THINKING_BUDGET,
            include_thoughts=INCLUDE_THOUGHTS,
        ),
        system_instruction=system_instruction,
    )


def _split_parts(response: Any) -> tuple[str, str | None]:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    for cand in getattr(response, "candidates", None) or []:
        if cand.content is None:
            continue
        for part in cand.content.parts or []:
            text = getattr(part, "text", None)
            if not text:
                continue
            if getattr(part, "thought", False):
                reasoning_parts.append(text)
            else:
                content_parts.append(text)
    return "".join(content_parts), ("\n".join(reasoning_parts) or None)


def _is_retryable(exc: BaseException) -> int | None:
    """Return the HTTP-ish status code if `exc` should trigger a retry, else None."""
    code = getattr(exc, "code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    return code if code in RETRY_STATUS_CODES else None


def query_gemini(args: "LaunchArgs | ModelQueryArgs", prompt: list[dict]) -> dict:
    """Non-streaming call. Returns `{"content": str, "reasoning": str | None}`."""
    client = _get_client()
    model = _strip_provider_prefix(args.model)
    contents, system_instruction = _messages_to_gemini(prompt)
    config = _build_config(args, system_instruction)

    start = time.time()
    retry = 1
    while True:
        try:
            response = client.models.generate_content(
                model=model, contents=contents, config=config,
            )
            break
        except Exception as e:
            code = _is_retryable(e)
            if code is None:
                raise
            sleep_time = 240 + random.uniform(-90, 90)
            print(
                f"Retry {retry}. Gemini call failed with {code}: {e}. "
                f"Retrying in {sleep_time:.0f}s..."
            )
            time.sleep(sleep_time)
            retry += 1

    elapsed = time.time() - start
    print(f"Time taken to query Gemini ({model}): {elapsed:.2f} seconds")

    if getattr(args, "debug", False):
        print(response)

    content, reasoning = _split_parts(response)
    return {"content": content, "reasoning": reasoning}


def query_gemini_streaming(
    args: "LaunchArgs | ModelQueryArgs", prompt: list[dict]
) -> Iterable[dict]:
    """Streaming call. Yields the same shape as `client.query_model_streaming`."""
    client = _get_client()
    model = _strip_provider_prefix(args.model)
    contents, system_instruction = _messages_to_gemini(prompt)
    config = _build_config(args, system_instruction)

    full_content = ""
    full_reasoning = ""
    start = time.time()

    stream = client.models.generate_content_stream(
        model=model, contents=contents, config=config,
    )
    for chunk in stream:
        for cand in getattr(chunk, "candidates", None) or []:
            if cand.content is None:
                continue
            for part in cand.content.parts or []:
                text = getattr(part, "text", None)
                if not text:
                    continue
                if getattr(part, "thought", False):
                    full_reasoning += text
                    yield {"type": "reasoning_delta", "content": text}
                else:
                    full_content += text
                    yield {"type": "content_delta", "content": text}

    elapsed = time.time() - start
    print(f"Time taken to query Gemini stream ({model}): {elapsed:.2f} seconds")
    if full_reasoning:
        print(f"Reasoning extracted ({len(full_reasoning)} chars)")
    else:
        print("No reasoning returned by model")

    yield {
        "type": "done",
        "content": full_content,
        "reasoning": full_reasoning or None,
    }
