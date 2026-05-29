"""Unit tests for capx.llm.gemini_client. No network calls."""

from __future__ import annotations

import base64

import pytest

genai = pytest.importorskip("google.genai")  # noqa: F841

from capx.llm.gemini_client import (  # noqa: E402
    _decode_data_uri,
    _messages_to_gemini,
    _strip_provider_prefix,
    is_gemini_native_model,
)


def test_is_gemini_native_model():
    assert is_gemini_native_model("google/gemini-3.1-pro-preview")
    assert is_gemini_native_model("google/gemini-2.5-flash-lite")
    assert is_gemini_native_model("gemini-robotics-er-1.6-preview")
    assert is_gemini_native_model("gemini-3.1-pro-preview")
    assert not is_gemini_native_model("openrouter/google/gemini-2.5-pro-preview")
    assert not is_gemini_native_model("openai/gpt-5.4")
    assert not is_gemini_native_model("anthropic/claude-opus-4-5")


def test_strip_provider_prefix():
    assert _strip_provider_prefix("google/gemini-3.1-pro-preview") == "gemini-3.1-pro-preview"
    assert _strip_provider_prefix("gemini-flash") == "gemini-flash"


def test_decode_data_uri():
    raw = b"\x89PNG\r\n\x1a\n-fake-bytes"
    uri = f"data:image/png;base64,{base64.b64encode(raw).decode()}"
    data, mime = _decode_data_uri(uri)
    assert data == raw
    assert mime == "image/png"


def test_decode_data_uri_rejects_non_data_uri():
    with pytest.raises(ValueError):
        _decode_data_uri("https://example.com/foo.png")


def test_messages_to_gemini_text_only():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": [{"type": "text", "text": "How are you?"}]},
    ]
    contents, system_instruction = _messages_to_gemini(messages)

    assert system_instruction == "You are helpful."
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[0].parts[0].text == "Hi."
    assert contents[1].parts[0].text == "Hello."
    assert contents[2].parts[0].text == "How are you?"


def test_messages_to_gemini_multimodal():
    raw = b"jpegbytes"
    uri = f"data:image/jpeg;base64,{base64.b64encode(raw).decode()}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": uri}},
            ],
        },
    ]
    contents, system_instruction = _messages_to_gemini(messages)

    assert system_instruction is None
    assert len(contents) == 1
    parts = contents[0].parts
    assert len(parts) == 2
    assert parts[0].text == "What's in this image?"
    assert parts[1].inline_data.data == raw
    assert parts[1].inline_data.mime_type == "image/jpeg"


def test_messages_to_gemini_merges_multiple_system_messages():
    messages = [
        {"role": "system", "content": "Rule 1."},
        {"role": "system", "content": "Rule 2."},
        {"role": "user", "content": "Go."},
    ]
    _, system_instruction = _messages_to_gemini(messages)
    assert system_instruction == "Rule 1.\n\nRule 2."


def test_messages_to_gemini_rejects_unknown_part_type():
    messages = [{"role": "user", "content": [{"type": "audio_url", "audio_url": "x"}]}]
    with pytest.raises(ValueError, match="Unsupported content part type"):
        _messages_to_gemini(messages)
