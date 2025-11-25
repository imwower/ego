"""Tests for the TeacherBridge prompt construction and mock replies."""

import pytest

from bridge.teacher_api import TeacherBridge


def test_prompt_contains_trigger_and_context():
    bridge = TeacherBridge(use_mock=True)
    ctx = {
        "prediction_error_norm": 0.42,
        "proto_state": {"energy": 0.8, "pain": 0.2, "curiosity": 0.5},
        "recent_spikes": {"vision": [1, 3], "assoc": [5]},
        "notes": "vision drift",
    }
    out = bridge.ask_gemini(ctx, trigger_type="CONFUSION")

    prompt = out["prompt"]
    assert "CONFUSION" in prompt
    assert "0.42" in prompt
    assert "pain=0.200" in prompt
    assert "vision:[1, 3]" in prompt
    assert "vision drift" in prompt


def test_invalid_trigger_raises():
    bridge = TeacherBridge(use_mock=True)
    with pytest.raises(ValueError):
        bridge.ask_gemini({}, trigger_type="OTHER")


def test_codex_cli_failure_falls_back_to_mock():
    # Use a command guaranteed to fail (non-zero exit) but present on most systems: "false".
    bridge = TeacherBridge(use_mock=False, provider="codex_cli", codex_cmd="false")
    out = bridge.ask_gemini({"prediction_error_norm": 0.5}, trigger_type="CONFUSION")
    assert "MOCK" in out["reply"]
    assert out["provider_ok"] is False
    assert out["provider"] == "codex_cli"


def test_codex_cli_success_returns_output():
    # Use printf to echo the prompt; ensures success path without real codex.
    bridge = TeacherBridge(
        use_mock=False,
        provider="codex_cli",
        codex_cmd="printf",
        codex_args=["%s"],
    )
    out = bridge.ask_gemini({"prediction_error_norm": 0.1}, trigger_type="CONFUSION")
    assert out["provider_ok"] is True
    assert "触发类型" in out["reply"]
    assert "CONFUSION" in out["reply"]
