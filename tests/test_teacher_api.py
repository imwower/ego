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
