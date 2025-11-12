"""Tests for the run_demo ticker prompt helpers."""
from __future__ import annotations

import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "vecm_project") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "vecm_project"))

import run_demo  # type: ignore  # noqa: E402


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("bbri, bbni", ["BBRI.JK", "BBNI.JK"]),
        ("BBRI.JK,usdidr=x", ["BBRI.JK", "USDIDR=X"]),
        ("^JKSE,bbca", ["^JKSE", "BBCA.JK"]),
    ],
)
def test_parse_ticker_prompt_normalises(prompt: str, expected: list[str]) -> None:
    assert run_demo.parse_ticker_prompt(prompt) == expected


def test_parse_ticker_prompt_removes_duplicates() -> None:
    result = run_demo.parse_ticker_prompt("bbri, bbri, bbni")
    assert result == ["BBRI.JK", "BBNI.JK"]


@pytest.mark.parametrize("prompt", ["", "bbri", " , , "])
def test_parse_ticker_prompt_requires_two(prompt: str) -> None:
    with pytest.raises(ValueError):
        run_demo.parse_ticker_prompt(prompt)
