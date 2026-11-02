"""Tests for the minimal command-line entry point."""

from __future__ import annotations

import pytest

from nano_vllm.cli import build_parser, main


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as result:
        parser.parse_args(["--version"])

    assert result.value.code == 0
    assert capsys.readouterr().out.strip() == "nano-vllm 0.1.0"


def test_module_cli_version(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["nano-vllm", "--version"])

    with pytest.raises(SystemExit) as result:
        main()

    assert result.value.code == 0
    assert capsys.readouterr().out.strip() == "nano-vllm 0.1.0"
