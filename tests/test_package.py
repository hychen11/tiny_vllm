"""Tests for the installed package boundary."""

from nano_vllm import __version__


def test_package_exposes_version() -> None:
    assert __version__ == "0.1.0"
