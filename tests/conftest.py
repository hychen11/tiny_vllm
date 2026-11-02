"""Shared pytest fixtures and deterministic test defaults."""

from __future__ import annotations

import random

import pytest


@pytest.fixture(autouse=True)
def _seed_python_random() -> None:
    """Keep tests deterministic before PyTorch-based tests are introduced."""
    random.seed(0)
