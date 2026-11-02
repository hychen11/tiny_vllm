"""Tests for the small set of M1 generation settings."""

import pytest

from nano_vllm.sampling_params import SamplingParams


def test_defaults_and_explicit_values() -> None:
    assert SamplingParams() == SamplingParams(temperature=1.0, max_tokens=128, ignore_eos=False)
    assert SamplingParams(temperature=0.7, max_tokens=1, ignore_eos=True).ignore_eos is True


@pytest.mark.parametrize("temperature", [0.0, -0.1, 1e-11])
def test_greedy_or_negative_temperature_is_rejected(temperature: float) -> None:
    with pytest.raises((AssertionError, ValueError), match="greedy sampling is not permitted"):
        SamplingParams(temperature=temperature)


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_nonpositive_max_tokens_is_rejected(max_tokens: int) -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        SamplingParams(max_tokens=max_tokens)
