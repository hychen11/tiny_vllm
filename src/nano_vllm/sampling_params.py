"""Generation settings for one request.

This module describes how tokens should be sampled and when generation should stop.
It stores user choices and validates them; the sampler and engine apply those choices.
"""

from dataclasses import dataclass


@dataclass
class SamplingParams:
    """Options supplied when creating a text-generation request."""

    temperature: float = 1.0
    max_tokens: int = 128
    ignore_eos: bool = False

    def __post_init__(self):
        """Validate field values after the dataclass initializer assigns them."""
        # Greedy sampling always picks the highest-probability next token, so
        # its output is deterministic. It is conventionally temperature=0.
        # This initial implementation requires a positive temperature instead.
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
