"""Track the tokens and execution progress of one generated sequence."""

from enum import Enum, auto
from itertools import count

from nano_vllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """Hold a prompt, its generated tokens, and scheduler/cache state."""

    # Each KV-cache block can store this many token positions.
    block_size = 256
    # This iterator is shared by all Sequence instances. Each call to next()
    # returns a fresh integer: 0, then 1, then 2, and so on.
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params=SamplingParams()):  # noqa: B008
        """Create a waiting sequence from prompt tokens and generation settings.

        token_ids is the input prompt; sampling_params controls token sampling
        and stopping conditions for this sequence.
        """
        self.seq_id = next(Sequence.counter)
        # Keep our own copy so generation cannot modify the caller's prompt list.
        self.token_ids = token_ids.copy()
        # This boundary separates prompt tokens from tokens generated later.
        self.last_token = token_ids[-1]
        self.num_prompt_tokens = len(token_ids)
        self.num_tokens = len(token_ids)
        self.sampling_params = sampling_params
        # A newly created sequence has not been selected for a model step.
        self.status = SequenceStatus.WAITING
        # Tokens whose model computation has finished and whose KV is cached.
        self.num_cached_tokens = 0
        # Tokens selected for the current step but not yet computed.
        self.num_scheduled_tokens = 0
        # The block manager will fill this logical-to-physical KV block mapping.
        # store physical kv block id, like seq.block_table = [12,7,19]
        self.block_table: list[int] = []

    def append_token(self, token_id: int) -> None:
        """Append a token to the sequence."""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED
