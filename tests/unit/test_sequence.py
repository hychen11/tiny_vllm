"""Tests for the M1 sequence state and token history."""

from nano_vllm.sampling_params import SamplingParams
from nano_vllm.sequence import Sequence, SequenceStatus


def test_num_tokens_and_last_token_follow_appends() -> None:
    prompt = [10, 20]
    seq = Sequence(prompt, SamplingParams())

    assert seq.num_tokens == 2
    assert seq.last_token == 20

    seq.append_token(30)

    assert seq.num_tokens == 3
    assert seq.last_token == 30
    assert seq.num_prompt_tokens == 2
    assert prompt == [10, 20]

    seq.append_token(40)
    assert seq.token_ids == [10, 20, 30, 40]
    assert seq.num_tokens == len(seq.token_ids) == 4
    assert seq.last_token == seq.token_ids[-1] == 40
    assert seq.num_prompt_tokens == 2


def test_new_sequences_have_unique_ids_and_independent_state() -> None:
    params = SamplingParams(temperature=0.7, max_tokens=5, ignore_eos=True)
    first = Sequence([1, 2], params)
    second = Sequence([1, 2], params)

    assert second.seq_id == first.seq_id + 1
    assert first.sampling_params is params
    assert first.status is second.status is SequenceStatus.WAITING
    assert first.is_finished is second.is_finished is False
    assert first.num_cached_tokens == second.num_cached_tokens == 0
    assert first.num_scheduled_tokens == second.num_scheduled_tokens == 0
    assert first.block_table == second.block_table == []

    first.append_token(3)
    first.block_table.append(12)
    first.status = SequenceStatus.FINISHED
    assert first.is_finished is True
    assert second.is_finished is False
    assert second.token_ids == [1, 2]
    assert second.block_table == []


def test_prompt_list_is_copied_on_construction() -> None:
    prompt = [7, 8]
    seq = Sequence(prompt, SamplingParams())
    prompt.append(9)

    assert seq.token_ids == [7, 8]
    assert seq.num_prompt_tokens == 2
