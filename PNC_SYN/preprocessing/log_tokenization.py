from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

IntArray = npt.NDArray[np.int_]
FloatArray = npt.NDArray[np.float_]


def tokenize_log(
    event_sequences: Sequence[Sequence[str]],
    time_sequences: Sequence[Sequence[float]],
) -> tuple[IntArray, IntArray, FloatArray, FloatArray, int, int, Tokenizer]:
    """
    Tokenize event sequences and align them with numeric time streams.

    Parameters:
    event_sequences: Token lists per trace (including START/END markers).
    time_sequences: Scaled time deltas aligned with ``event_sequences``.

    Returns:
    tuple: ``(event_x, event_y, time_x, time_y, vocab_size, max_len, tokenizer)``.

    Raises:
    ValueError: If inputs are empty or lengths mismatched.
    """
    if not event_sequences or not time_sequences:
        raise ValueError("Input sequences must contain at least one trace")
    if len(event_sequences) != len(time_sequences):
        raise ValueError("Event and time sequences must have equal length")

    tokenizer = Tokenizer(lower=False)
    tokenizer.fit_on_texts(event_sequences)
    total_words = len(tokenizer.word_index) + 1
    print(f"Number of unique event tokens: {total_words - 1}")

    encoded_events = tokenizer.texts_to_sequences(event_sequences)

    event_inputs: list[list[int]] = []
    event_targets: list[int] = []
    time_inputs: list[list[float]] = []
    time_targets: list[float] = []

    for events, times in zip(encoded_events, time_sequences):
        if len(events) != len(times):
            raise ValueError("Event and time sequence lengths must match")
        for i in range(1, len(events)):
            event_inputs.append(events[:i])
            event_targets.append(events[i])
            time_inputs.append(list(times[:i]))
            time_targets.append(float(times[i]))

    if not event_inputs:
        raise ValueError("Provided sequences produced no training samples")

    max_sequence_len = max(len(seq) for seq in event_inputs)
    event_x = pad_sequences(event_inputs, maxlen=max_sequence_len, padding="pre")
    time_x = pad_sequences(
        time_inputs, maxlen=max_sequence_len, padding="pre", dtype="float32"
    )

    event_targets_array: IntArray = np.asarray(event_targets, dtype=np.int_)
    time_targets_array: FloatArray = np.asarray(time_targets, dtype=np.float32)

    print(f"Number of training samples: {len(event_x)}")
    print(f"Sequence length: {max_sequence_len}")

    return (
        np.asarray(event_x, dtype=np.int_),
        event_targets_array,
        time_x,
        time_targets_array,
        total_words,
        max_sequence_len,
        tokenizer,
    )
