from __future__ import annotations

import time
from typing import Any

import numpy as np
from keras import backend as K
from keras.utils import pad_sequences


def _safe_normalize(probs: np.ndarray) -> np.ndarray:
    arr = np.asarray(probs, dtype=float)
    arr = np.clip(arr, a_min=0.0, a_max=None)
    total = float(arr.sum())
    if total > 0:
        return arr / total
    if arr.size == 0:
        return arr
    return np.full_like(arr, 1.0 / arr.size)


def sample_batch(
    sample_size: int,
    tokenizer: Any,
    max_sequence_len: int,
    model: Any,
    batch_size: int,
    start_token: str,
    end_token: str,
) -> tuple[list[list[str]], list[list[float]]]:
    """
    Generate synthetic dual-stream sequences (events + scaled time deltas).

    Parameters:
        sample_size: Number of traces to generate.
        tokenizer: Fitted tokenizer for event names.
        max_sequence_len: Maximum sequence length used during training.
        model: Trained Keras model returning (event_probs, time_delta).
        batch_size: Number of traces generated per loop iteration.
        start_token: Symbol used to start every trace.
        end_token: Symbol that terminates a trace.

    Returns:
        Tuple of ``(event_sequences, time_sequences)``.
    """
    start_id = tokenizer.word_index.get(start_token)
    end_id = tokenizer.word_index.get(end_token)
    if start_id is None or end_id is None:
        raise ValueError("Tokenizer is missing START/END tokens.")

    index_word: dict[int, str] = {index: word for word, index in tokenizer.word_index.items()}
    sequences_events: list[list[str]] = []
    sequences_times: list[list[float]] = []

    start_time = time.perf_counter()

    while len(sequences_events) < sample_size:
        current_batch = min(int(batch_size), sample_size - len(sequences_events))
        if current_batch <= 0:
            break

        event_histories: list[list[int]] = [[start_id] for _ in range(current_batch)]
        time_histories: list[list[float]] = [[0.0] for _ in range(current_batch)]
        active = np.ones(current_batch, dtype=bool)

        while np.any(active):
            padded_events = pad_sequences(
                event_histories, maxlen=max_sequence_len, padding="pre"
            )
            padded_times = pad_sequences(
                time_histories, maxlen=max_sequence_len, padding="pre", dtype="float32"
            )
            event_pred, time_pred = model.predict_on_batch([padded_events, padded_times])

            for idx in range(current_batch):
                if not active[idx]:
                    continue

                probs = event_pred[idx].astype(float)
                if probs.size == 0:
                    active[idx] = False
                    continue
                probs[0] = 0.0  # never sample padding token
                probs = _safe_normalize(probs)
                next_id = int(np.random.choice(len(probs), p=probs)) # Hier das Petri Net Sampling einfügen
                next_time = float(np.clip(time_pred[idx][0], 0.0, 1.0))

                event_histories[idx].append(next_id)
                time_histories[idx].append(next_time)

                if next_id == end_id or len(event_histories[idx]) >= max_sequence_len:
                    active[idx] = False

        for events, times in zip(event_histories, time_histories):
            # Ensure alignment and convert ids back to strings
            if len(events) != len(times):
                continue
            event_tokens = [index_word.get(token_id, end_token) for token_id in events]
            sequences_events.append(event_tokens)
            sequences_times.append(times)

    K.clear_session()

    elapsed = time.perf_counter() - start_time
    print(f"Generated {len(sequences_events)} sequences in {elapsed:.2f}s")

    return sequences_events, sequences_times
