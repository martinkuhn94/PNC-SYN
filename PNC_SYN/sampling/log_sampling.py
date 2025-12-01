from __future__ import annotations

import time
from typing import Any, Optional, Tuple

import numpy as np
from keras import backend as K
from keras.utils import pad_sequences

from pm4py.objects.petri_net.obj import PetriNet, Marking
from PNC_SYN.petri_net_util import (
    analyze_transitions_with_end_labels,
    execute_transition,
    get_enabled_tokens_and_transitions,
    get_filtered_probabilities,
)


def _safe_normalize(probs: np.ndarray) -> np.ndarray:
    arr = np.asarray(probs, dtype=float)
    arr = np.clip(arr, a_min=0.0, a_max=None)
    total = float(arr.sum())
    if total > 0:
        return arr / total
    if arr.size == 0:
        return arr
    return np.full_like(arr, 1.0 / arr.size)


def sample_batch_simulation(
    sample_size: int,
    tokenizer: Any,
    max_sequence_len: int,
    model: Any,
    batch_size: int,
    start_token: str,
    end_token: str,
    petri_net: Optional[Tuple[PetriNet, Marking, Marking]] = None,
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

    net, initial_marking, final_marking = petri_net

    print("index_word: ", index_word)

    token_labels = {
        index: index_word[index].split("==")[0]
        for index in index_word
    }

    concept_index = {v: k for k, v in index_word.items()}

    print("index_word", index_word)
    print("token_labels", token_labels)
    print("concept_index", concept_index)

    for transition in net.transitions:
        if transition.label is not None and transition.label not in concept_index:
            print(f"Marking transition '{transition.label}' as silent")
            transition.label = None

    visible_transitions, silent_transition_map = analyze_transitions_with_end_labels(net, final_marking)

    filtered_transitions = []
    for t in visible_transitions:
        if t.label in concept_index:
            filtered_transitions.append(t)
        else:
            print(f"Transition '{t.label}' not found. Prob set to 0")

    visible_transitions = filtered_transitions

    while len(sequences_events) < sample_size:
        current_batch = min(int(batch_size), sample_size - len(sequences_events))
        if current_batch <= 0:
            break


        last_marking_per_index = [initial_marking.copy() for _ in range(current_batch)]

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
            print("event_histories: ", event_histories)
            print("event_pred: ", event_pred)

            for idx in range(current_batch):
                if not active[idx]:
                    continue

                valid_tokens, currently_enabled_transitions = get_enabled_tokens_and_transitions(
                    net,
                    last_marking_per_index[idx],
                    visible_transitions,
                    silent_transition_map,
                    concept_index,
                )

                print("idx", idx)
                print("valid_tokens", valid_tokens)

                # If last marking is reached, end the trace.
                if last_marking_per_index[idx] == final_marking:
                    end_time = float(np.clip(time_pred[idx][0], 0.0, 1.0))
                    event_histories[idx].append(end_id)
                    time_histories[idx].append(end_time)
                    active[idx] = False
                    continue

                probs = event_pred[idx].astype(float)
                if probs.size == 0:
                    active[idx] = False
                    continue
                probs[0] = 0.0  # never sample padding token
                probs = _safe_normalize(probs)
                print("probs", probs)

                next_id: int | None = None
                while True:
                    print("currently_enabled_transitions", currently_enabled_transitions)
                    filtered_probabilities = get_filtered_probabilities(valid_tokens, probs)
                    filtered_probabilities = np.array(filtered_probabilities) / np.sum(filtered_probabilities)
                    selected_idx = np.random.choice(len(valid_tokens), p=filtered_probabilities)
                    print("selected_idx: ", selected_idx)
                    next_word_index = valid_tokens[selected_idx]

                    if not isinstance(next_word_index, list):
                        last_marking_per_index[idx] = execute_transition(
                            currently_enabled_transitions[selected_idx],
                            net,
                            last_marking_per_index[idx],
                        )
                        next_id = int(next_word_index)
                        break

                    # Silent transition: advance marking without emitting a token.
                    last_marking_per_index[idx] = execute_transition(
                        currently_enabled_transitions[selected_idx],
                        net,
                        last_marking_per_index[idx],
                    )

                    # If silent moves reach the final marking, emit END to stop.
                    if last_marking_per_index[idx] == final_marking:
                        next_id = end_id
                        break

                    valid_tokens, currently_enabled_transitions = get_enabled_tokens_and_transitions(
                        net,
                        last_marking_per_index[idx],
                        visible_transitions,
                        silent_transition_map,
                        concept_index
                    )
                    print(last_marking_per_index[idx])
                    print(final_marking)

                if next_id is None:
                    active[idx] = False
                    continue

                next_time = float(np.clip(time_pred[idx][0], 0.0, 1.0))
                print("next_id: ", next_id)

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