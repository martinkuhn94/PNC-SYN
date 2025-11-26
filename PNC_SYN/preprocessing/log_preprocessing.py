from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import pm4py

_CPU_COUNT = os.cpu_count() or 1
os.environ["LOKY_MAX_CPU_COUNT"] = str(max(_CPU_COUNT - 1, 1))

START_TOKEN = "START"  # noqa: S105 - sentinel token marker
END_TOKEN = "END"  # noqa: S105 - sentinel token marker
REQUIRED_COLUMNS = ["case:concept:name", "concept:name", "time:timestamp"]


def calculate_starting_epoch(df: pd.DataFrame) -> list[float]:
    """
    Calculate basic statistics for the first event timestamp per trace.

    Parameters:
    df (pd.DataFrame): Event log containing ``case:concept:name`` and ``time:timestamp``.

    Returns:
    list: ``[mean, std, min, max]`` describing the starting epoch distribution.
    """
    if "case:concept:name" not in df or "time:timestamp" not in df:
        raise ValueError("DataFrame must contain 'case:concept:name' and 'time:timestamp' columns")

    timestamps = pd.to_datetime(df["time:timestamp"], errors="coerce")
    if timestamps.isnull().any():
        raise ValueError("Invalid timestamps detected; please clean the log before preprocessing.")

    df = df.copy()
    df["time:timestamp"] = timestamps
    starting_epochs = (
        df.sort_values(by="time:timestamp")
        .groupby("case:concept:name")["time:timestamp"]
        .first()
    )
    starting_epoch_list = (starting_epochs.astype(np.int64) // 10**9).to_numpy()

    if starting_epoch_list.size == 0:
        raise ValueError("No valid starting timestamps found in the data.")

    starting_epoch_mean = float(np.mean(starting_epoch_list))
    starting_epoch_std = float(np.std(starting_epoch_list))
    starting_epoch_min = float(np.min(starting_epoch_list))
    starting_epoch_max = float(np.max(starting_epoch_list))

    return [starting_epoch_mean, starting_epoch_std, starting_epoch_min, starting_epoch_max]


def calculate_time_between_events(df: pd.DataFrame) -> list[float]:
    """
    Calculate per-trace event deltas for a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): Event log with ``case:concept:name`` and ``time:timestamp``.

    Returns:
    list: Time between events (seconds since epoch) for every trace.
    """
    if "case:concept:name" not in df or "time:timestamp" not in df:
        raise ValueError("DataFrame must contain 'case:concept:name' and 'time:timestamp' columns")

    df = df.copy()
    try:
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
    except Exception as exc:
        raise ValueError("Error converting 'time:timestamp' to datetime") from exc

    time_between_events: list[float] = []

    for _, group in df.groupby("case:concept:name"):
        if len(group) < 2:
            time_between_events.append(0.0)
            continue

        time_diffs = group["time:timestamp"].diff().dt.total_seconds().copy()
        time_diffs.fillna(0.0, inplace=True)
        time_diffs.iloc[0] = 0.0
        time_between_events.extend(time_diffs.astype(float).tolist())

    return time_between_events


def _scale_to_unit_interval(values: list[float]) -> tuple[list[float], float, float]:
    """Scale a list of numeric values to the [0, 1] interval."""
    if not values:
        return [], 0.0, 0.0

    array = np.asarray(values, dtype=float)
    valid_array = array[~np.isnan(array)]
    if valid_array.size == 0:
        return [0.0] * len(values), 0.0, 0.0

    min_val = float(valid_array.min())
    max_val = float(valid_array.max())
    if np.isclose(max_val, min_val):
        scaled = np.zeros_like(array)
    else:
        scaled = (array - min_val) / (max_val - min_val)
    scaled = np.clip(scaled, 0.0, 1.0)

    return scaled.tolist(), min_val, max_val


def preprocess_event_log(
    log: Any, trace_quantile: float
) -> tuple[list[list[str]], list[list[float]], dict[str, dict[str, Any]], list[float], int]:
    """
    Preprocess the raw event log into dual streams (events + scaled time deltas).

    The pipeline keeps only ``case:concept:name``, ``concept:name``, and ``time:timestamp``.
    Per-trace timestamps are converted to deltas, rescaled to the [0, 1] interval,
    and aligned with their corresponding concept tokens. Each trace results in
    ``event_sequence = [START, event_1, ..., END]`` and
    ``time_sequence = [0.0, delta_1, ..., 0.0]``.
    """
    try:
        df = pm4py.convert_to_dataframe(log)
    except Exception as exc:
        raise ValueError("Error converting log to DataFrame") from exc

    if df.empty:
        raise ValueError("The provided event log is empty.")

    print(f"Original columns: {sorted(df.columns.tolist())}")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    extra_columns = sorted(set(df.columns) - set(REQUIRED_COLUMNS))
    if extra_columns:
        print(f"Dropping non-required columns: {extra_columns}")

    df = df[REQUIRED_COLUMNS].copy()
    print("Retained columns:", df.columns.tolist())

    total_traces = df["case:concept:name"].nunique()
    print(f"Number of traces before filtering: {total_traces}")

    trace_length = df.groupby("case:concept:name").size()
    if not trace_length.empty:
        trace_length_q = trace_length.quantile(trace_quantile)
        df = df.groupby("case:concept:name").filter(lambda x: len(x) <= trace_length_q)
        print(f"Trace length quantile ({trace_quantile}): {trace_length_q}")

    filtered_traces = df["case:concept:name"].nunique()
    print(f"Number of traces after filtering: {filtered_traces}")

    if filtered_traces == 0:
        raise ValueError("No traces left after filtering; adjust trace_quantile.")

    df.sort_values(by=["case:concept:name", "time:timestamp"], inplace=True)
    num_examples = len(df)
    print(f"Total events after filtering: {num_examples}")

    df_datetime = df.copy()
    df_datetime["time:timestamp"] = pd.to_datetime(df_datetime["time:timestamp"], errors="coerce")
    if df_datetime["time:timestamp"].isnull().any():
        raise ValueError("Found invalid timestamps; please ensure the log uses ISO-8601 format.")

    starting_epoch_dist = calculate_starting_epoch(df_datetime)
    print(
        "Starting epoch stats:",
        f"mean={starting_epoch_dist[0]:.2f}, std={starting_epoch_dist[1]:.2f}, "
        f"min={starting_epoch_dist[2]:.0f}, max={starting_epoch_dist[3]:.0f}",
    )

    time_between_events = calculate_time_between_events(df_datetime)
    scaled_times, raw_min, raw_max = _scale_to_unit_interval(time_between_events)
    df["time:timestamp"] = scaled_times
    print(f"Raw time deltas min/max: {raw_min:.4f}/{raw_max:.4f}")
    if scaled_times:
        preview = np.round(scaled_times[:5], 5).tolist()
        print(f"First 5 scaled timestamps: {preview}")
    print("Time deltas were rescaled to the [0, 1] interval.")

    attribute_dtype_mapping = {
        "attribute_datatypes": {
            "case:concept:name": "string",
            "concept:name": "string",
            "time:timestamp": "float64",
        },
        "time_scaler": {"min": raw_min, "max": raw_max},
    }

    event_sequences: list[list[str]] = []
    time_sequences: list[list[float]] = []

    grouped_traces = df.groupby("case:concept:name")
    for _, trace_group in grouped_traces:
        events = trace_group["concept:name"].astype(str).tolist()
        deltas = trace_group["time:timestamp"].astype(float).tolist()
        event_seq = [START_TOKEN] + events + [END_TOKEN]
        time_seq = [0.0] + deltas + [0.0]
        event_sequences.append(event_seq)
        time_sequences.append(time_seq)

    if not event_sequences:
        raise ValueError("Event log sequences could not be generated.")

    print(f"Generated {len(event_sequences)} trace sequences.")
    print("Sample event sequence:", event_sequences[0])
    print("Sample time sequence:", [round(x, 6) for x in time_sequences[0]])

    return (
        event_sequences,
        time_sequences,
        attribute_dtype_mapping,
        starting_epoch_dist,
        num_examples,
    )
