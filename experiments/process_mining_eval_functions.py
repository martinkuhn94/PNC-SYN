from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pm4py
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator


def event_distribution(log: Any) -> pd.Series:
    """Return the count of each event type in a PM4Py event log as a pandas Series.

    :param log: The PM4Py event log.
    :type log: pm4py.objects.log.log.EventLog
    :returns: The count of each event type in the event log.
    :rtype: pd.Series
    """
    df = pm4py.convert_to_dataframe(log)
    count_data = df["concept:name"].value_counts()

    return count_data


def calculate_trace_length_distribution(log: Any) -> pd.Series:
    """Return the count of each trace length in a PM4Py event log as a pandas Series.

    :param log: The PM4Py event log.
    :type log: pm4py.objects.log.log.EventLog
    :returns: The count of each trace length in the event log.
    :rtype: pd.Series
    """
    df = pm4py.convert_to_dataframe(log)
    count_data = df["case:concept:name"].value_counts()
    # Get the count of each trace length
    count_data = count_data.value_counts()
    # Sort the series by the index
    count_data = count_data.sort_index()

    print(count_data)

    # Make index numeric
    # count_data.index = pd.to_numeric(count_data.index)

    return count_data


def calc_hellinger(real_data: Any, synthetic_data: Any, input_type: str = "column") -> float:
    """
    Calculate Hellinger distance between two distributions or columns.

    Args:
        data1: First distribution/column (pandas Series or column)
        data2: Second distribution/column (pandas Series or column)

    Returns:
        float: Hellinger distance
    """
    if input_type == "column":
        dist1 = real_data.value_counts()
        dist2 = synthetic_data.value_counts()
    else:
        dist1 = real_data
        dist2 = synthetic_data

    # Align distributions
    all_indices = sorted(set(dist1.index) | set(dist2.index))
    dist1_aligned = pd.Series(0, index=all_indices)
    dist2_aligned = pd.Series(0, index=all_indices)

    dist1_aligned[dist1.index] = dist1
    dist2_aligned[dist2.index] = dist2

    # Convert to probabilities
    p1 = dist1_aligned / dist1_aligned.sum()
    p2 = dist2_aligned / dist2_aligned.sum()

    hellinger = np.sqrt(np.sum((np.sqrt(p1.values) - np.sqrt(p2.values)) ** 2)) / np.sqrt(2.0)
    return float(hellinger)


def calculate_throughput_time(log: Any) -> list[float]:
    """Calculate per-trace throughput time (first to last timestamp)."""
    # Remove all timestamps that are NaN
    df = pm4py.convert_to_dataframe(log)
    df = df.dropna(subset=["time:timestamp"])
    # Transform df back to log
    log = pm4py.convert_to_event_log(df)

    all_case_durations = pm4py.get_all_case_durations(log)

    return [float(duration) for duration in all_case_durations]


def calculate_relative_timestamps(log: Any) -> list[float]:
    """
    Calculate relative timestamps for each event in a log.

    The timestamps are normalized per trace so that the first event is 0.0 and
    the last event is 1.0. Traces with zero duration are assigned 0.0 for all
    contained events.
    """
    df = pm4py.convert_to_dataframe(log)
    if df.empty:
        return []

    required_columns = {"case:concept:name", "time:timestamp"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns for relative timestamps: {missing_columns}")

    df = df.dropna(subset=["case:concept:name", "time:timestamp"]).copy()
    if df.empty:
        return []

    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce")
    df = df.dropna(subset=["time:timestamp"])
    if df.empty:
        return []

    df.sort_values(by=["case:concept:name", "time:timestamp"], inplace=True)

    relative_timestamps: list[float] = []
    grouped = df.groupby("case:concept:name", sort=False)
    for _, group in grouped:
        timestamps = group["time:timestamp"]
        if timestamps.empty:
            continue
        start_time = timestamps.iloc[0]
        end_time = timestamps.iloc[-1]
        duration = (end_time - start_time).total_seconds()

        if duration <= 0 or np.isclose(duration, 0.0):
            relative_timestamps.extend([0.0] * len(timestamps))
            continue

        normalized = (timestamps - start_time).dt.total_seconds() / duration
        normalized = normalized.clip(lower=0.0, upper=1.0)
        relative_timestamps.extend(normalized.astype(float).tolist())

    return relative_timestamps


def calculate_earth_mover_distance(real_log, synthetic_log):
    """Calculate the earth mover distance between two event logs. The earth mover distance is defined as the
    minimum cost of turning one distribution into the other.

    :param real_log: The real event log.
    :type real_log: pm4py.objects.log.log.EventLog
    :param synthetic_log: The synthetic event log.
    :type synthetic_log: pm4py.objects.log.log.EventLog
    :return: The earth mover distance between the two event logs.
    :rtype: float
    """
    language_log_real = pm4py.get_stochastic_language(real_log)
    language_log_synthetic = pm4py.get_stochastic_language(synthetic_log)

    earth_mover_distance = pm4py.compute_emd(language_log_real, language_log_synthetic)
    print(earth_mover_distance)

    return 1 - earth_mover_distance


def calculate_petri_nets(log: Any, threshold: float | None = None) -> dict[str, list[Any]]:
    """
    Discover Petri nets using inductive and heuristic mining algorithms.

    :param threshold: Dependency threshold for heuristic mining (``None`` uses PM4Py default).
    :param log: Event log to analyze.
    :return: Mapping of miner name to (net, initial marking, final marking).
    """
    net_inductive, initial_marking_inductive, final_marking_inductive = (
        pm4py.discover_petri_net_inductive(log)
    )
    heuristics_kwargs: dict[str, Any] = {}
    if threshold is not None:
        heuristics_kwargs["dependency_threshold"] = threshold
    net_heuristics, initial_marking_heuristics, final_marking_heuristics = (
        pm4py.discover_petri_net_heuristics(log, **heuristics_kwargs)
    )
    petri_net_list = [
        [net_inductive, initial_marking_inductive, final_marking_inductive],
        [net_heuristics, initial_marking_heuristics, final_marking_heuristics],
    ]
    petri_net_list_names = ["Inductive", "Heuristics"]
    petri_net_dict = dict(zip(petri_net_list_names, petri_net_list))
    return petri_net_dict


def compare_logs(real_event_log, synthetic_event_log, threshold: float | None = None):
    """Compare real and synthetic logs using token-based replay metrics.

    Petri nets discovered from the real log are reused to evaluate how well the
    synthetic log fits the original process using token-based replay metrics.

    :param real_event_log: The real event log.
    :type real_event_log: pm4py.objects.log.log.EventLog
    :param synthetic_event_log: The synthetic event log.
    :type synthetic_event_log: pm4py.objects.log.log.EventLog
    :param threshold: Optional dependency threshold for heuristic mining.
    :type threshold: float | None
    :return: Dictionary containing results for both real and synthetic data with prefixed keys
    """
    petri_net_dict = calculate_petri_nets(real_event_log, threshold)
    petri_net_dict_synth = calculate_petri_nets(synthetic_event_log, threshold)

    results = {}

    metrics = ["Fitness", "Precision", "Generalization", "Simplicity"]

    # Calculate metrics for real event log
    for key, petri_net in petri_net_dict.items():
        net, initial_marking, final_marking = petri_net
        fitness_info = pm4py.fitness_token_based_replay(
            real_event_log, net, initial_marking, final_marking
        )
        fitness = float(fitness_info.get("average_trace_fitness", 0.0))
        prec = float(
            pm4py.precision_token_based_replay(real_event_log, net, initial_marking, final_marking)
        )
        gen = float(generalization_evaluator.apply(real_event_log, net, initial_marking, final_marking))
        simp = float(simplicity_evaluator.apply(net))

        results[f"real_{key}_Fitness"] = fitness
        results[f"real_{key}_Precision"] = prec
        results[f"real_{key}_Generalization"] = gen
        results[f"real_{key}_Simplicity"] = simp

    # Calculate metrics for synthetic event log using the same Petri nets
    for key, petri_net in petri_net_dict.items():
        net, initial_marking, final_marking = petri_net
        fitness_info = pm4py.fitness_token_based_replay(
            synthetic_event_log, net, initial_marking, final_marking
        )
        fitness = float(fitness_info.get("average_trace_fitness", 0.0))
        prec = float(
            pm4py.precision_token_based_replay(
                synthetic_event_log, net, initial_marking, final_marking
            )
        )
        gen = float(
            generalization_evaluator.apply(synthetic_event_log, net, initial_marking, final_marking)
        )
        simp = float(simplicity_evaluator.apply(petri_net_dict_synth[key][0]))

        results[f"synthetic_{key}_Fitness"] = fitness
        results[f"synthetic_{key}_Precision"] = prec
        results[f"synthetic_{key}_Generalization"] = gen
        results[f"synthetic_{key}_Simplicity"] = simp

    # Calculate absolute differences
    for key in petri_net_dict.keys():
        for metric in metrics:
            real_value = results.get(f"real_{key}_{metric}")
            synthetic_value = results.get(f"synthetic_{key}_{metric}")
            if real_value is not None and synthetic_value is not None:
                results[f"abs_diff_{key}_{metric}"] = abs(real_value - synthetic_value)
            else:
                results[f"abs_diff_{key}_{metric}"] = None

    return results
