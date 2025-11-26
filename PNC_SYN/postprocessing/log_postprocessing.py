from __future__ import annotations

import datetime
from collections.abc import Sequence
from typing import Any
from xml.etree import ElementTree as StdlibET

import numpy as np
import pandas as pd
from defusedxml import ElementTree as SafeET
from scipy.stats import norm

from PNC_SYN.preprocessing.log_preprocessing import END_TOKEN, START_TOKEN

XES_NAMESPACE = "http://www.xes-standard.org/"
NS = {"": XES_NAMESPACE}
NA_VALUES = {
    "",
    "NA",
    "nan",
    "NaN",
    "null",
    "NULL",
    "<NA>",
    "NaT",
    "&lt;NA&gt;",
    "&lt;nan&gt;",
    "&lt;NaN&gt;",
    "&lt;null&gt;",
    "&lt;NULL&gt;",
    "&lt;NaT&gt;",
}


def clean_xes_file(xml_file: str, output_file: str) -> None:
    """
    Clean XES file by removing empty strings, NA values, and HTML-encoded NA strings.

    Parameters:
    xml_file (str): Path to input XES file
    output_file (str): Path to output cleaned XES file
    """
    tree = SafeET.parse(xml_file)
    root = tree.getroot()
    StdlibET.register_namespace("", XES_NAMESPACE)

    for event in root.findall(".//event", NS):
        to_remove = []
        for elem in event:
            value = elem.get("value", "").strip()
            if value.upper() in {x.upper() for x in NA_VALUES}:
                to_remove.append(elem)

        for elem in to_remove:
            event.remove(elem)

    tree.write(output_file, encoding="utf-8", xml_declaration=True)


def generate_df(
    event_sequences: Sequence[Sequence[str]],
    time_sequences: Sequence[Sequence[float]],
    dict_dtypes: dict[str, dict[str, Any]],
    start_epoch: Sequence[float],
) -> pd.DataFrame:
    """
    Generate a DataFrame from synthetic dual-stream sequences.

    Parameters:
    event_sequences: List of event token sequences (including START/END).
    time_sequences: List of scaled time-delta sequences aligned with events.
    dict_dtypes: Dictionary containing datatype metadata and scaler info.
    start_epoch: Statistical bounds for sampling the initial timestamp.

    Returns:
    pd.DataFrame: Generated DataFrame.
    """
    print("Creating DF-Event Log from synthetic data")
    scaler = dict_dtypes.get("time_scaler", {}) if isinstance(dict_dtypes, dict) else {}
    scale_min = float((scaler or {}).get("min", 0.0) or 0.0)
    scale_max = float((scaler or {}).get("max", 1.0) or 1.0)
    scale_range = max(scale_max - scale_min, 0.0)

    transformed_sentences = []
    for idx, (events, times) in enumerate(zip(event_sequences, time_sequences), start=1):
        if not events or not times or len(events) != len(times):
            continue
        epoch = create_start_epoch(start_epoch)
        case_id = f"case_{idx}"
        sentence: list[str] = [f"case:concept:name=={case_id}"]

        for event_label, delta in zip(events[1:-1], times[1:-1]):
            if not event_label or event_label in {START_TOKEN, END_TOKEN}:
                continue
            seconds = scale_min + delta * scale_range if scale_range > 0 else scale_min
            seconds = max(seconds, 0.0)
            epoch = epoch + datetime.timedelta(seconds=seconds)
            timestamp = epoch.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
            sentence.append(f"concept:name=={event_label}")
            sentence.append(f"time:timestamp=={timestamp}")

        transformed_sentences.append(sentence)

    df = create_dataframe_from_sentences(transformed_sentences, dict_dtypes)
    df = reorder_and_sort_df(df)
    return df


def reorder_and_sort_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort and re-order standard process columns if they exist.

    The DataFrame is sorted by ``case:concept:name`` and ``time:timestamp`` when
    both columns are present. Those columns plus ``concept:name`` are then
    moved to the beginning in the canonical order.

    Parameters:
    df (pd.DataFrame): DataFrame to be reordered and sorted.

    Returns:
    pd.DataFrame: Reordered and sorted DataFrame.
    """
    if "case:concept:name" in df.columns and "time:timestamp" in df.columns:
        df.sort_values(by=["case:concept:name", "time:timestamp"], inplace=True)

    columns_order = []
    if "case:concept:name" in df.columns:
        columns_order.append("case:concept:name")
    if "concept:name" in df.columns:
        columns_order.append("concept:name")
    if "time:timestamp" in df.columns:
        columns_order.append("time:timestamp")

    other_columns = [col for col in df.columns if col not in columns_order]
    df = df[columns_order + other_columns]

    return df


def create_start_epoch(start_epoch: Sequence[float]) -> datetime.datetime:
    """
    Sample a starting epoch constrained by the provided bounds.

    The helper draws from a normal distribution with the supplied mean and
    standard deviation and retries until the value lies within ``[min, max]``.

    Parameters:
    start_epoch (list[float]): ``[mean, std, min_bound, max_bound]``.

    Returns:
    datetime.datetime: Start epoch as a datetime object.
    """
    mean, std, min_bound, max_bound = start_epoch
    epoch_dist = norm(loc=mean, scale=std)

    while True:
        epoch_value = epoch_dist.rvs(1)[0]

        if min_bound <= epoch_value <= max_bound:
            break

    epoch = datetime.datetime.fromtimestamp(epoch_value)
    return epoch


def create_dataframe_from_sentences(
    transformed_sentences: Sequence[Sequence[str]], dict_dtypes: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    """
    Build a pandas DataFrame from transformed synthetic sentences.

    The parser reconstructs case-level dictionaries per trace, applies the
    configured dtypes, sorts traces chronologically, and interpolates missing
    timestamps.

    Parameters:
    transformed_sentences: Enriched synthetic sentences.
    dict_dtypes: Dictionary of attribute dtypes (under ``attribute_datatypes``).

    Returns:
    pd.DataFrame: DataFrame created from the synthetic sentences.
    """
    parsed_data = []
    removed_traces = 0

    for sentence in transformed_sentences:
        try:
            case_dict = {
                word.split("==")[0]: word.split("==")[1]
                for word in sentence
                if word.split("==")[0].startswith("case:")
            }
            event_indices = [i for i, s in enumerate(sentence) if s.startswith("concept:name")]
            event_indices.pop(0)
            events = np.split(sentence, event_indices)
            event_dict_list = []
            for event in events:
                event_dict = {word.split("==")[0]: word.split("==")[1] for word in event}
                event_dict.update(case_dict)
                event_dict_list.append(event_dict)
            parsed_data.append(event_dict_list)
        except Exception:
            removed_traces += 1

    df = pd.DataFrame()
    for case in parsed_data:
        df = pd.concat([df, pd.DataFrame(case)], ignore_index=True)

    dtype_mapping = dict_dtypes["attribute_datatypes"]

    for key, value in dtype_mapping.items():
        if key in df.columns:
            df[key] = convert_column_dtype(df[key], value)

    if "time:timestamp" not in df.columns:
        df["time:timestamp"] = [
            pd.Timestamp("2000-01-01").strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
        ] * len(df)
    else:
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce")

    df.sort_values(by=["case:concept:name", "time:timestamp"], inplace=True)
    df["time:timestamp"] = df.groupby("case:concept:name")["time:timestamp"].transform(
        lambda x: x.interpolate(method="ffill")
    )
    df["time:timestamp"] = df.groupby("case:concept:name")["time:timestamp"].transform(
        lambda x: x.ffill() if pd.isna(x.iloc[0]) else x
    )

    df = df.replace("nan", "")

    return df


def convert_column_dtype(column: pd.Series, dtype: str) -> pd.Series:
    """
    Convert a pandas Series to specified dtype with proper NA handling.

    Parameters:
    column (pd.Series): Column to convert
    dtype (str): Target data type

    Returns:
    pd.Series: Converted column
    """
    type_converters = {
        "int64": lambda col: pd.to_numeric(
            col.replace(["", "nan", "NaN", "NULL", "null"], np.nan), errors="coerce"
        ).astype("Int64"),
        "float": lambda col: col.astype(float) if col.name != "time:timestamp" else col.astype(str),
        "float64": lambda col: col.astype(float)
        if col.name != "time:timestamp"
        else col.astype(str),
        "boolean": lambda col: col.astype(bool),
        "date": lambda col: col.astype(str),
        "string": lambda col: col.astype(str),
        "object": lambda col: col.astype(str),
    }

    converter = type_converters.get(dtype)
    return converter(column) if converter else column
