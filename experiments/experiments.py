from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pm4py
import yaml
from sdmetrics.single_column import KSComplement

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_PATH = Path(__file__).with_name("sepsis_cases_experiment_config.yaml")

from process_mining_eval_functions import (  # noqa: E402
    calc_hellinger,
    calculate_relative_timestamps,
    calculate_throughput_time,
    calculate_trace_length_distribution,
    compare_logs,
)
from PNC_SYN.postprocessing.log_postprocessing import clean_xes_file  # noqa: E402
from PNC_SYN.synthesizer import PNCEventLogSynthesizer  # noqa: E402

REQUIRED_CONFIG_KEYS = (
    "event_log_path",
    "methods",
    "units_per_layer",
    "num_epochs",
    "breakpoint_interval",
    "embedding_output_dims",
    "dropout",
    "trace_quantile",
    "sample_size",
    "batch_size",
    "petri_net_keys",
    "process_metrics",
    "result_columns",
    "models_dir",
    "synthetic_log_dir",
)


def resolve_path(path_value: Path | str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def normalize_units_config(units_config) -> list[list[int]]:
    normalized = []
    for entry in units_config:
        if isinstance(entry, int):
            normalized.append([entry])
        elif isinstance(entry, (list, tuple)):
            normalized.append([int(value) for value in entry])
        else:
            raise ValueError(
                "units_per_layer entries must be integers or iterables of integers."
            )
    return normalized


def load_experiment_config(config_path: Path | str) -> dict:
    path = resolve_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise ValueError("Experiment config must be a YAML mapping/object.")

    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing_keys:
        raise KeyError(f"Experiment config missing required keys: {missing_keys}")
    return config


def run_experiments(config: dict) -> None:
    event_log_path = resolve_path(config["event_log_path"])
    event_log_name = str(event_log_path)
    event_log_label = event_log_path.stem

    methods = list(config["methods"])
    units_configurations = normalize_units_config(config["units_per_layer"])
    num_epochs = int(config["num_epochs"])
    breakpoint_interval = int(config["breakpoint_interval"])
    embedding_output_dims = int(config["embedding_output_dims"])
    dropout = float(config["dropout"])
    trace_quantile = float(config["trace_quantile"])
    sample_size = int(config["sample_size"])
    batch_size = int(config["batch_size"])
    petri_net_keys = tuple(config["petri_net_keys"])
    process_metric_names = tuple(config["process_metrics"])
    result_columns = list(config["result_columns"])

    models_dir = resolve_path(config["models_dir"])
    synthetic_log_dir = resolve_path(config["synthetic_log_dir"])
    evaluation_output_dir = resolve_path(config.get("evaluation_output_dir", PROJECT_ROOT))
    models_dir.mkdir(parents=True, exist_ok=True)
    synthetic_log_dir.mkdir(parents=True, exist_ok=True)
    evaluation_output_dir.mkdir(parents=True, exist_ok=True)

    event_log_train = pm4py.read_xes(event_log_name)
    df_result_array = []

    for method in methods:
        for unit_config in units_configurations:
            units_label = "-".join(str(value) for value in unit_config)
            model = PNCEventLogSynthesizer(
                embedding_output_dims=embedding_output_dims,
                epochs=num_epochs,
                batch_size=batch_size,
                dropout=dropout,
                trace_quantile=trace_quantile,
                method=method,
                units_per_layer=unit_config,
            )
            model.initialize_model(event_log_train)

            for current_epoch in range(
                breakpoint_interval, num_epochs + breakpoint_interval, breakpoint_interval
            ):
                results = {
                    "method": method,
                    "units": units_label,
                    "epochs": current_epoch,
                }

                start_time = time.time()
                print(
                    f"Training epochs {current_epoch - breakpoint_interval} to "
                    f"{current_epoch}"
                )
                model.train(epochs=breakpoint_interval)
                model_name = f"{method}_{event_log_label}_u={units_label}_ep={current_epoch}"
                model_path = models_dir / model_name
                model.save_model(str(model_path))
                print(f"Model saved at epoch {current_epoch}: {model_path}")

                training_time = time.time() - start_time
                results["training_time"] = training_time

                try:
                    print(
                        f"[{method} | units={units_label} | epoch={current_epoch}] "
                        "Sampling synthetic traces..."
                    )
                    start_time = time.time()
                    event_log_sample = model.sample(sample_size=sample_size, batch_size=batch_size)
                    event_log_xes = pm4py.convert_to_event_log(event_log_sample)
                    sampling_time = time.time() - start_time
                    results["sampling_time"] = sampling_time
                    print(
                        f"[{method} | units={units_label} | epoch={current_epoch}] "
                        f"Sampling completed in {sampling_time:.2f}s"
                    )

                    xes_filename = f"{method}_{event_log_label}_u={units_label}_ep={current_epoch}.xes"
                    xes_path = synthetic_log_dir / xes_filename
                    pm4py.write_xes(event_log_xes, str(xes_path))
                except Exception as exc:
                    print(f"Sampling or serialization failed: {exc}")
                    continue

                print(
                    f"[{method} | units={units_label} | epoch={current_epoch}] "
                    "Cleaning generated XES file..."
                )
                clean_xes_file(str(xes_path), str(xes_path))
                print(
                    f"[{method} | units={units_label} | epoch={current_epoch}] "
                    "Evaluating synthetic log..."
                )

                real_event_log = pm4py.read_xes(event_log_name)
                synthetic_event_log = pm4py.read_xes(str(xes_path))
                df_real = pm4py.convert_to_dataframe(real_event_log)
                df_synthetic = pm4py.convert_to_dataframe(synthetic_event_log)

                data_real = df_real["concept:name"].dropna()
                data_synthetic = df_synthetic["concept:name"].dropna()
                hellinger_distance_events = calc_hellinger(data_real, data_synthetic)
                results["tv_statistic_event_distribution"] = 1 - hellinger_distance_events

                trace_length_real = calculate_trace_length_distribution(real_event_log)
                trace_length_synthetic = calculate_trace_length_distribution(synthetic_event_log)
                hellinger_distance_trace = calc_hellinger(
                    trace_length_real, trace_length_synthetic, input_type="distribution"
                )
                results["hellinger_distance_trace_length_distribution"] = (
                    1 - hellinger_distance_trace
                )

                throughput_time_real = calculate_throughput_time(real_event_log)
                throughput_time_synthetic = calculate_throughput_time(synthetic_event_log)
                ks_statistic = KSComplement.compute(
                    real_data=throughput_time_real, synthetic_data=throughput_time_synthetic
                )
                results["ks_statistic_throughput_time_distribution"] = ks_statistic

                relative_ts_real = calculate_relative_timestamps(real_event_log)
                relative_ts_synthetic = calculate_relative_timestamps(synthetic_event_log)
                ks_statistic_relative = KSComplement.compute(
                    real_data=relative_ts_real, synthetic_data=relative_ts_synthetic
                )
                results["ks_statistic_relative_timestamp_distribution"] = ks_statistic_relative
                print(
                    f"[{method} | units={units_label} | epoch={current_epoch}] "
                    "Evaluation metrics computed."
                )
                print(
                    "  Calculating Process Perspective Metrics "
                    "(Fitness, Precision, Generalization, Simplicity)..."
                )
                try:
                    process_metrics = compare_logs(
                        real_event_log,
                        synthetic_event_log,
                    )
                    results.update(process_metrics)
                    print("  Process Perspective Metrics calculated.")
                except Exception as exc:
                    print(f"  Error calculating Process Perspective Metrics: {exc}")
                    for prefix in ("real", "synthetic"):
                        for key in petri_net_keys:
                            for metric in process_metric_names:
                                results[f"{prefix}_{key}_{metric}"] = None

                results["num_events_real"] = len(df_real)
                results["num_events_synthetic"] = len(df_synthetic)
                variants_real = (
                    df_real.groupby("case:concept:name")["concept:name"].apply(tuple).nunique()
                )
                variants_synthetic = (
                    df_synthetic.groupby("case:concept:name")["concept:name"]
                    .apply(tuple)
                    .nunique()
                )
                results["trace_variants_real"] = variants_real
                results["trace_variants_synthetic"] = variants_synthetic
                df_result_array.append(results)

    if not df_result_array:
        print("No successful experiment runs; skipping result export.")
        return

    df_results = pd.DataFrame(df_result_array)
    for column in result_columns:
        if column not in df_results.columns:
            df_results[column] = np.nan
    df_results["Average"] = df_results[result_columns].mean(axis=1)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_result_{event_log_label}_{timestamp}.xlsx"
    output_path = evaluation_output_dir / filename
    df_results.to_excel(output_path, index=False)
    print(f"Saved evaluation summary to {output_path}")


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else CONFIG_PATH
    config = load_experiment_config(config_path)
    run_experiments(config)


if __name__ == "__main__":
    main()
