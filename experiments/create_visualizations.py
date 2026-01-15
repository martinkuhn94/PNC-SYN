from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter


METRIC_SUFFIXES = {
    "Fitness": "Inductive_Fitness",
    "Precision": "Inductive_Precision",
    "Generalization": "Inductive_Generalization",
    "Simplicity": "Inductive_Simplicity",
}
CATEGORY_KEYS = ["real", "synthetic", "conditional_synthetic"]
CATEGORY_LABELS = [r"$L_{Real}$", r"$L_{AR}$", r"$L_{\mathrm{PNC}}$"]
CATEGORY_LABELS_CSV = ["L_Real", "L_AR", "L_PNC"]
BOXPLOT_Y_TICK_COUNT = 5

# Match the yellow -> light blue gradient from code_pm_utility.txt.
_base_cmap = plt.get_cmap("YlGnBu")
_cmap_colors = _base_cmap(np.linspace(0, 0.82, 256))
YL_TO_LIGHTBLUE = ListedColormap(_cmap_colors)
YL_TO_LIGHTBLUE_REVERSED = ListedColormap(_cmap_colors[::-1])
PALETTE = [YL_TO_LIGHTBLUE(x) for x in np.linspace(0.15, 0.75, 3)]


def apply_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")

    plt.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.1,
            "grid.alpha": 0.5,
            "grid.linestyle": "--",
            "figure.dpi": 100,
        }
    )


def find_result_files(search_dir: Path) -> tuple[Path, Path]:
    if not search_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {search_dir}")

    candidates = list(search_dir.rglob("*.xlsx"))
    if not candidates:
        raise FileNotFoundError(f"No Excel files found under: {search_dir}")

    normal = [path for path in candidates if "normal" in path.name.lower()]
    conditional = [path for path in candidates if "conditional" in path.name.lower()]

    if not normal or not conditional:
        raise FileNotFoundError(
            "Expected one normal and one conditional Excel file. "
            f"Found normal={len(normal)} conditional={len(conditional)} in {search_dir}"
        )

    normal_path = max(normal, key=lambda path: path.stat().st_mtime)
    conditional_path = max(conditional, key=lambda path: path.stat().st_mtime)
    return normal_path, conditional_path


def pick_search_dir(base_dir: Path) -> Path:
    for candidate in ("final_results", "final experiments", "final_experiments"):
        target = base_dir / candidate
        if target.exists():
            return target
    return base_dir


def validate_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {label}: {missing}")


def collect_metric_series(
    normal_df: pd.DataFrame, conditional_df: pd.DataFrame
) -> dict[str, dict[str, pd.Series]]:
    metrics: dict[str, dict[str, pd.Series]] = {}
    for metric_name, suffix in METRIC_SUFFIXES.items():
        real_col = f"real_{suffix}"
        synthetic_col = f"synthetic_{suffix}"

        validate_columns(normal_df, [real_col, synthetic_col], "normal file")
        validate_columns(conditional_df, [synthetic_col], "conditional file")

        metrics[metric_name] = {
            "real": normal_df[real_col].dropna(),
            "synthetic": normal_df[synthetic_col].dropna(),
            "conditional_synthetic": conditional_df[synthetic_col].dropna(),
        }

    return metrics


def build_heatmap_data(
    normal_df: pd.DataFrame, conditional_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    required_normal = [
        "num_events_real",
        "num_events_synthetic",
        "trace_variants_real",
        "trace_variants_synthetic",
        "tv_statistic_event_distribution",
        "hellinger_distance_trace_length_distribution",
        "ks_statistic_throughput_time_distribution",
    ]
    required_conditional = [
        "num_events_synthetic",
        "trace_variants_synthetic",
        "tv_statistic_event_distribution",
        "hellinger_distance_trace_length_distribution",
        "ks_statistic_throughput_time_distribution",
    ]

    validate_columns(normal_df, required_normal, "normal file")
    validate_columns(conditional_df, required_conditional, "conditional file")

    def mean_std(df: pd.DataFrame, column: str) -> tuple[float, float]:
        values = pd.to_numeric(df[column], errors="coerce")
        return float(values.mean()), float(values.std(ddof=1))

    num_events_real, num_events_real_std = mean_std(normal_df, "num_events_real")
    num_events_synth_ar, num_events_synth_ar_std = mean_std(
        normal_df, "num_events_synthetic"
    )
    num_events_synth_cond, num_events_synth_cond_std = mean_std(
        conditional_df, "num_events_synthetic"
    )

    trace_real, trace_real_std = mean_std(normal_df, "trace_variants_real")
    trace_synth_ar, trace_synth_ar_std = mean_std(
        normal_df, "trace_variants_synthetic"
    )
    trace_synth_cond, trace_synth_cond_std = mean_std(
        conditional_df, "trace_variants_synthetic"
    )

    tv_ar, tv_ar_std = mean_std(normal_df, "tv_statistic_event_distribution")
    tv_cond, tv_cond_std = mean_std(conditional_df, "tv_statistic_event_distribution")
    hellinger_ar, hellinger_ar_std = mean_std(
        normal_df, "hellinger_distance_trace_length_distribution"
    )
    hellinger_cond, hellinger_cond_std = mean_std(
        conditional_df, "hellinger_distance_trace_length_distribution"
    )
    ks_ar, ks_ar_std = mean_std(normal_df, "ks_statistic_throughput_time_distribution")
    ks_cond, ks_cond_std = mean_std(
        conditional_df, "ks_statistic_throughput_time_distribution"
    )

    values = np.array(
        [
            [num_events_real, num_events_synth_ar, num_events_synth_cond],
            [trace_real, trace_synth_ar, trace_synth_cond],
            [1.0, tv_ar, tv_cond],
            [1.0, hellinger_ar, hellinger_cond],
            [1.0, ks_ar, ks_cond],
        ],
        dtype=float,
    )
    stds = np.array(
        [
            [num_events_real_std, num_events_synth_ar_std, num_events_synth_cond_std],
            [trace_real_std, trace_synth_ar_std, trace_synth_cond_std],
            [0.0, tv_ar_std, tv_cond_std],
            [0.0, hellinger_ar_std, hellinger_cond_std],
            [0.0, ks_ar_std, ks_cond_std],
        ],
        dtype=float,
    )

    row_labels = [
        "Number of events",
        "Number of traces",
        "Event distribution",
        "Trace length",
        "Throughput time",
    ]
    col_labels = CATEGORY_LABELS
    return values, stds, row_labels, col_labels


def format_cell_value(value: float) -> str:
    if np.isnan(value):
        return ""
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def format_cell_stat(value: float, std: float, show_std: bool) -> str:
    if np.isnan(value):
        return ""
    if not show_std:
        return format_cell_value(value)
    return f"{format_cell_value(value)} +/- {format_cell_value(std)}"


def summarize_boxplot_metrics(
    metrics: dict[str, dict[str, pd.Series]]
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for metric_name, series_map in metrics.items():
        for key, label in zip(CATEGORY_KEYS, CATEGORY_LABELS_CSV, strict=False):
            values = pd.to_numeric(series_map[key], errors="coerce")
            rows.append(
                {
                    "metric": metric_name,
                    "category": label,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                }
            )
    return pd.DataFrame(rows)


def summarize_heatmap_metrics(
    values: np.ndarray,
    stds: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for row_idx, row_label in enumerate(row_labels):
        for col_idx, col_label in enumerate(col_labels):
            rows.append(
                {
                    "metric": row_label,
                    "category": col_label,
                    "mean": float(values[row_idx, col_idx]),
                    "std": float(stds[row_idx, col_idx]),
                }
            )
    return pd.DataFrame(rows)


def plot_metric(
    ax: plt.Axes,
    metric_name: str,
    series_map: dict[str, pd.Series],
) -> None:
    data = [series_map[key].dropna().values for key in CATEGORY_KEYS]
    boxplot = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.6,
        tick_labels=CATEGORY_LABELS,
        medianprops={"color": "#222222", "linewidth": 1.4},
        boxprops={"linewidth": 1.2, "edgecolor": "#333333"},
        whiskerprops={"linewidth": 1.1, "color": "#333333"},
        capprops={"linewidth": 1.1, "color": "#333333"},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#333333",
            "markeredgecolor": "#333333",
            "markersize": 3,
            "alpha": 0.4,
        },
    )
    for patch, color in zip(boxplot["boxes"], PALETTE, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel(metric_name, fontsize=14, fontweight="bold")
    ax.grid(True, axis="y")
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    y_min, y_max = ax.get_ylim()
    if y_min != y_max:
        ax.set_yticks(np.linspace(y_min, y_max, BOXPLOT_Y_TICK_COUNT))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def build_heatmap_colors(values: np.ndarray) -> np.ndarray:
    colors = values.copy()
    for row_idx in (0, 1):
        real = values[row_idx, 0]
        if not np.isfinite(real) or real == 0:
            colors[row_idx] = np.nan
            continue
        colors[row_idx] = 1 - np.abs(values[row_idx] - real) / real
    return np.clip(colors, 0.0, 1.0)


def plot_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    stds: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
) -> None:
    color_values = build_heatmap_colors(values)
    masked = np.ma.masked_invalid(color_values)
    YL_TO_LIGHTBLUE_REVERSED.set_bad("#F2F2F2")

    x_edges = np.arange(values.shape[1] + 1)
    y_edges = np.arange(values.shape[0] + 1)
    image = ax.pcolormesh(
        x_edges,
        y_edges,
        masked,
        cmap=YL_TO_LIGHTBLUE_REVERSED,
        vmin=0.0,
        vmax=1.0,
        shading="flat",
        edgecolors="none",
        linewidth=0,
        antialiased=False,
    )
    ax.set_xticks(np.arange(len(col_labels)) + 0.5, labels=col_labels)
    ax.set_yticks(np.arange(len(row_labels)) + 0.5, labels=row_labels)
    ax.set_xlim(0, values.shape[1])
    ax.set_ylim(values.shape[0], 0)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="both", length=0)
    ax.grid(False)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            cell_value = values[row_idx, col_idx]
            if np.isnan(cell_value):
                continue
            cell_std = stds[row_idx, col_idx]
            show_std = col_idx != 0
            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                format_cell_stat(cell_value, cell_std, show_std),
                ha="center",
                va="center",
                fontsize=14,
                color="#1A1A1A",
            )

    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)

    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.set_label("Resemblance Score", fontsize=15, fontweight="bold")
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    cbar.ax.tick_params(labelsize=13)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    search_dir = pick_search_dir(base_dir)
    normal_path, conditional_path = find_result_files(search_dir)

    apply_plot_style()

    print(f"Using normal file: {normal_path}")
    print(f"Using conditional file: {conditional_path}")

    normal_df = pd.read_excel(normal_path)
    conditional_df = pd.read_excel(conditional_path)

    metrics = collect_metric_series(normal_df, conditional_df)

    output_dir = search_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    boxplot_summary = summarize_boxplot_metrics(metrics)
    boxplot_summary.to_csv(output_dir / "boxplot_summary.csv", index=False)

    for metric_name, series_map in metrics.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_metric(ax, metric_name, series_map)
        fig.tight_layout()
        filename = f"boxplot_{metric_name.lower()}.png".replace(" ", "_")
        fig.savefig(output_dir / filename, dpi=300)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (metric_name, series_map) in zip(axes.ravel(), metrics.items()):
        plot_metric(ax, metric_name, series_map)
    fig.tight_layout()
    fig.savefig(output_dir / "boxplot_inductive_metrics_grid.png", dpi=300)
    plt.close(fig)

    heatmap_values, heatmap_stds, heatmap_rows, heatmap_cols = build_heatmap_data(
        normal_df, conditional_df
    )
    heatmap_summary = summarize_heatmap_metrics(
        heatmap_values, heatmap_stds, heatmap_rows, CATEGORY_LABELS_CSV
    )
    heatmap_summary.to_csv(output_dir / "heatmap_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_heatmap(ax, heatmap_values, heatmap_stds, heatmap_rows, heatmap_cols)
    fig.tight_layout()
    fig.savefig(output_dir / "heatmap_process_metrics.png", dpi=300)
    plt.close(fig)

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
