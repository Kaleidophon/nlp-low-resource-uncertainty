"""
Plot the development of metrics over the training time.
"""

# STD
import argparse
import os
import re
from typing import List, Optional, Dict

# EXT
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from nlp_uncertainty_zoo.config import AVAILABLE_MODELS
import pandas as pd

# PROJECT
from src.config import AVAILABLE_DATASETS

# CONST
SEED = 123456
ALPHA = 0.6
RESULT_DIR = "./results"
MODEL_DIR = "./models"
IMG_DIR = "./img/time"

# Plotting
MODEL_COLORS = {
    "lstm": ("firebrick", "lightcoral"),
    "lstm_ensemble": ("forestgreen", "yellowgreen"),
    "st_tau_lstm": ("midnightblue", "skyblue"),
    "bayesian_lstm": ("orangered", "lightsalmon"),
    "variational_lstm": ("darkmagenta", "orchid"),
    "ddu_bert": ("lightseagreen", "mediumturquoise"),
    "variational_bert": ("gold", "lemonchiffon"),
    "sngp_bert": ("dimgray", "silver"),
}
MODEL_NAMES = {
    "lstm": "LSTM",
    "lstm_ensemble": "LSTM Ensemble",
    "st_tau_lstm": "ST-tau LSTM",
    "bayesian_lstm": "Bayesian LSTM",
    "variational_lstm": "Variational LSTM",
    "ddu_bert": "DDU Bert",
    "variational_bert": "Variational Bert",
    "sngp_bert": "SNGP Bert",
}
TRAINING_SIZE_LINESTYLES = {
    "dan+": {
        1000: "dotted",
        2000: "dashed",
        4000: "solid",
    },
    "finnish_ud": {5000: "dotted", 7500: "dashed", 10000: "solid"},
    "clinc_plus": {10000: "dotted", 12500: "dashed", 15000: "solid"},
}
plt.style.use("science")


def plot_timeseries(
    data: pd.DataFrame,
    dataset: str,
    target: str,
    target_name: Optional[str] = None,
    step_cutoff: Optional[int] = None,
    colors: Optional[Dict[str, str]] = None,
    metrics: List[str] = None,
    save_path: Optional[str] = None,
):
    fig = plt.figure(figsize=(7.5, 4))
    ax = plt.gca()
    ax.grid(visible=True, axis="both", which="major", linestyle=":", color="grey")

    target_columns = set(
        col.replace("_mean", "").replace("_std", "")
        for col in data.columns
        if target in col
    )

    found_models = set()
    found_training_sizes = set()

    for target_column in target_columns:

        if metrics is None:
            continue

        else:
            if all([metric not in target_column for metric in metrics]):
                continue

        model_name, training_size = (
            re.compile(r"(.+)_(\d+)_.+").match(target_column).groups()
        )
        found_models.add(model_name)
        training_size = int(training_size)
        found_training_sizes.add(training_size)

        x = data[f"{target_column}_mean"].index - 1
        data_means, data_stds = (
            data[f"{target_column}_mean"].values,
            data[f"{target_column}_std"].values,
        )

        if step_cutoff is not None:
            mask = step_cutoff >= x
            x = x[mask]
            data_means = data_means[mask]
            data_stds = data_stds[mask]

        # Plot line
        ax.plot(
            x,
            data_means,
            label=f"{MODEL_NAMES[model_name]} ({training_size})",
            color=colors[model_name][0],
            alpha=0.8,
            markersize=12,
            linestyle=TRAINING_SIZE_LINESTYLES[dataset][training_size],
            linewidth=1.6,
        )

        # Plot +/- standard deviation
        ax.fill_between(
            x=x,
            y1=data_means + data_stds,
            y2=data_means - data_stds,
            color=colors[model_name][1],
            alpha=0.2,
        )

    # plt.xticks(x, xticks, fontsize=12)
    plt.xlabel("Training step", fontsize=16, alpha=0.6)
    plt.tick_params(axis="both", which="major", labelsize=16 - 4)

    if target_name is None:
        plt.ylabel(target, fontsize=16, alpha=0.6)
    else:
        plt.ylabel(target_name, fontsize=16, alpha=0.6)

    # plt.xticks(np.arange(0, len(x)), x, fontsize=12)
    # ax.xaxis.set_ticklabels([])
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Create custom legend
    legend_elements = [
        # Add models
        *[
            Patch(
                facecolor=MODEL_COLORS[model_name][1],
                edgecolor=MODEL_COLORS[model_name][0],
                label=MODEL_NAMES[model_name],
                alpha=ALPHA,
            )
            for model_name in found_models
        ],
        # Add training sizes
        *[
            Line2D(
                [0],
                [0],
                linestyle=TRAINING_SIZE_LINESTYLES[dataset][training_size],
                linewidth=1.5,
                alpha=ALPHA + 0.2,
                color="black",
                label=training_size,
            )
            for training_size in sorted(found_training_sizes)
        ],
    ]
    plt.legend(
        handles=legend_elements,
        loc="lower right",
        ncol=1,
        fontsize=16 - 4,
        handlelength=1,
        facecolor="white",
        framealpha=0.8,
        frameon=True,
    )

    fig.tight_layout()

    if not save_path:
        plt.show()

    else:
        plt.savefig(
            save_path,
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=AVAILABLE_DATASETS.keys(),
        help="Dataset to run experiments on.",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Metric to plot.",
    )
    parser.add_argument(
        "--target-name",
        type=str,
        help="Name of metric to plot for display purposes",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=AVAILABLE_MODELS.keys(),
        nargs="+",
        choices=AVAILABLE_MODELS.keys(),
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        nargs="+",
    )
    parser.add_argument("--identifier", type=str, default="")
    parser.add_argument("--training-sizes", type=int, nargs="+", default=tuple())
    parser.add_argument("--step-cutoff", type=int, default=None)
    parser.add_argument("--result-dir", type=str, default=RESULT_DIR)
    parser.add_argument("--output-dir", type=str, default=IMG_DIR)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Filter irrelevant files
    # Retrieve score files that fit description
    def is_match(
        path: str, data_set: str, models: List[str], training_sizes: List[int]
    ) -> bool:
        if data_set not in path:
            return False

        if not any(
            [re.search(rf"\d+_{model}_\d", path) is not None for model in models]
        ):
            return False

        if len(args.training_sizes) > 0:
            if not any(
                [f"_{training_size}_" in path for training_size in training_sizes]
            ):
                return False

        if not path.endswith("_stats.csv"):
            return False

        return True

    result_paths = os.listdir(args.result_dir)
    result_paths = list(
        filter(
            lambda path: is_match(path, args.dataset, args.models, args.training_sizes),
            result_paths,
        )
    )

    # Load data
    all_data = None
    metric_columns = set()

    for path in result_paths:
        _, training_size, model_name, run = (
            re.compile(r"(.+?)_(\d+)_(.+)_(\d)_\d{2}-\d{2}-\d{4}").match(path).groups()
        )
        training_size = int(training_size)

        data = pd.read_csv(f"{args.result_dir}/{path}", delimiter="\t")
        data = data.set_index("batch_num")

        for col in data.columns:
            metric_columns.add(f"{model_name}_{training_size}_{col}")

        # Rename columns with uncertainty metric values to make joining tables easier
        data = data.rename(
            columns=lambda col: f"{run}_{model_name}_{training_size}_{col}"
        )

        # Join data
        if all_data is None:
            all_data = data

        else:
            # Remove these columns before joining to avoid duplication
            try:
                all_data = all_data.join(data)

            except ValueError:
                print(f"Duplicate data found for {model_name} {training_size}")

    # Aggregate measurements across runs
    for metric_column in metric_columns:
        run_columns = [
            column
            for column in all_data.columns
            if re.match(rf"\d_{metric_column}", column)
        ]
        all_data = all_data.assign(
            **{
                f"{metric_column}_mean": all_data[run_columns].mean(axis=1),
                f"{metric_column}_std": all_data[run_columns].std(axis=1),
            }
        ).drop(run_columns, 1)

    metric_model_infix = ""

    if args.metrics is not None:
        metric_model_infix += f"_{'_'.join(args.metrics)}_"

    if args.models is not None:
        metric_model_infix += f"_{'_'.join(args.models)}_"

    # Plot data
    plot_timeseries(
        all_data,
        dataset=args.dataset,
        target=args.target,
        target_name=args.target_name,
        step_cutoff=args.step_cutoff,
        colors=MODEL_COLORS,
        metrics=args.metrics,
        save_path=f"{args.output_dir}/{args.dataset}_{args.identifier}_{args.target}.pdf",
    )
