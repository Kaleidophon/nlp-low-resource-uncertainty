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
import numpy as np
from nlp_uncertainty_zoo.config import AVAILABLE_MODELS
import pandas as pd

# PROJECT
from src.config import AVAILABLE_DATASETS

# CONST
SEED = 123456
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


def plot_timeseries(
    data: pd.DataFrame,
    target: str,
    target_name: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None,
):
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.grid(visible=True, axis="both", which="major", linestyle=":", color="grey")

    target_columns = set(
        col.replace("_mean", "").replace("_std", "")
        for col in data.columns
        if target in col
    )

    for target_column in target_columns:
        model_name, training_size = (
            re.compile(r"(.+)_(\d+)_.+").match(target_column).groups()
        )

        x = data[f"{target_column}_mean"].index - 1
        data_means, data_stds = (
            data[f"{target_column}_mean"].values,
            data[f"{target_column}_std"].values,
        )

        # Plot line
        ax.plot(
            data_means,
            label=MODEL_NAMES[model_name],
            color=colors[model_name][0],
            alpha=0.8,
            markersize=12,
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
    plt.xlabel("Batch number")

    if target_name is None:
        plt.ylabel(target)
    else:
        plt.ylabel(target_name)

    plt.xticks(np.arange(0, len(x)), x, fontsize=12)
    # ax.xaxis.set_ticklabels([])
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.legend(loc="upper right")

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
    parser.add_argument("--training-sizes", type=int, nargs="+", default=tuple())
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

        if not any([f"_{model}_" in path for model in models]):
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
            all_data = all_data.join(data)

    # Aggregate measurements across runs
    for metric_column in metric_columns:
        run_columns = [
            column
            for column in all_data.columns
            if re.match(fr"\d_{metric_column}", column)
        ]
        all_data = all_data.assign(
            **{
                f"{metric_column}_mean": all_data[run_columns].mean(axis=1),
                f"{metric_column}_std": all_data[run_columns].std(axis=1),
            }
        ).drop(run_columns, 1)

    # Plot data
    plot_timeseries(
        all_data,
        target=args.target,
        target_name=args.target_name,
        colors=MODEL_COLORS,
        save_path=f"{args.output_dir}/{args.dataset}_{args.target}.pdf",
    )
