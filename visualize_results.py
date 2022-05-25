"""
Produce scatter plots based on experimental results.
"""

# STD
import argparse
from collections import defaultdict
import os
import re
from typing import List, Dict, Optional, Tuple
import pickle

# EXT
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from nlp_uncertainty_zoo.config import AVAILABLE_MODELS

# PROJECT
from src.config import AVAILABLE_DATASETS


# CONST
RESULT_DIR = "./results"
MODEL_DIR = "./models"
IMG_DIR = "./img/scatter_plots"


# Plotting defaults
ALPHA = 0.4
MODEL_COLORS = {
    "lstm": ("firebrick", "lightcoral"),
    "lstm_ensemble": ("forestgreen", "yellowgreen"),
    "st_tau_lstm": ("midnightblue", "skyblue"),
    "bayesian_lstm": ("orangered", "lightsalmon"),
    "variational_lstm": ...,
    "ddu_bert": ...,
    "variational_bert": ...,
    "sngp_bert": ...,
}
METRIC_MARKERS = {
    "max_prob": "o",
    "predictive_entropy": "^",
    "variance": "s",
    "softmax_gap": "p",
    "dempster_shafer": "P",
    "mutual_information": "X",
    "log_prob": "D",
}
METRIC_NAMES = {
    "max_prob": "Max. Prob.",
    "predictive_entropy": "Pred. Entropy",
    "variance": "Variance",
    "softmax_gap": "Softmax gap",
    "dempster_shafer": "Dempster-Shafer",
    "mutual_information": "Mutual Inf.",
    "log_prob": "Log. Prob.",
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
TRAINING_SIZE_SCALES = {"dan+": {1000: 8, 2000: 40, 4000: 80}}
plt.style.use("science")


def plot_results(
    x_axis: str,
    x_label: str,
    y_label: str,
    metric_prefix: str,
    metrics: List[str],
    data,
    save_path: Optional[str] = None,
    model_colors: Dict[str, Tuple[str, str]] = MODEL_COLORS,
    metric_markers: Dict[str, str] = METRIC_MARKERS,
    size_scales: Optional[Dict[int, float]] = None,
):
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.grid(visible=True, axis="both", which="major", linestyle=":", color="grey")

    for model in data.keys():
        for training_size in data[model]:
            for metric in metrics:
                edge_color, color = model_colors[model]
                target_x = data[model][training_size][x_axis]
                target_y = data[model][training_size][metric_prefix + metric]

                if len(target_x) != len(target_y):
                    continue

                plt.scatter(
                    target_x,
                    target_y,
                    marker=metric_markers[metric],
                    color=color,
                    edgecolor=edge_color,
                    alpha=ALPHA,
                    s=size_scales[training_size],
                )

    # ax.set_xlim([0, 1])
    # ax.set_ylim([0, 1])
    ax.set_xlabel(x_label, alpha=0.6)
    ax.set_ylabel(y_label, alpha=0.6)
    # ax.legend(labels=data_labels, handles=handles, loc="upper left")
    # ax.axvline(x=2.5, c="black")

    # Create legend
    legend_elements = [
        # Add metrics
        *[
            Line2D(
                [0],
                [0],
                markersize=10,
                alpha=ALPHA,
                markerfacecolor="black",
                color="w",
                label=METRIC_NAMES[metric],
                marker=metric_markers[metric],
            )
            for metric in metrics
        ],
        *[
            Patch(
                facecolor=model_colors[model_name][1],
                edgecolor=model_colors[model_name][0],
                label=MODEL_NAMES[model_name],
                alpha=ALPHA,
            )
            for model_name in data.keys()
        ],
    ]
    ax.legend(handles=legend_elements, loc="upper right", ncol=2, fontsize=8)

    fig.tight_layout()

    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")


def plot_results_id_ood(
    x_axis: str,
    metrics: List[str],
    data,
    save_path: Optional[str] = None,
    model_colors: Dict[str, Tuple[str, str]] = MODEL_COLORS,
    metric_markers: Dict[str, str] = METRIC_MARKERS,
    size_scales: Optional[Dict[int, float]] = None,
):
    pass


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
        "--models",
        type=str,
        required=True,
        nargs="+",
        choices=AVAILABLE_MODELS.keys(),
    )
    parser.add_argument("--result-dir", type=str, default=RESULT_DIR)
    args = parser.parse_args()

    # Retrieve score files that fit description
    def is_match(path: str, data_set: str, models: List[str]) -> bool:
        if data_set not in path:
            return False

        if not any([f"_{model}_" in path for model in models]):
            return False

        if "_scores.pkl" not in path:
            return False

        return True

    result_paths = os.listdir(args.result_dir)
    result_paths = list(
        filter(lambda path: is_match(path, args.dataset, args.models), result_paths)
    )

    # Read in data
    # Map from model to training size to ID / OOD to metric
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    found_metrics = set()

    for result_path in result_paths:
        _, training_size, model_name = (
            re.compile(r"(.+?)_(\d+)_(.+)_\d{2}-\d{2}-\d{4}")
            .match(result_path)
            .groups()
        )
        training_size = int(training_size)

        with open(f"{args.result_dir}/{result_path}", "rb") as result_file:
            scores = pickle.load(result_file)

            for name, score in scores.items():

                if name.startswith("auroc"):
                    found_metrics.add(name.replace("auroc_", ""))

                data[model_name][training_size][name] += score

    # Create plots
    plot_results(
        x_axis="macro_f1_scores",
        metrics=found_metrics,
        metric_prefix="auroc_",
        x_label="Macro F1 score",
        y_label="ID / OOD AUROC",
        data=data,
        save_path=f"{IMG_DIR}/scatter_auroc.pdf",
        size_scales=TRAINING_SIZE_SCALES[args.dataset],
    )

    import sys

    sys.exit(0)

    plot_results(
        x_axis="macro_f1",
        metrics=[f"aupr_{metric_name}" for metric_name in found_metrics],
        data=data,
        save_path=f"{IMG_DIR}/scatter_aupr.pdf",
    )

    # Create ID / OOD plots
    # TODO: Check for the kind of dataset
    plot_results(
        x_axis="macro_f1_scores",
        metrics=[f"{metric_name}_kendalls_tau_token" for metric_name in found_metrics],
        data=data,
        save_path=f"{IMG_DIR}/scatter_kendalls_tau_token.pdf",
    )
    plot_results(
        x_axis="macro_f1",
        metrics=[f"{metric_name}_kendalls_tau_seq" for metric_name in found_metrics],
        data=data,
        save_path=f"{IMG_DIR}/scatter_kendalls_tau_seq.pdf",
    )
    plot_results(
        x_axis="macro_f1",
        metrics=[
            f"{metric_name}_kendalls_tau_seq_max" for metric_name in found_metrics
        ],
        data=data,
        save_path=f"{IMG_DIR}/scatter_kendalls_tau_seq_max.pdf",
    )
