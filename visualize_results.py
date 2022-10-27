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
import matplotlib.patches as mpatches
import numpy as np
from nlp_uncertainty_zoo.config import AVAILABLE_MODELS

# PROJECT
from src.config import AVAILABLE_DATASETS


# CONST
RESULT_DIR = "./results"
MODEL_DIR = "./models"
IMG_DIR = "./img/scatter_plots"


# Plotting defaults
ALPHA = 0.6
MARKER_SCALE = 240
FONT_SIZE = 18
MODEL_COLORS = {
    "lstm": ("firebrick", "lightcoral"),
    "lstm_ensemble": ("forestgreen", "yellowgreen"),
    "st_tau_lstm": ("midnightblue", "skyblue"),
    "bayesian_lstm": ("orangered", "lightsalmon"),
    "variational_lstm": ("darkmagenta", "orchid"),
    "ddu_bert": ("darkcyan", "mediumturquoise"),
    "variational_bert": ("mediumvioletred", "pink"),
    "sngp_bert": ("black", "silver"),
}
METRIC_MARKERS = {
    "max_prob": "o",
    "predictive_entropy": "^",
    "variance": "s",
    "softmax_gap": "p",
    "dempster_shafer": "P",
    "mutual_information": "X",
    "log_prob": "D",
    "ece": "2",
    "ace": "3",
    "sce": "4",
}
METRIC_NAMES = {
    "max_prob": "Max. Prob.",
    "predictive_entropy": "Pred. Entropy",
    "variance": "Variance",
    "softmax_gap": "Softmax gap",
    "dempster_shafer": "Dempster-Shafer",
    "mutual_information": "Mutual Inf.",
    "log_prob": "Log. Prob.",
    "ece": "ECE",
    "sce": "SCE",
    "ace": "ACE",
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
TRAINING_SIZE_SCALES = {
    "dan+": [1000, 2000, 4000],
    "finnish_ud": [5000, 7500, 10000],
    "clinc_plus": [10000, 12500, 15000],
}
plt.style.use("science")


def plot_results(
    x_axis: str,
    x_label: str,
    y_label: str,
    metric_prefix: str,
    metric_postfix: str,
    metrics: List[str],
    data,
    plot_ood=False,
    model_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    model_colors: Dict[str, Tuple[str, str]] = MODEL_COLORS,
    metric_markers: Dict[str, str] = METRIC_MARKERS,
    size_scales: Optional[Dict[int, float]] = None,
):
    num_sizes = len(size_scales)
    fig, ax = plt.subplots(figsize=(15, 5), ncols=num_sizes, sharey="row", sharex="row")
    fig.supxlabel(x_label, alpha=0.6, y=0.05, x=0.45, fontsize=FONT_SIZE)

    for a in ax:
        a.grid(visible=True, axis="both", which="major", linestyle=":", color="grey")

    for model in data.keys():
        for a, training_size in zip(ax, sorted(data[model])):

            a.set_title(
                f"{training_size} instances",
                loc="center",
                fontdict={"alpha": 0.6, "fontsize": FONT_SIZE},
            )

            for metric in metrics:
                edge_color, color = model_colors[model]
                target_x = data[model][training_size][x_axis]

                if not plot_ood:
                    target_y = data[model][training_size][
                        metric_prefix + metric + metric_postfix
                    ]

                else:
                    target_y = data[model][training_size][
                        "id_" + metric_prefix + metric + metric_postfix
                    ]

                if len(target_x) != len(target_y):
                    continue

                jitter_x = np.random.normal(0, 0.01, target_x.shape)
                jitter_y = np.random.normal(0, 0.01, target_y.shape)

                target_x = np.mean(target_x + jitter_x)
                target_y = np.mean(target_y + jitter_y)

                a.scatter(
                    target_x,
                    target_y,
                    marker=metric_markers[metric],
                    color=color,
                    edgecolor=edge_color,
                    alpha=ALPHA,
                    s=MARKER_SCALE,  # size_scales[training_size],
                )

                if plot_ood:

                    ood_target_x = data[model][training_size]["ood_" + x_axis]
                    ood_target_x = np.mean(ood_target_x + jitter_x)
                    ood_target_y = data[model][training_size][
                        "ood_" + metric_prefix + metric + metric_postfix
                    ]
                    ood_target_y = np.mean(ood_target_y + jitter_y)

                    a.scatter(
                        ood_target_x,
                        ood_target_y,
                        marker=metric_markers[metric],
                        color=color,
                        edgecolor=edge_color,
                        alpha=ALPHA,
                        s=MARKER_SCALE,  # size_scales[training_size],
                    )

                    dx, dy = ood_target_x - target_x, ood_target_y - target_y
                    norm = np.linalg.norm([dx, dy])
                    dx /= norm
                    dy /= norm
                    offset_factor = 0.0025

                    # Draw arrow
                    arrow = mpatches.FancyArrowPatch(
                        (target_x + offset_factor * dx, target_y + offset_factor * dy),
                        (
                            ood_target_x - offset_factor * dx,
                            ood_target_y - offset_factor * dy,
                        ),
                        mutation_scale=2,
                        arrowstyle=mpatches.ArrowStyle(
                            "simple", head_width=4, head_length=2
                        ),
                        alpha=ALPHA + 0.1,
                        color=edge_color,
                        # color="black",
                        fill=True,
                    )
                    try:
                        a.add_patch(arrow)
                    except StopIteration:
                        continue

    ax[0].set_ylabel(y_label, alpha=0.6, fontsize=FONT_SIZE, labelpad=-2)

    if model_names is None:
        model_names = list(MODEL_NAMES.keys())

    # Create legend
    legend_elements = [
        # Add metrics
        *[
            Line2D(
                [0],
                [0],
                markersize=16,
                alpha=ALPHA + 0.2,
                markerfacecolor="gray",
                color="w",
                label=METRIC_NAMES[metric],
                marker=metric_markers[metric],
            )
            for metric in metrics
        ],
        # Add models
        *[
            Patch(
                facecolor=model_colors[model_name][1],
                edgecolor=model_colors[model_name][0],
                label=MODEL_NAMES[model_name],
                alpha=ALPHA,
            )
            for model_name in model_names
        ],
    ]
    # Add training sizes
    ax[-1].legend(
        bbox_to_anchor=(0.99, 1.05),
        handles=legend_elements,
        loc="upper left",
        ncol=1,
        fontsize=FONT_SIZE - 4,
        handlelength=0.75,
    )

    for a in ax:
        a.tick_params(axis="both", which="major", labelsize=FONT_SIZE - 4)

    fig.tight_layout(w_pad=0.95)

    plt.savefig(save_path, format="pdf", dpi=120, bbox_inches="tight")


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
        nargs="+",
        choices=AVAILABLE_MODELS.keys(),
        default=AVAILABLE_MODELS.keys(),
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

    max_run_per_model = defaultdict(int)

    for result_path in result_paths:
        _, training_size, model_name, run = (
            re.compile(r"(.+?)_(\d+)_(.+)_(\d)_\d{2}-\d{2}-\d{4}")
            .match(result_path)
            .groups()
        )
        training_size = int(training_size)

        run = int(run)

        if run < max_run_per_model[model_name]:
            continue

        with open(f"{args.result_dir}/{result_path}", "rb") as result_file:
            scores = pickle.load(result_file)

            for name, score in scores.items():

                if name in ("train_loss", "_timestamp", "_runtime"):
                    continue

                if name.startswith("auroc"):
                    found_metrics.add(name.replace("auroc_", ""))

                data[model_name][training_size][name] = np.array(score)

    model_names = None

    if args.dataset == "clinc_plus":
        model_names = [
            "lstm",
            "lstm_ensemble",
            "bayesian_lstm",
            "ddu_bert",
            "variational_bert",
        ]

    # Create plots
    plot_results(
        x_axis="macro_f1_scores",
        metrics=list(found_metrics),
        metric_prefix="auroc_",
        metric_postfix="",
        x_label="Macro F1 score",
        y_label="ID / OOD AUROC",
        data=data,
        model_names=model_names,
        save_path=f"{IMG_DIR}/{args.dataset}_scatter_auroc.pdf",
        size_scales=TRAINING_SIZE_SCALES[args.dataset],
    )

    plot_results(
        x_axis="macro_f1_scores",
        metrics=list(found_metrics),
        metric_prefix="aupr_",
        metric_postfix="",
        x_label="Macro F1 score",
        y_label="ID / OOD AUPR",
        data=data,
        model_names=model_names,
        save_path=f"{IMG_DIR}/{args.dataset}_scatter_aupr.pdf",
        size_scales=TRAINING_SIZE_SCALES[args.dataset],
    )

    # Create ID / OOD plots
    if args.dataset != "clinc_plus":
        plot_results(
            x_axis="macro_f1_scores",
            metric_prefix="",
            metric_postfix="_kendalls_tau_token",
            metrics=found_metrics,
            x_label="Macro F1 score",
            y_label="Token-level Kendall's tau",
            data=data,
            model_names=model_names,
            plot_ood=True,
            save_path=f"{IMG_DIR}/{args.dataset}_scatter_kendalls_tau_token.pdf",
            size_scales=TRAINING_SIZE_SCALES[args.dataset],
        )

    plot_results(
        x_axis="macro_f1_scores",
        metric_prefix="" if args.dataset != "clinc_plus" else "id_",
        metric_postfix="_kendalls_tau_seq",
        metrics=list(found_metrics),
        x_label="Macro F1 score",
        y_label="Sequence-level Kendall's tau",
        data=data,
        model_names=model_names,
        plot_ood=args.dataset != "clinc_plus",
        save_path=f"{IMG_DIR}/{args.dataset}_scatter_kendalls_tau_seq.pdf",
        size_scales=TRAINING_SIZE_SCALES[args.dataset],
    )

    if args.dataset != "clinc_plus":
        plot_results(
            x_axis="macro_f1_scores",
            metric_prefix="",
            metric_postfix="_kendalls_tau_max_seq",
            metrics=list(found_metrics),
            x_label="Macro F1 score",
            y_label="Sequence-level Kendall's tau (max)",
            data=data,
            model_names=model_names,
            plot_ood=True,
            save_path=f"{IMG_DIR}/{args.dataset}_scatter_kendalls_tau_seq_max.pdf",
            size_scales=TRAINING_SIZE_SCALES[args.dataset],
        )
