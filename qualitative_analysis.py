"""
Perform qualitative analysis, where uncertainty estimates of different models are plotted over the course of a sequence.
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
IMG_DIR = "./img"
TOP_N = 10

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


def plot_uncertainties_over_sequence(
    uncertainties: Dict[str, np.array],
    sentence: List[str],
    labels: Optional[List[str]],
    colors: Optional[Dict[str, str]] = None,
    markers: Optional[Dict[str, str]] = None,
    normalize: bool = True,
    save_path: Optional[str] = None,
):
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.grid(visible=True, axis="both", which="major", linestyle=":", color="grey")

    x = np.arange(0, list(uncertainties.values())[0].shape[1])

    for name, data in uncertainties.items():

        if normalize:
            flattened_data = data.flatten()
            data = (data - np.mean(flattened_data)) / np.std(flattened_data)

        # Plot line
        ax.plot(
            data.mean(axis=0),
            label=name,
            marker=markers[name],
            color=colors[name],
            alpha=0.8,
            markersize=12,
        )

        # Plot +/- standard deviation
        ax.fill_between(
            x=x,
            y1=data.mean(axis=0) + data.std(axis=0),
            y2=data.mean(axis=0) - data.std(axis=0),
            color=colors[name],
            alpha=0.2,
        )

    ax.set_ylabel("Uncertainty", alpha=0.6)
    ax.legend(loc="upper right")

    # Set xticks
    if labels is not None:
        xticks = [f"{token} ({label})" for token, label in zip(sentence, labels)]

    else:
        xticks = sentence

    plt.xticks(x, xticks, fontsize=12)
    ax.yaxis.set_ticklabels([])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

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
        "--models",
        type=str,
        default=AVAILABLE_MODELS.keys(),
        nargs="+",
        choices=AVAILABLE_MODELS.keys(),
    )
    parser.add_argument("--training-sizes", type=int, nargs="+", default=tuple())
    parser.add_argument("--result-dir", type=str, default=RESULT_DIR)
    parser.add_argument("--output-dir", type=str, default=IMG_DIR)
    parser.add_argument(
        "--top-n", type=str, default=TOP_N, help="Top n samples to plot per analysis."
    )
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

        if not any([f"_{training_size}_" in path for training_size in training_sizes]):
            return False

        if not path.endswith("_uncertainty.csv"):
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

    for path in result_paths:
        _, training_size, model_name, run = (
            re.compile(r"(.+?)_(\d+)_(.+)_(\d)_\d{2}-\d{2}-\d{4}").match(path).groups()
        )
        training_size = int(training_size)

        data = pd.read_csv(f"{args.result_dir}/{path}", delimiter="\t")
        data = data.drop(columns=["Unnamed: 0"])

        # Rename columns with uncertainty metric values to make joining tables easier
        data = data.rename(
            columns=lambda col: f"{model_name}_{training_size}_{run}_{col}"
            if col not in ["sentence", "labels"]
            else col
        )

        # Join data
        if all_data is None:
            all_data = data

        else:
            # Remove these columns before joining to avoid duplication
            data = data.drop(columns=["sentence", "labels"])
            all_data = all_data.join(data)

    # Create additional statistics
    for index, row in all_data.iterrows():
        ...

    # 1. Most uncertain sentences
    # TODO: Max mean across metrics

    # 1. Least uncertain sentences
    # TODO: Min mean across metrics

    # 2. Sentences with biggest disagreements between models / metrics
    # TODO: Max variance across metrics

    # 3. Somehow biggest differences between metrics

    # TODO: Plot some of those below
    # TODO: Remove sooner or later
    dummy_colors = {
        "model 1 - metric A": "firebrick",
        "model 1 - metric B": "firebrick",
        "model 2 - metric A": "forestgreen",
    }
    dummy_markers = {
        "model 1 - metric A": "o",
        "model 1 - metric B": "^",
        "model 2 - metric A": "o",
    }
    dummy_sentence = "De slår løs på kvinder , ' sagde Eduardas Potashinskas .".split(
        " "
    )
    dummy_labels = "O O O O O O O O B-PER I-PER O".split(" ")

    dummy_uncertainties = {
        "model 1 - metric A": np.cos(np.arange(0, 11, 1))
        * np.random.normal(0, 0.4, size=(5, 11)),
        "model 1 - metric B": np.sin(np.arange(0, 11, 1))
        * np.random.normal(0.3, 0.2, size=(5, 11)),
        "model 2 - metric A": np.cos(np.arange(0, 11, 1))
        * np.random.normal(-0.1, 0.8, size=(5, 11)),
    }

    plot_uncertainties_over_sequence(
        uncertainties=dummy_uncertainties,
        sentence=dummy_sentence,
        labels=dummy_labels,
        colors=dummy_colors,
        markers=dummy_markers,
        normalize=False,
    )
