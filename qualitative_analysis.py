"""
Perform qualitative analysis, where uncertainty estimates of different models are plotted over the course of a sequence.
"""

# STD
import argparse
from typing import List, Optional, Dict

# EXT
import matplotlib.pyplot as plt
import numpy as np
from nlp_uncertainty_zoo.config import AVAILABLE_MODELS

# PROJECT
from src.config import AVAILABLE_DATASETS

# CONST
SEED = 123456
RESULT_DIR = "./results"
MODEL_DIR = "./models"
IMG_DIR = "./img"


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
        required=True,
        nargs="+",
        choices=AVAILABLE_MODELS.keys(),
    )
    parser.add_argument("--training-size", type=int, default=None)
    parser.add_argument("--result-dir", type=str, default=RESULT_DIR)
    args = parser.parse_args()

    # TODO: Find all the relevant results
    # TODO: Aggregate results per model and metric
    # TODO: Find most interesting sentences
    # 1. Most uncertainty sentences
    # 2. Sentences with biggest disagreements between models / metrics
    # 3. Somehow biggest differences between mtrics

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
