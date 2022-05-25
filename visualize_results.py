"""
Produce scatter plots based on experimental results.
"""

# STD
import argparse
from collections import defaultdict
import os
from typing import List, Dict, Optional, Tuple
import pickle

# EXT
from matplotlib import pyplot as plt
from nlp_uncertainty_zoo.config import AVAILABLE_MODELS

# PROJECT
from src.config import AVAILABLE_DATASETS


# CONST
RESULT_DIR = "./results"
MODEL_DIR = "./models"
IMG_DIR = "./img"


# Plotting defaults
ALPHA = 0.6
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
TRAINING_SIZE_SCALES = {"dan+": {1000: 1, 2000: 1.2, 4000: 1.4}}
PLOT_STYLE = {
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.spines.right": True,
    "axes.spines.top": True,
    "axes.edgecolor": ".15",
    "grid.color": ".8",
    "ytick.left": True,
    "text.usetex": True,
}
plt.style.use("science")


def plot_results(
    x_axis: str,
    y_axis: str,
    data,
    ood_data: Optional[Dict] = None,
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
    data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict[list]))
    )

    for result_path in result_paths:
        with open(f"{args.result_dir}/{result_path}", "rb") as result_file:
            scores = pickle.load(result_file)
            ...

    # Create scatter plot: Kendall's tau as

    # TODO: Read in relevant data per dataset / model
    # TODO: Create beautiful plots
