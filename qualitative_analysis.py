"""
Perform qualitative analysis, where uncertainty estimates of different models are plotted over the course of a sequence.
"""

# STD
import argparse
from collections import defaultdict
import itertools
import os
import re
from typing import List, Optional, Dict

# EXT
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
from nlp_uncertainty_zoo.config import AVAILABLE_MODELS
import pandas as pd

# PROJECT
from src.config import AVAILABLE_DATASETS

# CONST
SEED = 123456
RESULT_DIR = "./results"
MODEL_DIR = "./models"
IMG_DIR = "./img/qualitative"
TOP_N = 10
ALPHA = 0.6
FONT_SIZE = 22
DATA_DIR = "data/processed"

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
    fig = plt.figure(figsize=(15, 8))
    ax = plt.gca()
    ax.grid(visible=True, axis="both", which="major", linestyle=":", color="grey")

    x = np.arange(0, list(uncertainties.values())[0].shape[1])
    models, metrics = set(), set()

    if sentence is not None:
        try:
            sentence_end = sentence.index("[SEP]")
        except ValueError:
            sentence_end = len(sentence)

        x = x[: sentence_end - 1]
        sentence = sentence[1:sentence_end]
        labels = labels[1:sentence_end]

    for name, data in uncertainties.items():

        model, metric = re.compile(r"(.+?) \d+ - (.+)").match(name).groups()
        models.add(model)
        metrics.add(metric)

        if sentence is not None:
            data = data[:, 1:sentence_end]

        if normalize:
            flattened_data = data.flatten()
            data = (data - np.mean(flattened_data)) / np.std(flattened_data)

        # Cut data - mostly when transformers pad sentence too long
        if data.shape[1] > x.shape[0]:
            data = data[:, x.shape[0]]

        # Plot line
        ax.plot(
            data.mean(axis=0),
            label=name,
            marker=markers[metric],
            markersize=18,
            color=colors[model][0],
            alpha=0.8,
        )

        # Plot +/- standard deviation
        ax.fill_between(
            x=x,
            y1=data.mean(axis=0) + data.std(axis=0),
            y2=data.mean(axis=0) - data.std(axis=0),
            color=colors[model][1],
            alpha=0.2,
        )

    ax.set_ylabel(
        f"{'Normalized ' if normalize else 'Unnormalized '}Uncertainty",
        alpha=0.6,
        fontsize=FONT_SIZE,
    )

    # Set xticks
    if labels is not None:
        xticks = [f"{token} ({label})" for token, label in zip(sentence, labels)]

    else:
        xticks = sentence

    plt.xticks(x, xticks, fontsize=FONT_SIZE)
    # ax.yaxis.set_ticklabels([])
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        alpha=0.6,
        fontsize=FONT_SIZE,
    )

    legend_elements = [
        # Add metrics
        *[
            Line2D(
                [0],
                [0],
                markersize=18,
                alpha=ALPHA,
                markerfacecolor="black",
                color="w",
                label=METRIC_NAMES[metric],
                marker=markers[metric],
            )
            for metric in metrics
        ],
        # Add models
        *[
            Patch(
                facecolor=colors[model_name][1],
                edgecolor=colors[model_name][0],
                label=MODEL_NAMES[model_name],
                alpha=ALPHA,
            )
            for model_name in models
        ],
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        ncol=2,
        fontsize=FONT_SIZE - 2,
        # bbox_to_anchor=(1.02, 1),
        handlelength=0.75,
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

    plt.close()


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
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        nargs="+",
    )
    parser.add_argument("--training-sizes", type=int, nargs="+", default=tuple())
    parser.add_argument("--result-dir", type=str, default=RESULT_DIR)
    parser.add_argument("--output-dir", type=str, default=IMG_DIR)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument(
        "--top-n", type=int, default=TOP_N, help="Top n samples to plot per analysis."
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
    models2sizes = defaultdict(list)
    models2metrics = defaultdict(list)

    # TODO: This could be written more simply by avoiding pandas altogether
    for path in result_paths:
        _, training_size, model_name, run = (
            re.compile(r"(.+?)_(\d+)_(.+)_(\d)_\d{2}-\d{2}-\d{4}").match(path).groups()
        )
        training_size = int(training_size)

        data = pd.read_csv(f"{args.result_dir}/{path}", delimiter="\t")
        data = data.drop(columns=["Unnamed: 0"])

        if args.metrics is not None:
            drop_metric_columns = (
                set(data.columns)
                - {"sentence", "labels", "predictions"}
                - set(args.metrics)
            )
            data = data.drop(columns=drop_metric_columns)

        # Gather info about available results
        models2sizes[model_name].append(training_size)
        for column in data.columns:
            if column in ["sentence", "labels", "predictions"]:
                continue

            models2metrics[model_name].append(column)

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

            try:
                all_data = all_data.join(data)

            # Duplicate data, skip
            except ValueError:
                print(
                    f"Duplicate data found for {args.dataset} {training_size}, {model_name}"
                )
                continue

    # Aggregate measurements across runs
    aggregate_data = defaultdict(dict)

    for model in models2metrics:
        for metric, training_size in itertools.product(
            models2metrics[model], models2sizes[model]
        ):

            # Find all columns corresponding to some model, training size and metric
            run_columns = [
                column
                for column in all_data.columns
                if re.match(fr"{model}_{training_size}_\d_{metric}", column)
            ]

            # Compute mean and std over all time steps over model runs
            run_data = all_data[run_columns].applymap(
                lambda cell: np.array(list(map(float, cell.split())))
            )

            for i, row in run_data.iterrows():
                combined_data = np.stack(row.values)
                aggregate_data[i][f"{model} {training_size} - {metric}"] = combined_data

    # Sample indices
    sampled_indices = np.random.choice(range(0, len(all_data)), args.top_n)
    dataset_builder = AVAILABLE_DATASETS[args.dataset](
        data_dir=DATA_DIR,
        max_length=35,
    )
    data_splits = dataset_builder.build(batch_size=32, drop_last=True)
    tokenizer = dataset_builder.tokenizer
    label_encoder = dataset_builder.label_encoder
    del dataset_builder, data_splits

    if not os.path.exists(f"{IMG_DIR}/{args.dataset}"):
        os.makedirs(f"{IMG_DIR}/{args.dataset}")

    for i, idx in enumerate(sampled_indices):
        uncertainties = aggregate_data[idx]
        labels = list(map(int, all_data.loc[idx, "labels"].split()))
        labels = [
            label_encoder.inverse_transform([label])[0] if label != -100 else -100
            for label in labels
        ]
        sentence = all_data.loc[idx, "sentence"]
        sentence = tokenizer.tokenize(sentence)

        # Skip short sentences
        if sentence.index("[SEP]") <= 10:
            continue

        try:
            plot_uncertainties_over_sequence(
                uncertainties=uncertainties,
                sentence=sentence,
                labels=labels,
                colors=MODEL_COLORS,
                markers=METRIC_MARKERS,
                normalize=args.normalize,
                save_path=f"{IMG_DIR}/{args.dataset}/{i}.pdf",
            )
        except ValueError:
            continue
