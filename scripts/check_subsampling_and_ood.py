"""
Create sub-sampled version of training sets and check corpus statistics to ensure that sub-sampling is sensible.
"""

# STD
from collections import Counter
from typing import Tuple, List, Optional

# EXT
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nlp_uncertainty_zoo.utils.samplers import (
    TokenClassificationSampler,
    SequenceClassificationSampler,
)
from torch.utils.data import DataLoader

# PROJECT
from src.data import DanPlusBuilder, FinnishUDBuilder, ClincPlusBuilder

# CONST
SAMPLING_PARAMS_TOKEN_PRED = {"num_jobs": 2, "seed": 1234}
SAMPLING_PARAMS_LANGUAGE_MODELLING = {
    "num_jobs": 2,
    "seed": 1234,
    "sample_range": [0, 3],
}
TARGET_SIZES = [100, 250, 500, 1000]
DATA_DIR = "data/processed"
MAX_LENGTH = 50
BATCH_SIZE = 1
IGNORE_TOKENS = [-100, 0, 1, 2, 3, 4]
IMG_PATH = "img"
TOP_N = 25
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
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def collect_sentence_length_and_class_dict(
    dataloader: DataLoader, ignore_tokens: List[int]
) -> Tuple[Counter, Counter, Counter]:
    """
    Collect the frequencies of sentence lengths and classes.
    """
    seq_freqs = Counter()
    token_freqs = Counter()
    class_freqs = Counter()

    for batch in dataloader:
        input_ids = [
            idx for idx in batch["input_ids"].tolist()[0] if idx not in ignore_tokens
        ]
        # For sequence classification
        if batch["labels"].shape == (1,):
            labels = batch["labels"].tolist()

        else:
            labels = [
                idx for idx in batch["labels"].tolist()[0] if idx not in ignore_tokens
            ]
        seq_freqs.update([len(input_ids)])
        class_freqs.update(labels)
        token_freqs.update(input_ids)

    return seq_freqs, token_freqs, class_freqs


def compute_coverage(
    token_freqs_orig: Counter, token_freqs_sampled: Counter
) -> Tuple[float, float]:
    """
    Computer the average coverage of the vocabulary (percentage of types and tokens) of the sub-sampled corpus compared
    to the original corpus.
    """
    type_percentage = len(token_freqs_sampled) / len(token_freqs_orig)

    token_percentage = 0

    for token in token_freqs_sampled:
        token_percentage += token_freqs_orig[token]

    token_percentage /= sum(token_freqs_orig.values())

    return type_percentage, token_percentage


def plot_coverage(
    sizes: List[int],
    type_coverages: List[float],
    token_coverage: List[float],
    colors: List[str] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Plot the type and token coverage as a function of sub-sampled corpus size.
    """
    data = pd.DataFrame.from_dict(
        {
            "size": sizes * 2,
            "coverage": type_coverages + token_coverage,
            "kind": ["type"] * len(sizes) + ["token"] * len(sizes),
        }
    )

    with sns.axes_style("whitegrid"):
        sns.set(
            font_scale=3,
            font="Computer Modern",
            rc={"xtick.bottom": True, **PLOT_STYLE},
        )
        sns.set_palette("viridis")

        plot = sns.barplot(
            data=data,
            x="size",
            y="coverage",
            hue="kind",
            alpha=0.8,
            ci=None,
            palette=colors,
            edgecolor="black",
        )
        plot.legend().set_title(None)

        if title:
            plot.set_title(title)

        plt.tight_layout()

        if save_path is None:
            plt.show()

        else:
            plt.savefig(save_path)

        plt.close()


def plot_dists(
    freqs_orig: Counter,
    freqs_sampled: Counter,
    x_axis: str = "type",
    colors: List[str] = None,
    top_n: Optional[int] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    compare_label: str = "subsampled",
):
    """
    Plot distributions of the top(-n) tokens and labels in the same histogram.
    """
    if top_n is not None:
        freqs_orig = dict(freqs_orig.most_common(top_n))
        freqs_sampled = {type_: freqs_sampled[type_] for type_ in freqs_orig}

    # Sort keys by frequency, decendingly
    sorted_keys = list(zip(*sorted(freqs_orig.items(), key=lambda t: t[1])))[0]

    num_types = len(freqs_orig)
    total_orig = sum(freqs_orig.values())
    total_sampled = sum(freqs_sampled.values())
    freqs = np.zeros(2 * num_types)

    for i, key in enumerate(sorted_keys):
        freqs[i] = freqs_orig.get(key, 0) / total_orig
        freqs[i + num_types] = freqs_sampled.get(key, 0) / total_sampled

    data = pd.DataFrame.from_dict(
        {
            x_axis: list(freqs_orig.keys()) * 2,
            "relative frequencies": freqs,
            "corpus": ["original"] * num_types + [compare_label] * num_types,
        }
    )

    with sns.axes_style("whitegrid"):
        sns.set(
            rc={"figure.figsize": (15, 10), **PLOT_STYLE},
            font_scale=3,
            font="Computer Modern",
        )
        plot = sns.barplot(
            data=data,
            x=x_axis,
            y="relative frequencies",
            hue="corpus",
            alpha=0.8,
            ci=None,
            palette=colors,
            edgecolor="black",
        )
        plot.set(xticklabels=[])
        plot.legend().set_title(None)

        if title:
            plot.set_title(title)

        plt.tight_layout()

        if save_path is None:
            plt.show()

        else:
            plt.savefig(save_path)

        plt.close()


def plot_length_dists(
    freqs_orig: Counter,
    freqs_sampled: Counter,
    colors: List[str] = None,
    top_n: Optional[int] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    compare_label: str = "subsampled",
):
    """
    Plot distributions of sentence lengths in the same histogram.
    """
    if top_n is None:
        top_n = max(max(freqs_orig.keys()), max(freqs_sampled.keys()))

    total_orig = sum(freqs_orig.values())
    total_sampled = sum(freqs_sampled.values())
    freqs = np.zeros(2 * top_n)

    for length in range(top_n):
        freqs[length] = freqs_orig.get(length, 0) / total_orig
        freqs[top_n + length] = freqs_sampled.get(length, 0) / total_sampled

    data = pd.DataFrame.from_dict(
        {
            "sequence_length": list(range(top_n)) + list(range(top_n)),
            "relative frequencies": freqs,
            "corpus": ["original"] * top_n + [compare_label] * top_n,
        }
    )

    with sns.axes_style("whitegrid"):
        sns.set(
            rc={"figure.figsize": (14, 10), "xtick.bottom": True, **PLOT_STYLE},
            font_scale=3,
            font="Computer Modern",
        )

        plot = sns.barplot(
            data=data,
            x="sequence_length",
            y="relative frequencies",
            hue="corpus",
            alpha=0.8,
            ci=None,
            palette=colors,
            edgecolor="black",
        )
        plot.legend().set_title(None)

        for i, label in enumerate(plot.xaxis.get_ticklabels()):
            if i % 5 != 0:
                label.set_visible(False)

        if title:
            plot.set_title(title)

        plt.tight_layout()

        if save_path is None:
            plt.show()

        else:
            plt.savefig(save_path)

        plt.close()


if __name__ == "__main__":

    for dataset_name, builder, sampler_class, sampler_kwargs in zip(
        ["danplus", "finnish_ud", "clinc_plus"],
        [DanPlusBuilder, FinnishUDBuilder, ClincPlusBuilder],
        [
            TokenClassificationSampler,
            TokenClassificationSampler,
            SequenceClassificationSampler,
        ],
        [
            SAMPLING_PARAMS_TOKEN_PRED,
            SAMPLING_PARAMS_TOKEN_PRED,
            SAMPLING_PARAMS_TOKEN_PRED,
        ],
    ):

        orig_data = builder(data_dir=DATA_DIR, num_jobs=2, max_length=MAX_LENGTH).build(
            batch_size=BATCH_SIZE
        )
        (
            orig_seq_freqs,
            orig_token_freqs,
            orig_label_freqs,
        ) = collect_sentence_length_and_class_dict(
            orig_data["train"], ignore_tokens=IGNORE_TOKENS
        )

        # Perform analyses on OOD data
        (
            ood_seq_freqs,
            ood_token_freqs,
            ood_label_freqs,
        ) = collect_sentence_length_and_class_dict(
            orig_data["ood_test"], ignore_tokens=IGNORE_TOKENS
        )
        del orig_data

        # Plot those results
        # Plot sentence length distributions
        plot_length_dists(
            orig_seq_freqs,
            ood_seq_freqs,
            colors=COLORS[1::2],
            # title="Relative sentence length frequencies",
            save_path=f"{IMG_PATH}/{dataset_name}_seq_lengths_ood.png",
            compare_label="ood",
        )

        # Plot token distributions
        plot_dists(
            orig_token_freqs,
            ood_token_freqs,
            colors=COLORS[1::2],
            top_n=TOP_N,
            # title="Relative type frequencies",
            save_path=f"{IMG_PATH}/{dataset_name}_type_freqs_ood.png",
            compare_label="ood",
        )

        # Plot task distributions
        if dataset_name != "clinc_plus":
            plot_dists(
                orig_label_freqs,
                ood_label_freqs,
                x_axis="label",
                colors=COLORS[1::2],
                top_n=TOP_N,
                # title="Relative type frequencies",
                save_path=f"{IMG_PATH}/{dataset_name}_class_freqs_ood.png",
                compare_label="ood",
            )

        # Perform the same tests for sub-samples of increasing size
        type_coverages, token_coverages = [], []

        for target_size in TARGET_SIZES:
            subsampled_data = builder(
                data_dir=DATA_DIR,
                num_jobs=2,
                max_length=MAX_LENGTH,
                sampler_class=sampler_class,
                sampler_kwargs={
                    "train": {"target_size": target_size, **sampler_kwargs}
                },
            ).build(batch_size=BATCH_SIZE)

            (
                sampled_seq_freqs,
                sampled_token_freqs,
                sampled_label_freqs,
            ) = collect_sentence_length_and_class_dict(
                subsampled_data["train"], ignore_tokens=IGNORE_TOKENS
            )
            del subsampled_data

            # Plot results
            type_cover, token_cover = compute_coverage(
                orig_token_freqs, sampled_token_freqs
            )
            type_coverages.append(type_cover)
            token_coverages.append(token_cover)

            # Plot sentence length distributions
            plot_length_dists(
                orig_seq_freqs,
                sampled_seq_freqs,
                colors=COLORS[::2],
                top_n=TOP_N,
                # title=f"Relative sentence length frequencies (n={target_size})",
                save_path=f"{IMG_PATH}/{dataset_name}_seq_lengths_{target_size}.png",
            )

            # Plot token distributions
            plot_dists(
                orig_token_freqs,
                sampled_token_freqs,
                colors=COLORS[::2],
                top_n=TOP_N,
                # title=f"Relative type frequencies (n={target_size})",
                save_path=f"{IMG_PATH}/{dataset_name}_type_freqs_{target_size}.png",
            )

            # Plot task distributions
            plot_dists(
                orig_label_freqs,
                sampled_label_freqs,
                x_axis="label",
                colors=COLORS[::2],
                top_n=TOP_N,
                # title=f"Relative type frequencies (n={target_size})",
                save_path=f"{IMG_PATH}/{dataset_name}_class_freqs_{target_size}.png",
            )

        # Plot coverages a function of sample size
        plot_coverage(
            TARGET_SIZES,
            type_coverages,
            token_coverages,
            colors=COLORS[::2],
            # title="Type and token coverage of the sub-sampled vocabulary",
            save_path=f"{IMG_PATH}/{dataset_name}_coverage.png",
        )
