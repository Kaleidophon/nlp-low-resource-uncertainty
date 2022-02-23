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
    LanguageModellingSampler,
    TokenClassificationSampler,
)
from nlp_uncertainty_zoo.utils.data import LanguageModellingDatasetBuilder
from torch.utils.data import DataLoader

# PROJECT
from src.data import (
    DanPlusBuilder,
    FinnishUDBuilder,
    SwahiliWikiBuilder,
    EnglishWikiBuilder,
)

# CONST
SAMPLING_PARAMS_TOKEN_PRED = {"num_jobs": 2, "seed": 1234}
SAMPLING_PARAMS_LANGUAGE_MODELLING = {
    "num_jobs": 2,
    "seed": 1234,
    "sample_range": [0, 3],
}
TARGET_SIZES = [100, 250, 500, 1000]
DATA_DIR = "../data/processed"
MAX_LENGTH = 50
BATCH_SIZE = 1
IGNORE_TOKENS = [-100, 0, 1, 2, 3, 4]
IMG_PATH = "../img"
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
}


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
        sns.set(font_scale=3, rc={"xtick.bottom": True, **PLOT_STYLE})
        sns.set_palette("viridis")
        plot = sns.barplot(
            data=data,
            x="size",
            y="coverage",
            hue="kind",
            alpha=0.8,
            ci=None,
            edgecolor="black",
        )

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
            "type": list(freqs_orig.values()) * 2,
            "relative frequencies": freqs,
            "corpus": ["original"] * num_types + [compare_label] * num_types,
        }
    )

    with sns.axes_style("whitegrid"):
        sns.set(rc={"figure.figsize": (15, 10), **PLOT_STYLE}, font_scale=3)
        plot = sns.barplot(
            data=data,
            x="type",
            y="relative frequencies",
            hue="corpus",
            alpha=0.8,
            ci=None,
            edgecolor="black",
        )
        plot.set(xticklabels=[])

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
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    compare_label: str = "subsampled",
):
    """
    Plot distributions of sentence lengths in the same histogram.
    """
    max_length = max(max(freqs_orig.keys()), max(freqs_sampled.keys()))
    total_orig = sum(freqs_orig.values())
    total_sampled = sum(freqs_sampled.values())
    freqs = np.zeros(2 * max_length)

    for length in range(max_length):
        freqs[length] = freqs_orig.get(length, 0) / total_orig
        freqs[max_length + length] = freqs_sampled.get(length, 0) / total_sampled

    data = pd.DataFrame.from_dict(
        {
            "sequence_length": list(range(max_length)) + list(range(max_length)),
            "relative frequencies": freqs,
            "corpus": ["original"] * max_length + [compare_label] * max_length,
        }
    )

    with sns.axes_style("whitegrid"):
        sns.set(
            rc={"figure.figsize": (14, 10), "xtick.bottom": True, **PLOT_STYLE},
            font_scale=3,
        )

        plot = sns.barplot(
            data=data,
            x="sequence_length",
            y="relative frequencies",
            hue="corpus",
            alpha=0.8,
            ci=None,
            edgecolor="black",
        )

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
        ["danplus", "finnish_ud", "swahili_wiki", "english_wiki"],
        [DanPlusBuilder, FinnishUDBuilder, SwahiliWikiBuilder, EnglishWikiBuilder],
        [
            TokenClassificationSampler,
            TokenClassificationSampler,
            LanguageModellingSampler,
            LanguageModellingSampler,
        ],
        [
            SAMPLING_PARAMS_TOKEN_PRED,
            SAMPLING_PARAMS_TOKEN_PRED,
            SAMPLING_PARAMS_LANGUAGE_MODELLING,
            SAMPLING_PARAMS_LANGUAGE_MODELLING,
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
            title="Relative sentence length frequencies",
            save_path=f"{IMG_PATH}/{dataset_name}_seq_lengths_ood.png",
            compare_label="ood",
        )

        # Plot token distributions
        plot_dists(
            orig_token_freqs,
            ood_token_freqs,
            top_n=50,
            title="Relative type frequencies",
            save_path=f"{IMG_PATH}/{dataset_name}_type_freqs_ood.png",
            compare_label="ood",
        )

        # Plot task distributions
        plot_dists(
            orig_label_freqs,
            ood_label_freqs,
            top_n=50 if isinstance(builder, LanguageModellingDatasetBuilder) else None,
            title="Relative type frequencies",
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
                title=f"Relative sentence length frequencies (n={target_size})",
                save_path=f"{IMG_PATH}/{dataset_name}_seq_lengths_{target_size}.png",
            )

            # Plot token distributions
            plot_dists(
                orig_token_freqs,
                sampled_token_freqs,
                top_n=50,
                title=f"Relative type frequencies (n={target_size})",
                save_path=f"{IMG_PATH}/{dataset_name}_type_freqs_{target_size}.png",
            )

            # Plot task distributions
            plot_dists(
                orig_label_freqs,
                sampled_label_freqs,
                top_n=50
                if isinstance(builder, LanguageModellingDatasetBuilder)
                else None,
                title=f"Relative type frequencies (n={target_size})",
                save_path=f"{IMG_PATH}/{dataset_name}_class_freqs_{target_size}.png",
            )

        # Plot coverages a function of sample size
        plot_coverage(
            TARGET_SIZES,
            type_coverages,
            token_coverages,
            title="Type and token coverage of the sub-sampled vocabulary",
            save_path=f"{IMG_PATH}/{dataset_name}_coverage.png",
        )
