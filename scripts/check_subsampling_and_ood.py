"""
Create sub-sampled version of training sets and check corpus statistics to ensure that sub-sampling is sensible.
"""

# STD
from collections import Counter
import os
import string
from typing import Tuple, List, Optional

# EXT
from sklearn.preprocessing import LabelEncoder
from transformers.tokenization_utils import PreTrainedTokenizerBase
import numpy as np
import matplotlib.pyplot as plt
from nlp_uncertainty_zoo.utils.samplers import (
    TokenClassificationSampler,
    SequenceClassificationSampler,
)
from torch.utils.data import DataLoader
import pickle

# PROJECT
from src.data import DanPlusBuilder, FinnishUDBuilder, ClincPlusBuilder

# CONST
SAMPLING_PARAMS_TOKEN_PRED = {"num_jobs": 2, "seed": 1234}
SAMPLING_PARAMS_LANGUAGE_MODELLING = {
    "num_jobs": 2,
    "seed": 1234,
    "sample_range": [0, 3],
}
TARGET_SIZES = [100, 1000]
DATA_DIR = "data/processed"
MAX_LENGTH = 50
BATCH_SIZE = 1
IGNORE_PUNCTUATION = list(string.punctuation) + ["”"] + list(map(str, range(0, 10)))
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
PICKLE_PATH = "scripts/subsampling_data.pkl"
COLORS = ["firebrick", "forestgreen", "midnightblue", "orangered"]
EDGECOLORS = ["lightcoral", "yellowgreen", "skyblue", "lightsalmon"]
HATCHES = ["..", "//", "\\\\", "xx", "--"]
DISPLAY_NAMES = {
    "danplus": "Dan+",
    "finnish_ud": "Finnish UD",
    "clinc_plus": "Clinc Plus",
}


def collect_sentence_length_and_class_dict(
    dataloader: DataLoader,
    ignore_tokens: List[int],
    stop_words: Optional[List[str]] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    label_encoder: Optional[LabelEncoder] = None,
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

        if stop_words is not None and tokenizer is not None:
            sentence = tokenizer.decode(input_ids)

            for char in IGNORE_PUNCTUATION:
                sentence = sentence.replace(char, "")

            input_ids = list(
                filter(lambda token: token not in stop_words, sentence.split())
            )

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

    if label_encoder is not None:
        class_freqs = Counter(
            {
                label_encoder.inverse_transform([key])[0]: value
                for key, value in class_freqs.items()
            }
        )

    # if tokenizer is not None:
    #    token_freqs = Counter({
    #        tokenizer.decode(key): value
    #        for key, value in token_freqs.items()
    #    })

    return seq_freqs, token_freqs, class_freqs


def create_id_ood_plot(
    dataset_names, data, x_label, target, top_n=None, sort=False, text_labels=False
):

    fig, ax = plt.subplots(figsize=(10, 5), ncols=len(dataset_names), sharey="row")
    fig.supxlabel(x_label, alpha=0.6, y=0.06, x=0.525)
    bar_width = 2
    skip = 2.5
    data_labels = ["ID", "OOD"]

    for a in ax:
        a.grid(visible=True, axis="both", which="major", linestyle=":", color="grey")

    for ax_num, dataset_name in enumerate(dataset_names):
        data_id = data[dataset_name][f"orig_{target}"]
        data_ood = data[dataset_name][f"ood_{target}"]

        if top_n is not None:
            data_id = dict(data_id.most_common(top_n))
            data_ood = {type_: data_ood[type_] for type_ in data_id}
        else:
            top_n = len(data_id)

        x = np.arange(0, top_n * (skip + 2 * bar_width), skip + 2 * bar_width)

        # Sort keys by frequency, decendingly
        if sort:
            sorted_keys = list(zip(*sorted(data_id.items(), key=lambda t: t[1])))[0]

        else:
            sorted_keys = np.arange(top_n)

        if text_labels:
            plt.sca(ax[ax_num])
            plt.xticks(x, sorted_keys, fontsize=8, alpha=0.7)
            plt.setp(
                ax[ax_num].get_xticklabels(),
                rotation=50,
                ha="right",
                rotation_mode="anchor",
            )

        else:
            plt.sca(ax[ax_num])
            plt.xticks(range(0, (top_n + 2) * 5 + 1, top_n), range(0, top_n + 1, 5))

        total_orig = sum(data_id.values())
        total_ood = sum(data_ood.values())
        freqs_id = np.zeros(top_n)
        freqs_ood = np.zeros(top_n)

        for i, key in enumerate(sorted_keys):
            freqs_id[i] = data_id.get(key, 0) / total_orig

            try:
                freqs_ood[i] = data_ood.get(key, 0) / total_ood
            except ZeroDivisionError:
                freqs_ood[i] = 0

        ax[ax_num].bar(
            x - bar_width / 2,
            freqs_id,
            label="ID",
            width=bar_width,
            color=COLORS[0],
            hatch=HATCHES[0],
            edgecolor=EDGECOLORS[0],
            linewidth=0.5,
            alpha=0.7,
        )
        ax[ax_num].bar(
            x + bar_width / 2,
            freqs_ood,
            label="OOD",
            width=bar_width,
            color=COLORS[1],
            hatch=HATCHES[1],
            edgecolor=EDGECOLORS[1],
            linewidth=0.5,
            alpha=0.7,
        )
        ax[ax_num].title.set_text(DISPLAY_NAMES[dataset_name])
        # ax[ax_num].set_xticklabels([])
        # if sort:
        #    ax[ax_num].set_xticklabels(range(top_n + 5, 0, -5))

    ax = ax[0]
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]

    # prepare y-axis
    ax.set_ylabel("Relative frequency", alpha=0.6)
    ax.legend(labels=data_labels, handles=handles, loc="upper left")
    # ax.axvline(x=2.5, c="black")
    fig.tight_layout()

    plt.savefig(
        f"{IMG_PATH}/{target}_id_ood.pdf", format="pdf", dpi=120, bbox_inches="tight"
    )


def create_subsampled_plot(
    dataset_names, data, x_label, target, top_n=None, sort=False, text_labels=False
):

    fig, ax = plt.subplots(figsize=(10, 5), ncols=len(dataset_names), sharey="row")
    fig.supxlabel(x_label, alpha=0.6, y=0.06, x=0.525)
    bar_width = 1
    skip = 2.5

    for a in ax:
        a.grid(visible=True, axis="both", which="major", linestyle=":", color="grey")

    for ax_num, dataset_name in enumerate(dataset_names):
        data_original = data[dataset_name][f"orig_{target}"]
        data_subsampled = data[dataset_name]["subsampled"]

        if top_n is not None:
            data_original = dict(data_original.most_common(top_n))
            data_subsampled = {
                target_size: {
                    f"sampled_{target}": {
                        type_: data_subsampled[target_size][f"sampled_{target}"][type_]
                        for type_ in data_original
                    }
                }
                for target_size in TARGET_SIZES
            }

        else:
            top_n = len(data_original.keys())

        x = np.arange(
            0,
            top_n * (skip + len(TARGET_SIZES) * bar_width),
            skip + len(TARGET_SIZES) * bar_width,
        )
        target_sizes = list(data[dataset_name]["subsampled"].keys())

        # Sort keys by frequency, decendingly
        if sort:
            sorted_keys = list(zip(*sorted(data_original.items(), key=lambda t: t[1])))[
                0
            ]

        else:
            sorted_keys = np.arange(top_n)

        if text_labels:
            plt.sca(ax[ax_num])
            plt.xticks(x, sorted_keys, fontsize=8, alpha=0.7)
            plt.setp(
                ax[ax_num].get_xticklabels(),
                rotation=50,
                ha="right",
                rotation_mode="anchor",
            )

        else:
            plt.sca(ax[ax_num])
            plt.xticks(range(0, top_n * 5 + 1, top_n), range(0, top_n + 6, 5))

        total_orig = sum(data_original.values())

        freqs_subsampled = {
            target_size: np.zeros(top_n) for target_size in target_sizes
        }
        total_subsampled = {
            target_size: sum(data_subsampled[target_size][f"sampled_{target}"].values())
            for target_size in target_sizes
        }
        freqs_id = np.zeros(top_n)

        for i, key in enumerate(sorted_keys):
            freqs_id[i] = data_original.get(key, 0) / total_orig

            for target_size in target_sizes:
                try:
                    freqs_subsampled[target_size][i] = (
                        data[dataset_name]["subsampled"][target_size][
                            f"sampled_{target}"
                        ].get(key, 0)
                        / total_subsampled[target_size]
                    )
                except ZeroDivisionError:
                    freqs_subsampled[target_size][i] = 0

        ax[ax_num].bar(
            x - bar_width,
            freqs_id,
            label="Original",
            width=bar_width,
            color=COLORS[0],
            hatch=HATCHES[0],
            edgecolor=EDGECOLORS[0],
            linewidth=0.5,
            alpha=0.7,
        )

        for i, target_size in enumerate(target_sizes):
            ax[ax_num].bar(
                x - bar_width + (i + 1) * bar_width,
                freqs_subsampled[target_size],
                label=target_size,
                width=bar_width,
                color=COLORS[1 + i],
                hatch=HATCHES[1 + i],
                edgecolor=EDGECOLORS[1 + i],
                linewidth=0.5,
                alpha=0.7,
            )
        ax[ax_num].title.set_text(DISPLAY_NAMES[dataset_name])

        # ax[ax_num].set_xticklabels([])
        # ax[ax_num].set_xlim([0, top_n * len(TARGET_SIZES)])

        # if sort:
        #    ax[ax_num].set_xticklabels(range(top_n + 5, 0, -5))

    ax = ax[0]
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]

    # prepare y-axis
    ax.set_ylabel("Relative frequency", alpha=0.6)
    ax.legend(labels=["Original"] + TARGET_SIZES, handles=handles, loc="upper left")
    # ax.axvline(x=2.5, c="black")
    fig.tight_layout()

    plt.savefig(
        f"{IMG_PATH}/{target}_subsampled.pdf",
        format="pdf",
        dpi=120,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    datasets = ["danplus", "finnish_ud", "clinc_plus"]
    all_stop_words = [
        list(map(lambda line: line.strip(), open(path, "r").readlines()))
        for path in [
            "scripts/danish_stopwords.txt",
            "scripts/finnish_stopwords.txt",
            "scripts/english_stopwords.txt",
        ]
    ]
    data = {
        dataset: {"subsampled": {target_size: {} for target_size in TARGET_SIZES}}
        for dataset in datasets
    }

    if os.path.exists(PICKLE_PATH):
        with open(PICKLE_PATH, "rb") as pkl_file:
            data = pickle.load(pkl_file)

    else:
        for (
            dataset_name,
            builder_class,
            stop_words,
            sampler_class,
            sampler_kwargs,
        ) in zip(
            datasets,
            [DanPlusBuilder, FinnishUDBuilder, ClincPlusBuilder],
            all_stop_words,
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

            builder = builder_class(
                data_dir=DATA_DIR, num_jobs=2, max_length=MAX_LENGTH
            )
            orig_data = builder.build(batch_size=BATCH_SIZE)
            (
                orig_seq_freqs,
                orig_token_freqs,
                orig_label_freqs,
            ) = collect_sentence_length_and_class_dict(
                orig_data["train"],
                ignore_tokens=[-100] + builder.tokenizer.all_special_ids,
                tokenizer=builder.tokenizer,
                stop_words=stop_words,
                label_encoder=builder.label_encoder,
            )

            data[dataset_name]["orig_seq_freqs"] = orig_seq_freqs
            data[dataset_name]["orig_token_freqs"] = orig_token_freqs
            data[dataset_name]["orig_label_freqs"] = orig_label_freqs

            # Perform analyses on OOD data
            (
                ood_seq_freqs,
                ood_token_freqs,
                ood_label_freqs,
            ) = collect_sentence_length_and_class_dict(
                orig_data["ood_test"],
                ignore_tokens=[-100] + builder.tokenizer.all_special_ids,
                tokenizer=builder.tokenizer,
                stop_words=stop_words,
                label_encoder=builder.label_encoder,
            )

            data[dataset_name]["ood_seq_freqs"] = ood_seq_freqs
            data[dataset_name]["ood_token_freqs"] = ood_token_freqs
            data[dataset_name]["ood_label_freqs"] = ood_label_freqs

            del orig_data

            for target_size in TARGET_SIZES:
                subsampled_data = builder_class(
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
                    subsampled_data["train"],
                    ignore_tokens=[-100] + builder.tokenizer.all_special_ids,
                    tokenizer=builder.tokenizer,
                    stop_words=stop_words,
                    label_encoder=builder.label_encoder,
                )

                data[dataset_name]["subsampled"][target_size][
                    "sampled_seq_freqs"
                ] = sampled_seq_freqs
                data[dataset_name]["subsampled"][target_size][
                    "sampled_token_freqs"
                ] = sampled_token_freqs
                data[dataset_name]["subsampled"][target_size][
                    "sampled_label_freqs"
                ] = sampled_label_freqs

                del subsampled_data

        with open(PICKLE_PATH, "wb") as pkl_file:
            pickle.dump(data, pkl_file)

    # Create the following plots
    datasets_wo_clinc = datasets[:2]

    # Sequence length distributions ID / OOD for all languages side by side
    create_id_ood_plot(
        datasets,
        data,
        x_label="Sequence length",
        top_n=25,
        target="seq_freqs",
        sort=False,
    )

    # Top 25 type distributions ID / OOD for all languages side by side
    create_id_ood_plot(
        datasets,
        data,
        x_label="Type Rank",
        target="token_freqs",
        top_n=25,
        sort=True,
        text_labels=True,
    )

    # Label frequency comparisons between ID / OOD
    create_id_ood_plot(
        datasets_wo_clinc,
        data,
        x_label="Label Rank",
        target="label_freqs",
        sort=True,
        text_labels=True,
    )

    # Relative frequency of labels for finnish / danish for original and subsampled
    create_subsampled_plot(
        datasets_wo_clinc,
        data,
        x_label="Label Rank",
        target="label_freqs",
        sort=True,
        text_labels=True
        # break_yaxis={"hspace": 0.05}
    )

    # Relative frequency of sentence length for all languages for original and subsampled
    create_subsampled_plot(
        datasets,
        data,
        x_label="Sequence length",
        target="seq_freqs",
        top_n=20,
        sort=False,
    )

    # Relative frequency of top 25 types for all languages for original and subsampled
    create_subsampled_plot(
        datasets,
        data,
        x_label="Type Rank",
        target="token_freqs",
        top_n=20,
        sort=True,
        text_labels=True,
    )
