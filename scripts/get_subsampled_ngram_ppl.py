"""
Get the perplexity using a trigram language model for some sub-sampled corpus.
"""

# STD
from collections import defaultdict
import codecs
import os

# EXT
import numpy as np
from transformers.tokenization_utils import PreTrainedTokenizer
from nlp_uncertainty_zoo.utils.samplers import (
    LanguageModellingSampler,
    TokenClassificationSampler,
)
from torch.utils.data import DataLoader

# PROJECT
from src.data import (
    DanPlusBuilder,
    FinnishUDBuilder,
    SwahiliWikiBuilder,
    EnglishWikiBuilder,
)

# CONST
# TODO: Turn these into script args
MACHINE_TYPE = "macosx"
RELATIVE_SRILM_PATH = "../../srilm-1.7.3/"
RELATIVE_SRILM_CMD_PATH = f"../../srilm-1.7.3/bin/{MACHINE_TYPE}"
NUM_EVALS = 5

SAMPLING_PARAMS_TOKEN_PRED = {"num_jobs": 2}
SAMPLING_PARAMS_LANGUAGE_MODELLING = {
    "num_jobs": 2,
    "sample_range": [0, 3],
}
TARGET_SIZES = [100, 250, 500, 1000]
DATA_DIR = "../data/processed"
MAX_LENGTH = 50
BATCH_SIZE = 1
IGNORE_TOKENS = [-100, 0, 1, 2, 3, 4]
RESULTS_DIR = "./"


def get_srilm_perplexity(train_path: str, test_path: str) -> float:
    ngrams_path = "./temp.lm"
    weights_path = "./temp2.lm"

    # Build n-gram counts and interpolation weights
    os.system(
        f"{RELATIVE_SRILM_CMD_PATH}/ngram-count -text {train_path} -lm {ngrams_path} -order 3 -gt1min 5"
    )
    os.system(
        f"{RELATIVE_SRILM_CMD_PATH}/ngram-count -text {train_path} -init-lm {ngrams_path} -lm {weights_path} -order 3"
    )

    # Get results
    stream = os.popen(
        f"{RELATIVE_SRILM_CMD_PATH}/ngram -lm {weights_path} -ppl {test_path}"
    )
    output = str(stream.read())
    ppl = float(output.strip().split()[-3])

    # Clean up
    os.remove(ngrams_path)
    os.remove(weights_path)

    return ppl


def write_dl_to_file(
    data_loader: DataLoader, tmp_path: str, tokenizer: PreTrainedTokenizer
):

    with codecs.open(tmp_path, "w", "utf-8") as tmp_file:
        for batch in data_loader:
            input_ids = batch["input_ids"]
            line = tokenizer.decode(
                input_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            tmp_file.write(f"{line}\n")


if __name__ == "__main__":
    # Set some environment variables for SRILM
    os.environ["SRILM"] = os.path.abspath(RELATIVE_SRILM_PATH)
    os.environ["MACHINE_TYPE"] = MACHINE_TYPE
    os.environ[
        "PATH"
    ] = f"{os.environ['PATH']}:{os.path.abspath(RELATIVE_SRILM_PATH)}/bin/{MACHINE_TYPE}"

    # TODO: Also score OOD datasets

    # Sub-sample training data and save to temporary file
    for dataset_name, builder, sampler_class, sampler_kwargs in zip(
        ["dan+", "finnish_ud", "swahili_wiki", "english_wiki"],
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
        tmp_path = "./tmp.txt"
        tmp_train_path = "./tmp_train.txt"
        tmp_test_path = "./tmp_test.txt"

        # Retrieve ppl of original corpus
        data_builder = builder(data_dir=DATA_DIR, num_jobs=2, max_length=MAX_LENGTH)
        orig_data = data_builder.build(BATCH_SIZE, shuffle=True)
        write_dl_to_file(orig_data["train"], tmp_train_path, data_builder.tokenizer)
        write_dl_to_file(orig_data["test"], tmp_test_path, data_builder.tokenizer)
        original_train_ppl = get_srilm_perplexity(tmp_train_path, tmp_train_path)
        original_test_ppl = get_srilm_perplexity(tmp_train_path, tmp_test_path)

        # Clean up
        os.remove(tmp_train_path)
        os.remove(tmp_test_path)
        del orig_data

        ppls = defaultdict(list)

        for target_size in TARGET_SIZES:

            for _ in range(NUM_EVALS):
                data_builder = builder(
                    data_dir=DATA_DIR,
                    num_jobs=2,
                    max_length=MAX_LENGTH,
                    sampler_class=sampler_class,
                    sampler_kwargs={
                        "train": {"target_size": target_size, **sampler_kwargs}
                    },
                )
                subsampled_data = data_builder.build(batch_size=BATCH_SIZE)

                write_dl_to_file(
                    subsampled_data["train"], tmp_path, data_builder.tokenizer
                )
                del subsampled_data

                # Get perplexities over multiple runs
                ppls[target_size].append(
                    get_srilm_perplexity(data_builder.splits["train"], tmp_path)
                )

                # Remove temporary file
                os.remove(tmp_path)

        with open(f"{RESULTS_DIR}/{dataset_name}_results.txt", "w") as results_file:
            results_file.write(
                f"Original PPL: Train {original_train_ppl:.4f} / Test {original_test_ppl:.4f}\n"
            )

            for target_size, ppls in ppls.items():
                results_file.write(
                    f"n={target_size}: {np.mean(ppls):.4f}±{np.std(ppls):.4f}\n"
                )
