"""
Create sub-sampled version of training sets and check corpus statistics to ensure that sub-sampling is sensible.
"""

# EXT
from nlp_uncertainty_zoo.utils.samplers import (
    LanguageModellingSampler,
    TokenClassificationSampler,
)

# PROJECT
from src.data import DanPlusBuilder, FinnishUDBuilder, SwahiliWikiBuilder

# CONST
SAMPLING_PARAMS_TOKEN_PRED = {"num_jobs": 2, "seed": 1234}
SAMPLING_PARAMS_LANGUAGE_MODELLING = {
    "num_jobs": 2,
    "seed": 1234,
    "sample_range": [5, 15],
}
TARGET_SIZES = [50, 250, 500, 1000]
DATA_DIR = "data/processed"
MAX_LENGTH = 50
BATCH_SIZE = 1


# TODO: Load datasets
# TODO: Implement comparison metrics between original and subsampled corpus:
# - Sentence length distribution
# - Coverage of vocabulary
# - Distribution of classes
# TODO: Plot results


if __name__ == "__main__":

    for builder, sampler_class, sampler_kwargs in zip(
        [DanPlusBuilder, FinnishUDBuilder, SwahiliWikiBuilder],
        [
            TokenClassificationSampler,
            TokenClassificationSampler,
            LanguageModellingSampler,
        ],
        [
            SAMPLING_PARAMS_TOKEN_PRED,
            SAMPLING_PARAMS_TOKEN_PRED,
            SAMPLING_PARAMS_LANGUAGE_MODELLING,
        ],
    ):
        orig_data = builder(data_dir=DATA_DIR, num_jobs=2, max_length=MAX_LENGTH).build(
            batch_size=BATCH_SIZE
        )

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

            # TODO: Debug
            break

        # TODO: Debug
        break
