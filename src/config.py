# STD
from collections import namedtuple

# EXT
from nlp_uncertainty_zoo.utils.samplers import (
    TokenClassificationSampler,
    LanguageModellingSampler,
)
from nlp_uncertainty_zoo.config import AVAILABLE_DATASETS, DATASET_TASKS

# PROJECT
from src.data import EnglishWikiBuilder, SwahiliWikiBuilder, FinnishUDBuilder
from src.danplus_config import DANPLUS_MODEL_PARAMS
from src.enwiki_config import ENWIKI_MODEL_PARAMS

# CONST
SamplerConfig = namedtuple("SamplerConfig", ["sampler_class", "sampler_kwargs"])


# AVAILABLE DATASETS AND MODELS
AVAILABLE_DATASETS = {
    **AVAILABLE_DATASETS,
    "enwiki": EnglishWikiBuilder,
    "swwiki": SwahiliWikiBuilder,
    "finnish_ud": FinnishUDBuilder,
}
DATASET_TASKS = {
    **DATASET_TASKS,
    "enwiki": "language_modelling",
    "swwiki": "language_modelling",
    "finnish_ud": "token_classification",
}

DATASET_SAMPLE_CONFIGS = {
    "dan+": [
        SamplerConfig(
            sampler_class=TokenClassificationSampler,
            sampler_kwargs={
                "train": {
                    "target_size": train_size,
                    "num_jobs": 4,
                }
                if train_size is not None
                else None,
            },
        )
        for train_size in [4000, 2000, 1000, 500]
    ],
    "enwiki": [
        SamplerConfig(
            sampler_class=LanguageModellingSampler,
            sampler_kwargs={
                "train": {
                    "target_size": train_size,
                    "sample_range": [0, 3],
                    "num_jobs": 4,
                },
            },
        )
        for train_size in [10000, 5000, 2000, 1000, 50]
    ],
}

MODEL_PARAMS = {"dan+": DANPLUS_MODEL_PARAMS, "enwiki": ENWIKI_MODEL_PARAMS}
