# STD
from collections import namedtuple

# EXT
from nlp_uncertainty_zoo.utils.samplers import (
    TokenClassificationSampler,
    LanguageModellingSampler,
    SequenceClassificationSampler,
)
from nlp_uncertainty_zoo.config import AVAILABLE_DATASETS, DATASET_TASKS

# PROJECT
from src.clinc_config import CLINC_MODEL_PARAMS
from src.data import (
    EnglishWikiBuilder,
    SwahiliWikiBuilder,
    FinnishUDBuilder,
    ClincPlusBuilder,
)
from src.danplus_config import DANPLUS_MODEL_PARAMS
from src.enwiki_config import ENWIKI_MODEL_PARAMS
from src.finnish_ud_config import FINNISH_UD_MODEL_PARAMS

# CONST
SamplerConfig = namedtuple("SamplerConfig", ["sampler_class", "sampler_kwargs"])


# AVAILABLE DATASETS AND MODELS
AVAILABLE_DATASETS = {
    **AVAILABLE_DATASETS,
    "enwiki": EnglishWikiBuilder,
    "swwiki": SwahiliWikiBuilder,
    "finnish_ud": FinnishUDBuilder,
    "clinc_plus": ClincPlusBuilder,
}
DATASET_TASKS = {
    **DATASET_TASKS,
    "enwiki": "language_modelling",
    "swwiki": "language_modelling",
    "finnish_ud": "token_classification",
    "clinc_plus": "sequence_classification",
}

DATASET_SAMPLE_CONFIGS = {
    "clinc_plus": lambda train_size: SamplerConfig(
        sampler_class=SequenceClassificationSampler,
        sampler_kwargs={
            "train": {
                "target_size": train_size,
                "num_jobs": 4,
            }
        }
        if train_size is not None
        else {},
    ),
    "dan+": lambda train_size: SamplerConfig(
        sampler_class=TokenClassificationSampler,
        sampler_kwargs={
            "train": {
                "target_size": train_size,
                "num_jobs": 4,
            },
        }
        if train_size is not None
        else {},
    ),
    "enwiki": lambda train_size: SamplerConfig(
        sampler_class=LanguageModellingSampler,
        sampler_kwargs={
            "train": {
                "target_size": train_size,
                "sample_range": [0, 3],
                "num_jobs": 4,
            },
        }
        if train_size is not None
        else {},
    ),
    "finnish_ud": lambda train_size: SamplerConfig(
        sampler_class=TokenClassificationSampler,
        sampler_kwargs={
            "train": {
                "target_size": train_size,
                "num_jobs": 4,
            }
        }
        if train_size is not None
        else {},
    ),
}

MODEL_PARAMS = {
    "dan+": DANPLUS_MODEL_PARAMS,
    "enwiki": ENWIKI_MODEL_PARAMS,
    "finnish_ud": FINNISH_UD_MODEL_PARAMS,
    "clinc_plus": CLINC_MODEL_PARAMS,
}
