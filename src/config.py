# EXT
from nlp_uncertainty_zoo.config import AVAILABLE_DATASETS, DATASET_TASKS

# PROJECT
from src.data import EnglishWikiBuilder, SwahiliWikiBuilder, FinnishUDBuilder
from src.danplus_config import DANPLUS_MODEL_PARAMS
from src.enwiki_config import ENWIKI_MODEL_PARAMS

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

MODEL_PARAMS = {"dan+": DANPLUS_MODEL_PARAMS, "enwiki": ENWIKI_MODEL_PARAMS}
