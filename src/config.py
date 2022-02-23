# EXT
from nlp_uncertainty_zoo.config import AVAILABLE_DATASETS, DATASET_TASKS

# PROJECT
from src.data import EnglishWikiBuilder, SwahiliWikiBuilder, FinnishUDBuilder

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
