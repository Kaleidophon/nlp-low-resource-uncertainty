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


# TODO: Load datasets
# TODO: Define sampling parameters
# TODO: Implement comparison metrics between original and subsampled corpus
# TODO: Plot results
