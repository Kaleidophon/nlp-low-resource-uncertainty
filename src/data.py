"""
Define datasets used in this project.
"""

# STD
from typing import Optional, Type

# EXT
from nlp_uncertainty_zoo.utils.data import (
    DanPlusBuilder,
    LanguageModellingDatasetBuilder,
    ClassificationDatasetBuilder,
    SamplerKwargs,
)
from transformers import (
    BertTokenizerFast,
    BertTokenizer,
)

# TODO: Add English Wikipedia here


class FinnishUDBuilder(ClassificationDatasetBuilder):
    """
    Dataset class for the Dan+ dataset.
    """

    def __init__(
        self,
        data_dir: str,
        max_length: int,
        sampler_class: Optional[Type] = None,
        sampler_kwargs: Optional[SamplerKwargs] = None,
        num_jobs: Optional[int] = 1,
    ):
        super().__init__(
            name="finnish_ud",
            data_dir=data_dir,
            splits={
                "train": f"{data_dir}/finnish/train.csv",
                "valid": f"{data_dir}/finnish/val.csv",
                "test": f"{data_dir}/finnish/test.csv",
                "ood_test": f"{data_dir}/finnish/ood_test.csv",
            },
            type_="token_classification",
            tokenizer=BertTokenizerFast.from_pretrained(
                "alexanderfalk/danbert-small-cased"
            ),
            max_length=max_length,
            sampler_class=sampler_class,
            sampler_kwargs=sampler_kwargs,
            num_jobs=num_jobs,
        )


class SwahiliWikiBuilder(LanguageModellingDatasetBuilder):
    """
    Dataset class for the Swahili Wikipedia.
    """

    def __init__(
        self,
        data_dir: str,
        max_length: int,
        sampler_class: Optional[Type] = None,
        sampler_kwargs: Optional[SamplerKwargs] = None,
        num_jobs: Optional[int] = 1,
    ):
        super().__init__(
            name="swahili_wiki",
            data_dir=data_dir,
            splits={
                "train": f"{data_dir}/swahili/train.txt",
                "valid": f"{data_dir}/swahili/val.txt",
                "test": f"{data_dir}/swahili/test.txt",
                "ood_test": f"{data_dir}/swahili/ood_test.txt",
            },
            type_="next_token_prediction",
            tokenizer=BertTokenizer.from_pretrained("bert-base-multilingual-cased"),
            max_length=max_length,
            sampler_class=sampler_class,
            sampler_kwargs=sampler_kwargs,
            num_jobs=num_jobs,
        )
