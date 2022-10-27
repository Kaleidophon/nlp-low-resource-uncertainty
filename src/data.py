"""
Define datasets used in this project.
"""

# STD
from typing import Optional, Type

# EXT
from nlp_uncertainty_zoo.utils.data import (
    LanguageModellingDatasetBuilder,
    ClassificationDatasetBuilder,
    SamplerKwargs,
)
from transformers import (
    BertTokenizerFast,
    BertTokenizer,
)


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
                "train": f"{data_dir}/finnish_ud/train.csv",
                "valid": f"{data_dir}/finnish_ud/val.csv",
                "test": f"{data_dir}/finnish_ud/test.csv",
                "ood_test": f"{data_dir}/finnish_ud/ood_test.csv",
            },
            type_="token_classification",
            tokenizer=BertTokenizerFast.from_pretrained(
                "TurkuNLP/bert-base-finnish-cased-v1"
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
                "train": f"{data_dir}/swwiki/train.txt",
                "valid": f"{data_dir}/swwiki/val.txt",
                "test": f"{data_dir}/swwiki/test.txt",
                "ood_test": f"{data_dir}/swwiki/ood_test.txt",
            },
            type_="next_token_prediction",
            tokenizer=BertTokenizer.from_pretrained("bert-base-multilingual-cased"),
            max_length=max_length,
            sampler_class=sampler_class,
            sampler_kwargs=sampler_kwargs,
            num_jobs=num_jobs,
        )


class EnglishWikiBuilder(LanguageModellingDatasetBuilder):
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
            name="english_wiki",
            data_dir=data_dir,
            splits={
                "train": f"{data_dir}/enwiki/train.txt",
                "valid": f"{data_dir}/enwiki/val.txt",
                "test": f"{data_dir}/enwiki/test.txt",
                "ood_test": f"{data_dir}/enwiki/ood_test.txt",
            },
            type_="next_token_prediction",
            tokenizer=BertTokenizer.from_pretrained("bert-base-cased"),
            max_length=max_length,
            sampler_class=sampler_class,
            sampler_kwargs=sampler_kwargs,
            num_jobs=num_jobs,
        )


class ClincPlusBuilder(ClassificationDatasetBuilder):
    """
    Dataset class for the CLINC Plus OOS dataset.
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
            name="clinc",
            data_dir=data_dir,
            splits={
                "train": f"{data_dir}/clinc_plus/train.csv",
                "valid": f"{data_dir}/clinc_plus/val.csv",
                "test": f"{data_dir}/clinc_plus/test.csv",
                "ood_test": f"{data_dir}/clinc_plus/oos_test.csv",
            },
            type_="sequence_classification",
            tokenizer=BertTokenizer.from_pretrained("bert-base-uncased"),
            max_length=max_length,
            sampler_class=sampler_class,
            sampler_kwargs=sampler_kwargs,
            num_jobs=num_jobs,
        )


class DanPlusBuilder(ClassificationDatasetBuilder):
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
            name="dan+",
            data_dir=data_dir,
            splits={
                "train": f"{data_dir}/danplus/train.csv",
                "valid": f"{data_dir}/danplus/val.csv",
                "test": f"{data_dir}/danplus/test.csv",
                "ood_test": f"{data_dir}/danplus/ood_test.csv",
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
