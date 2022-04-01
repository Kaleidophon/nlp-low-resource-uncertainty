"""
Evaluate different aspects of uncertainty and calibration of a classifier.
"""

# STD
from typing import List, Optional

# EXT
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

# CONST


def evaluate_uncertainty(
    model,
    id_eval_split: DataLoader,
    ood_eval_split: DataLoader,
    task: str,
    uncertainty_metrics: List[str] = ["aupr"],
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    predictions_path: Optional[str] = None,
) -> float:
    """
    Evaluate a model and save predictions (if applicable).

    Parameters
    ----------
    model: Model
        Model to be evaluated.
    eval_split: DataSplit
        Data split the model is being evaluated on.
    task: str
        Task type, specified using a string.
    tokenizer: Optional[PreTrainedTokenizerBase]
        Tokenizer of the evaluated model. If given and predictions_path is specified, the input_ids of (sub-word) tokens
        are turned back into strings and saved.
    predictions_path: Optional[str]
        File that predictions are being written to if specified.

    Returns
    -------
    float
        Return score on test set.
    """
