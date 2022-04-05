"""
Evaluate different aspects of uncertainty and calibration of a classifier.
"""

# STD
import codecs
from collections import defaultdict
from typing import Optional, Dict

# EXT
from einops import rearrange
from nlp_uncertainty_zoo.utils.uncertainty_eval import (
    aupr,
    auroc,
    kendalls_tau,
    ece,
    sce,
    ace,
    coverage_percentage,
    coverage_width,
)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase


def evaluate_uncertainty(
    model,
    id_eval_split: DataLoader,
    ood_eval_split: DataLoader,
    tokenizer: PreTrainedTokenizerBase,
    predictions_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate a model and save predictions (if applicable).

    Parameters
    ----------
    model: Model
        Model to be evaluated.
    id_eval_split: DataSplit
        In-distribution data split the model is being evaluated on.
    odd_eval_split: DataSplit
        Out-of-distribution data split the model is being evaluated on.
    tokenizer: Optional[PreTrainedTokenizerBase]
        Tokenizer of the evaluated model. If given and predictions_path is specified, the input_ids of (sub-word) tokens
        are turned back into strings and saved.
    predictions_path: Optional[str]
        File that predictions are being written to if specified.

    Returns
    -------
    Dict[str, float]
        Return score on test set.
    """
    model_uncertainty_metrics = {
        **model.module.single_prediction_uncertainty_metrics,
        **model.module.multi_prediction_uncertainty_metrics,
    }
    scores = defaultdict(float)
    id_uncertainties = defaultdict(list)
    ood_uncertainties = defaultdict(list)
    loss_func = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    columns = ["sentence", "labels", "predictions"] + list(
        model_uncertainty_metrics.keys()
    )
    predictions_df = pd.DataFrame(columns=columns)
    sentence_i = 0

    for split_name, eval_split, uncertainties in [
        ("id", id_eval_split, id_uncertainties),
        ("ood", ood_eval_split, ood_uncertainties),
    ]:

        split_predictions = []
        split_labels = []
        split_losses = []

        for batch in eval_split:
            attention_mask, input_ids, labels = (
                batch["attention_mask"].to(model.device),
                batch["input_ids"].to(model.device),
                batch["labels"].to(model.device),
            )

            if len(labels.shape) == 2:
                batch_size, seq_len = labels.shape
            else:
                batch_size, seq_len = labels.shape[0], 1

            # Get predictions
            predictions = model.predict(input_ids, attention_mask=attention_mask)

            # Save predictions and labels
            if predictions_path is not None:
                for batch_i in range(batch_size):
                    predictions_df.at[
                        sentence_i + batch_i, "sentence"
                    ] = tokenizer.decode(input_ids[batch_i, :])
                    predictions_df.at[sentence_i + batch_i, "labels"] = (
                        " ".join(str(label) for label in labels[batch_i].cpu().tolist())
                        if seq_len > 1
                        else str(labels[batch_i].cpu().numpy())
                    )
                    predictions_df.at[sentence_i + batch_i, "predictions"] = " ".join(
                        str(pred)
                        for pred in torch.argmax(predictions[batch_i], dim=-1)
                        .detach()
                        .cpu()
                        .tolist()
                    )

            predictions = rearrange(predictions, "b t p -> (b t) p")

            if seq_len > 1:
                labels = rearrange(labels, "b l -> (b l)")

            # Filter irrelevant tokens for language modelling / sequence labelling / token predictions
            ignore_indices = tokenizer.all_special_ids + [-100]
            batch_mask = rearrange(
                torch.all(
                    torch.stack([input_ids != idx for idx in ignore_indices]), dim=0
                ),
                "b s -> (b s)",
            )

            if seq_len > 1:
                predictions = predictions[batch_mask]
                labels = labels[batch_mask]

            split_predictions.append(predictions.detach().cpu().numpy())
            split_labels.append(labels.detach().cpu().numpy())

            # Compute uncertainty
            losses = loss_func(predictions, labels)
            split_losses.append(losses.detach().cpu().numpy())

            for metric_name in model_uncertainty_metrics:
                uncertainty = model.get_uncertainty(
                    input_ids, metric_name=metric_name, attention_mask=attention_mask
                )

                if predictions_path is not None:
                    for batch_i in range(batch_size):
                        predictions_df.at[sentence_i + batch_i, metric_name] = " ".join(
                            f"{score:.4f}"
                            for score in uncertainty[batch_i].detach().cpu().tolist()
                        )

                uncertainty = rearrange(uncertainty, "b l -> (b l)")

                # Filter uncertainties for uninteresting tokens
                if seq_len > 1:
                    uncertainty = uncertainty[batch_mask]

                uncertainties[metric_name].append(uncertainty.detach().cpu().numpy())

            sentence_i += batch_size

        split_predictions = np.concatenate(split_predictions, axis=0)
        split_labels = np.concatenate(split_labels, axis=0)
        split_losses = np.concatenate(split_losses, axis=0)

        # Compute calibration scores
        scores[f"{split_name}_ece"] = ece(split_labels, split_predictions)
        scores[f"{split_name}_sce"] = sce(split_labels, split_predictions)
        scores[f"{split_name}_ace"] = ace(split_labels, split_predictions)

        # Compute coverage scores
        scores[f"{split_name}_coverage_percentage"] = coverage_percentage(
            split_labels, split_predictions
        )
        scores[f"{split_name}_coverage_width"] = coverage_width(split_predictions)

        # Compute Kendall's tau scores
        for metric_name in model_uncertainty_metrics:
            uncertainties[metric_name] = np.concatenate(uncertainties[metric_name])
            scores[f"{split_name}_{metric_name}_kendalls_tau"] = kendalls_tau(
                split_losses, uncertainties[metric_name]
            )

        del split_losses, split_predictions, split_labels

    metric_key = list(model_uncertainty_metrics.keys())[0]
    num_id = len(id_uncertainties[metric_key])
    num_ood = len(ood_uncertainties[metric_key])

    for metric_name in model_uncertainty_metrics:
        scores[f"aupr_{metric_name}"] = aupr(
            [0] * num_id + [1] * num_ood,
            np.concatenate(
                (id_uncertainties[metric_name], ood_uncertainties[metric_name])
            ),
        )
        scores[f"auroc_{metric_name}"] = auroc(
            [0] * num_id + [1] * num_ood,
            np.concatenate(
                (id_uncertainties[metric_name], ood_uncertainties[metric_name])
            ),
        )

    if predictions_path is not None:
        with codecs.open(f"{predictions_path}.csv", "wb", "utf-8") as predictions_file:
            predictions_file.write(predictions_df.to_csv(sep="\t"))

        predictions_df.to_pickle(f"{predictions_path}.pkl")

    return scores
