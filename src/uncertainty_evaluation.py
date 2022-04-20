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
    id_eval_split: DataLoader
        In-distribution data split the model is being evaluated on.
    ood_eval_split: DataLoader
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
    model_uncertainty_metrics = list(
        model.module.single_prediction_uncertainty_metrics.keys()
    ) + list(model.module.multi_prediction_uncertainty_metrics.keys())
    loss_func = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    # Initialize data structure that track stats
    scores = defaultdict(float)  # Final scores
    # Uncertainties for tokens and sequences (in-distribution)
    id_uncertainties, id_seq_uncertainties = defaultdict(list), defaultdict(list)
    # Uncertainties for tokens and sequences (out-of-distribution)
    ood_uncertainties, ood_seq_uncertainties = defaultdict(list), defaultdict(list)

    # Initialize result df that will later be written to .csv
    columns = ["sentence", "labels", "predictions"] + model_uncertainty_metrics
    predictions_df = pd.DataFrame(columns=columns)
    sentence_i = 0

    # Get scores for both test splits
    for split_name, eval_split, uncertainties, seq_uncertainties in [
        ("id", id_eval_split, id_uncertainties, id_seq_uncertainties),
        ("ood", ood_eval_split, ood_uncertainties, ood_seq_uncertainties),
    ]:
        split_predictions = []
        split_labels = []
        split_losses = []
        split_seq_losses = []

        for batch in eval_split:
            attention_mask, input_ids, labels = (
                batch["attention_mask"].to(model.device),
                batch["input_ids"].to(model.device),
                batch["labels"].to(model.device),
            )

            # Determine if sequence labelling / token prediction or sequence predction
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
                    ] = tokenizer.decode(
                        input_ids[batch_i, :]
                    )  # Get tokens from ids
                    predictions_df.at[sentence_i + batch_i, "labels"] = (
                        " ".join(str(label) for label in labels[batch_i].cpu().tolist())
                        if seq_len > 1
                        else str(labels[batch_i].cpu().numpy())
                    )  # Write labels
                    predictions_df.at[sentence_i + batch_i, "predictions"] = " ".join(
                        str(pred)
                        for pred in torch.argmax(predictions[batch_i], dim=-1)
                        .detach()
                        .cpu()
                        .tolist()
                    )  # Write predictions

            predictions = rearrange(predictions, "b t p -> (b t) p")

            # Filter irrelevant tokens for language modelling / sequence labelling / token predictions
            ignore_indices = tokenizer.all_special_ids + [-100]
            batch_mask = rearrange(
                torch.all(
                    torch.stack([input_ids != idx for idx in ignore_indices]), dim=0
                ),
                "b s -> (b s)",
            ).to(model.device)
            seq_batch_mask = rearrange(batch_mask, "(b s) -> b s", b=batch_size).to(
                model.device
            )

            if seq_len > 1:
                labels = rearrange(labels, "b l -> (b l)")

                # Mask out losses for ignore tokens and recompute sequence losses
                seq_losses = rearrange(
                    loss_func(predictions, labels), "(b l) -> b l", b=batch_size
                )
                seq_losses *= (
                    seq_batch_mask.int()
                )  # Mask out all uninteresting tokens' uncertainties
                seq_losses = seq_losses.mean(dim=1)
                seq_losses *= seq_len
                seq_losses /= seq_batch_mask.int().sum(dim=1)
                split_seq_losses.append(seq_losses.detach().cpu().numpy())

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

                seq_uncertainty = torch.clone(uncertainty)
                uncertainty = rearrange(uncertainty, "b l -> (b l)")

                # Filter uncertainties for uninteresting tokens
                if seq_len > 1:
                    uncertainty = uncertainty[batch_mask]

                    # Get the sequence uncertainties setting non batch-mask tokens to zero and re-normalizing means
                    # across second axis
                    seq_uncertainty *= (
                        seq_batch_mask.int()
                    )  # Mask out all uninteresting tokens' uncertainties
                    seq_uncertainty = seq_uncertainty.mean(dim=1)
                    seq_uncertainty *= seq_len
                    seq_uncertainty /= seq_batch_mask.int().sum(dim=1)
                    seq_uncertainties[metric_name].append(
                        seq_uncertainty.detach().cpu().numpy()
                    )

                # Sequence classification tasks, sequence uncertainties are just the uncertainties of the single
                # sequence-wide prediction
                else:
                    seq_uncertainties[metric_name].append(
                        uncertainty.detach().cpu().numpy()
                    )

                uncertainties[metric_name].append(uncertainty.detach().cpu().numpy())

            sentence_i += batch_size

        split_predictions = np.concatenate(split_predictions, axis=0)
        split_labels = np.concatenate(split_labels, axis=0)
        split_losses = np.concatenate(split_losses, axis=0)

        # Mask out predictions for -100
        split_labels = split_labels[split_labels != -100]
        split_predictions = split_predictions[split_labels != -100]

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

            # TODO: Potentially remove this control flow
            if len(uncertainties[metric_name]) > 2 and len(split_losses) > 2:
                scores[f"{split_name}_{metric_name}_kendalls_tau_token"] = kendalls_tau(
                    split_losses, uncertainties[metric_name]
                )
            else:
                scores[f"{split_name}_{metric_name}_kendalls_tau_token"] = 0

            # TODO: Potentially remove this
            if np.isnan(scores[f"{split_name}_{metric_name}_kendalls_tau_token"]):
                print(
                    f"NaN found for {split_name}_{metric_name}, values:\n{split_losses}\n{uncertainties[metric_name]}"
                )

            if seq_len > 1:
                seq_uncertainties[metric_name] = np.concatenate(
                    seq_uncertainties[metric_name]
                )
                scores[f"{split_name}_{metric_name}_kendalls_tau_seq"] = kendalls_tau(
                    split_seq_losses, seq_uncertainties[metric_name]
                )

        del split_losses, split_predictions, split_labels

    metric_key = model_uncertainty_metrics[0]
    num_id = len(id_seq_uncertainties[metric_key])
    num_ood = len(ood_seq_uncertainties[metric_key])

    for metric_name in model_uncertainty_metrics:
        scores[f"aupr_{metric_name}"] = aupr(
            [0] * num_id + [1] * num_ood,
            np.concatenate(
                (id_seq_uncertainties[metric_name], ood_seq_uncertainties[metric_name])
            ),
        )
        scores[f"auroc_{metric_name}"] = auroc(
            [0] * num_id + [1] * num_ood,
            np.concatenate(
                (id_seq_uncertainties[metric_name], ood_seq_uncertainties[metric_name])
            ),
        )

    if predictions_path is not None:
        with codecs.open(f"{predictions_path}.csv", "wb", "utf-8") as predictions_file:
            predictions_file.write(predictions_df.to_csv(sep="\t"))

        predictions_df.to_pickle(f"{predictions_path}.pkl")

    return scores
