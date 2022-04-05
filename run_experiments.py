"""
Execute experiments.
"""

# STD
import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
import pickle
import types
from typing import List, Dict, Optional

# EXT
from codecarbon import OfflineEmissionsTracker
from knockknock import telegram_sender
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.tokenization_utils import PreTrainedTokenizerBase
import wandb
from nlp_uncertainty_zoo.utils.task_eval import evaluate
from nlp_uncertainty_zoo.config import AVAILABLE_MODELS
from nlp_uncertainty_zoo.utils.custom_types import Device, WandBRun

# PROJECT
from src.config import MODEL_PARAMS, AVAILABLE_DATASETS
from src.uncertainty_evaluation import evaluate_uncertainty

# CONST
SEED = 123456
RESULT_DIR = "./results"
MODEL_DIR = "./models"
DATA_DIR = "data/processed"
EMISSION_DIR = "./emissions"
PROJECT_NAME = "nlp-low-resource-uncertainty"

# GLOBALS
SECRET_IMPORTED = False

# Knockknock support
try:
    from secret import TELEGRAM_API_TOKEN, TELEGRAM_CHAT_ID, COUNTRY_CODE

    SECRET_IMPORTED = True

except ImportError:
    pass

# CUDA
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def create_patched_eval(
    iid_data_split: DataLoader,
    ood_data_split: DataLoader,
    tokenizer: PreTrainedTokenizerBase,
):
    """
    Create a modified version of a model's eval function that also tracks uncertainty estimates on the validation
    and OOD test set over time.

    """

    def eval_with_tracking_uncertainties(
        self, data_split: DataLoader, wandb_run: Optional[WandBRun] = None
    ) -> torch.Tensor:
        """
        Evaluate a data split.

        Parameters
        ----------
        data_split: DataSplit
            Data split the model should be evaluated on.
        wandb_run: Optional[WandBRun]
            Weights and Biases run to track training statistics. Training and validation loss (if applicable) are
            tracked by default, everything else is defined in _epoch_iter() and _finetune() depending on the model.

        Returns
        -------
        torch.Tensor
            Loss on evaluation split.
        """
        self.module.eval()
        loss = self._epoch_iter(0, data_split)

        # Also track uncertainty performance and calibration over time
        if wandb_run is not None:
            scores = evaluate_uncertainty(
                self, iid_data_split, ood_data_split, tokenizer
            )
            wandb_run.log(scores)

        self.module.train()

        return loss

    return eval_with_tracking_uncertainties


def run_experiments(
    model_name: List[str],
    dataset_name: str,
    runs: int,
    seed: int,
    device: Device,
    data_dir: str,
    model_dir: str,
    result_dir: str,
    wandb_run: Optional[WandBRun] = None,
) -> str:
    """
    Run experiments. An experiment consists of training evaluating a number of models on a dataset and saving
    the models and model outputs.

    Parameters
    ----------
    model_name: List[str]
        Names of models that experiments should be run for.
    dataset_name: str
        Name of the dataset the model should be run on.
    runs: int
        Number of runs with different random seeds per model.
    seed: int
        Initial seed for every model.
    device: Device
        Device the model is being trained on.
    data_dir: str
        Directory the data is stored in.
    model_dir: str
        Directory that models are being saved to.
    result_dir: str
        Directory where results will be written to.
    wandb_run: Optional[WandBRun]
        Weights and Biases Run to track training statistics. Training and validation loss (if applicable) are tracked by
        default, everything else is defined in _epoch_iter() and _finetune() depending on the model.

    Returns
    -------
    Dict[str, Any]
        Information about experiments that is being sent by knockknock.
    """
    scores = defaultdict(list)

    np.random.seed(seed)
    torch.manual_seed(seed)

    timestamp = str(datetime.now().strftime("%d-%m-%Y_(%H:%M:%S)"))

    # Get model (hyper-)parameters
    model_params = MODEL_PARAMS[dataset_name][model_name]

    # Read data and build data splits
    dataset_builder = AVAILABLE_DATASETS[dataset_name](
        data_dir=data_dir, max_length=model_params["sequence_length"]
    )
    data_splits = dataset_builder.build(
        batch_size=model_params["batch_size"], drop_last=True
    )

    for run in range(runs):

        model = AVAILABLE_MODELS[model_name](
            model_params, model_dir=model_dir, device=device
        )

        # If WandB tracking is enables, patch the object's eval method with the one define at the top of this script
        # that also tracks uncertainty properties over the course of the training
        if wandb_run is not None:
            patched_eval_func = create_patched_eval(
                iid_data_split=data_splits["test"],
                ood_data_split=data_splits["ood_test"],
                tokenizer=dataset_builder.tokenizer,
            )
            model.eval = types.MethodType(patched_eval_func, model)

        result_dict = model.fit(
            train_split=data_splits["train"],
            valid_split=data_splits["valid"],
            wandb_run=wandb_run,
        )
        scores["train_loss"].append(result_dict["train_loss"])
        scores["best_val_loss"].append(result_dict["best_val_loss"])

        # Evaluate task performance
        model.module.eval()

        task_scores = evaluate(
            model,
            eval_split=data_splits["test"],
            tokenizer=dataset_builder.tokenizer,
        )

        for score_name, score in task_scores.items():
            scores[score_name].append(score)

        # Evaluate uncertainty experiments
        uncertainty_scores = evaluate_uncertainty(
            model,
            id_eval_split=data_splits["test"],
            ood_eval_split=data_splits["ood_test"],
            tokenizer=dataset_builder.tokenizer,
            predictions_path=f"{result_dir}/{model_name}_{run + 1}_{timestamp}_uncertainty",
        )

        for score_name, score in uncertainty_scores.items():
            scores[score_name].append(score)

        # Save all scores in pickle file
        with open(
            f"{result_dir}/{model_name}_{timestamp}_scores.pkl", "wb"
        ) as scores_path:
            pickle.dump(scores, scores_path)

        # Add all info to Weights & Biases
        if wandb_run is not None and run < runs - 1:
            wandb_run.finish()
            wandb_run = wandb.init(
                project=PROJECT_NAME, tags=[dataset_name, model_name]
            )

    # # Reset for potential next run
    if wandb_run is not None:
        wandb_run.config.update(model_params)
        wandb_run.log(
            {
                score_name: f"{np.mean(scores):.4f} ±{np.std(scores):.2f}"
                for score_name, scores in scores.items()
            }
        )

    return json.dumps(
        {
            "dataset": dataset_name,
            "runs": runs,
            "scores": {
                score_name: f"{np.mean(scores):.4f} ±{np.std(scores):.2f}"
                for score_name, scores in scores.items()
            },
            "url": wandb.run.get_url(),
        },
        indent=4,
        ensure_ascii=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=AVAILABLE_DATASETS.keys(),
        help="Dataset to run experiments on.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=AVAILABLE_MODELS.keys(),
    )
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--result-dir", type=str, default=RESULT_DIR)
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR)
    parser.add_argument("--emission-dir", type=str, default=EMISSION_DIR)
    parser.add_argument("--track-emissions", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--knock", action="store_true", default=False)
    parser.add_argument("--wandb", action="store_true", default=False)
    args = parser.parse_args()

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    tracker = None
    wandb_run = None

    if args.wandb:
        wandb_run = wandb.init(project=PROJECT_NAME, tags=[args.dataset, args.model])

    if args.track_emissions:
        timestamp = str(datetime.now().strftime("%d-%m-%Y (%H:%M:%S)"))
        emissions_path = os.path.join(args.emission_dir, timestamp)
        os.mkdir(emissions_path)
        tracker = OfflineEmissionsTracker(
            project_name=PROJECT_NAME,
            country_iso_code=COUNTRY_CODE,
            output_dir=emissions_path,
        )
        tracker.start()

    # Apply decorator
    if args.knock:
        if not SECRET_IMPORTED:
            raise ImportError(
                "secret.py wasn't found, please rename secret_template.py and fill in the information."
            )

        run_experiments = telegram_sender(
            token=TELEGRAM_API_TOKEN, chat_id=TELEGRAM_CHAT_ID
        )(run_experiments)

    try:
        run_experiments(
            args.model,
            args.dataset,
            args.runs,
            args.seed,
            args.device,
            args.data_dir,
            args.model_dir,
            args.result_dir,
            wandb_run,
        )

    except Exception as e:
        # Save data from emission tracker in any case
        if tracker is not None:
            tracker.stop()

        raise e
