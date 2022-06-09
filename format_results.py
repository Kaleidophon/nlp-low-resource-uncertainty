"""
Format the results into a pseudo-latex form to make adding them to the paper easier.
"""

# STD
import argparse
from collections import defaultdict
import os
import re
import pickle
from typing import List

# EXT
from nlp_uncertainty_zoo.config import AVAILABLE_MODELS
import numpy as np

# PROJECT
from src.config import AVAILABLE_DATASETS


# CONST
RESULT_DIR = "./results"
OUTPUT_DIR = "./results/significance_testing"
MODEL_DIR = "./models"
TABLE_COLUMNS = [
    "accuracy",
    "macro_f1_scores",
    "ece",
    "sce",
    "ace",
    "coverage_percentage",
    "coverage_width",
    "auroc",
    "aupr",
    "kendalls_tau_token",
    "kendalls_tau_seq",
]
METRICS_TO_LATEX_SYMBOLS = {
    "max_prob": "{\\bigcirc}",
    "predictive_entropy": "{\\bigtriangleup}",
    "variance": "{\Box}",
    "softmax_gap": "{\pentagon}",
    "dempster_shafer": "{\\text{\\tiny \Plus}}",
    "mutual_information": "{\\text{\\tiny \XSolidBold}}",
    "log_prob": "{\BigDiamondshape}",
}


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
        "--models",
        type=str,
        default=AVAILABLE_MODELS.keys(),
        nargs="+",
        choices=AVAILABLE_MODELS.keys(),
    )
    parser.add_argument("--training-sizes", type=int, nargs="+", default=tuple())
    parser.add_argument("--result-dir", type=str, default=RESULT_DIR)
    args = parser.parse_args()

    # Retrieve score files that fit description
    def is_match(path: str, data_set: str, models: List[str]) -> bool:
        if data_set not in path:
            return False

        if not any([f"_{model}_" in path for model in models]):
            return False

        if "_scores.pkl" not in path:
            return False

        return True

    result_paths = os.listdir(args.result_dir)
    result_paths = list(
        filter(lambda path: is_match(path, args.dataset, args.models), result_paths)
    )

    # Read in data
    # Map from model to training size to ID / OOD to metric
    data = defaultdict(lambda: defaultdict(list))
    found_metrics = set()

    max_run_per_model = defaultdict(int)

    for result_path in result_paths:
        _, training_size, model_name, run = (
            re.compile(r"(.+?)_(\d+)_(.+)_(\d)_\d{2}-\d{2}-\d{4}")
            .match(result_path)
            .groups()
        )
        run = int(run)

        if run < max_run_per_model[model_name]:
            continue

        max_run_per_model[model_name] = run

        training_size = int(training_size)

        if len(args.training_sizes) > 0 and training_size not in args.training_sizes:
            continue

        with open(f"{args.result_dir}/{result_path}", "rb") as result_file:
            scores = pickle.load(result_file)

            for name, score in scores.items():

                if name in ("train_loss", "_timestamp", "_runtime"):
                    continue

                # For auroc_{metric} or aupr_{metric}
                if name.startswith("auroc") or name.startswith("aupr"):
                    parts = name.split("_")
                    eval_metric, uncertainty_metric = parts[0], "_".join(parts[1:])
                    found_metrics.add(eval_metric)
                    data[model_name][f"{eval_metric}_{uncertainty_metric}"] = score

                elif "kendalls_tau" in name:
                    split, uncertainty_metric, level = (
                        re.compile(r"([ioo]+d)_(\w+?)_kendalls_tau_(\w+)")
                        .match(name)
                        .groups()
                    )
                    eval_metric = f"{split}_kendalls_tau_{level}"
                    found_metrics.add(eval_metric)
                    data[model_name][f"{eval_metric}_{uncertainty_metric}"] = score

                else:
                    found_metrics.add(name)
                    data[model_name][name] = score

    def format_res(num: float):
        return f"{round(num, 2):.2f}".replace("0.", ".")

    for model in data:

        row_string = f"{model} "

        for column in TABLE_COLUMNS:

            if column in ["auroc", "aupr"]:
                all_scores = {
                    name: scores
                    for name, scores in data[model].items()
                    if column in name
                }
                max_metric_key = max(
                    all_scores, key=lambda key: np.mean(all_scores.get(key))
                )
                max_metric = "_".join(max_metric_key.split("_")[1:])
                scores = all_scores[max_metric_key]
                mean_scores = format_res(np.mean(scores))
                std_scores = format_res(np.std(scores))
                metric_marker = f"^{METRICS_TO_LATEX_SYMBOLS[max_metric]}"

                row_string += (
                    f"& $\\srs{{{mean_scores}{metric_marker}}}{{{std_scores}}}$ "
                )

            else:
                id_metric_marker = ""
                ood_metric_marker = ""

                if column in ["accuracy", "macro_f1_score"]:
                    id_scores = data[model][column]
                    ood_scores = data[model][f"ood_{column}"]

                elif column in [
                    "ece",
                    "sce",
                    "ace",
                    "coverage_percentage",
                    "coverage_width",
                ]:
                    id_scores = data[model][f"id_{column}"]
                    ood_scores = data[model][f"ood_{column}"]

                # Metric-specific results: Get the best performing one
                elif "kendalls_tau" in column:

                    if "token" in column and args.dataset == "clinc_plus":
                        row_string += "& \\backslashbox[15mm]{}{} "
                        continue

                    all_id_scores = {
                        name: scores
                        for name, scores in data[model].items()
                        if f"id_{column}" in name
                    }
                    max_id_metric_key = max(
                        all_id_scores, key=lambda key: np.mean(all_id_scores.get(key))
                    )
                    id_scores = all_id_scores[max_id_metric_key]
                    max_id_metric = max_id_metric_key.replace("id_", "").replace(
                        f"{column}_", ""
                    )
                    id_metric_marker = f"^{METRICS_TO_LATEX_SYMBOLS[max_id_metric]}"

                    id_scores = all_id_scores[max_id_metric_key]
                    all_ood_scores = {
                        name: scores
                        for name, scores in data[model].items()
                        if f"ood_{column}" in name
                    }
                    max_ood_metric_key = max(
                        all_ood_scores, key=lambda key: np.mean(all_ood_scores.get(key))
                    )
                    ood_scores = all_ood_scores[max_ood_metric_key]
                    max_ood_metric = max_ood_metric_key.replace("ood_", "").replace(
                        f"{column}_", ""
                    )
                    ood_metric_marker = f"^{METRICS_TO_LATEX_SYMBOLS[max_ood_metric]}"

                mean_id = format_res(np.mean(id_scores))
                std_id = format_res(np.std(id_scores))
                mean_ood = format_res(np.mean(ood_scores))
                std_ood = format_res(np.std(ood_scores))

                row_string += (
                    f"& $\\rs{{{mean_id}{id_metric_marker}}}{{{std_id}}}"
                    f"{{{mean_ood}{ood_metric_marker}}}{{{std_ood}}}$ "
                )

        row_string += " \\\\"
        print(row_string)
