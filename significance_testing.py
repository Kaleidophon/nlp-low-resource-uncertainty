"""
Perform significance testing on obtained results.
"""

# STD
import argparse
from collections import defaultdict
import pickle
import os
import psutil
import re
from typing import List

# EXT
from deepsig import multi_aso
from nlp_uncertainty_zoo.config import AVAILABLE_MODELS
import numpy as np
from tqdm import tqdm

# PROJECT
from src.config import AVAILABLE_DATASETS

# CONST
RESULT_DIR = "./results"
OUTPUT_DIR = "./results/significance_testing"
MODEL_DIR = "./models"


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
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

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

                if name == "train_loss":
                    continue

                # For auroc_{metric} or aupr_{metric}
                if name.startswith("auroc") or name.startswith("aupr"):
                    parts = name.split("_")
                    eval_metric, uncertainty_metric = parts[0], "_".join(parts[1:])
                    found_metrics.add(eval_metric)
                    data[eval_metric][
                        f"{model_name}_{training_size}_{uncertainty_metric}"
                    ] = score

                elif "kendalls_tau" in name:
                    split, uncertainty_metric, level = (
                        re.compile(r"([ioo]+d)_(\w+?)_kendalls_tau_(\w+)")
                        .match(name)
                        .groups()
                    )
                    eval_metric = f"{split}_kendalls_tau_{level}"
                    found_metrics.add(eval_metric)
                    data[eval_metric][
                        f"{model_name}_{training_size}_{uncertainty_metric}"
                    ] = score

                else:
                    found_metrics.add(name)
                    data[name][f"{model_name}_{training_size}"] = score

    # For some of the evaluation metrics for which we have multiple measurements for every uncertainty metrics, take
    # the best result per model
    multi_metric_results = [
        "auroc",
        "aupr",
        "id_kendalls_tau_max_seq",
        "id_kendalls_tau_seq",
        "ood_kendalls_tau_max_seq",
        "ood_kendalls_tau_seq",
    ]

    if args.dataset != "clinc_plus":
        multi_metric_results += ["id_kendalls_tau_token", "ood_kendalls_tau_token"]

    for evaluation_metric in multi_metric_results:
        metric_data = data[evaluation_metric]
        best_scores_by_model, best_uncertainty_metric_by_model = (
            defaultdict(lambda: [-100]),
            {},
        )

        # Only keep the best results
        for identifier, scores in metric_data.items():
            model_name, training_size, metric_name = (
                re.compile(r"(.+?)_(\d+)_(.+)").match(identifier).groups()
            )

            if np.mean(scores) > np.mean(best_scores_by_model[model_name]):
                best_scores_by_model[model_name] = scores
                best_uncertainty_metric_by_model[model_name] = metric_name

        new_metric_data = {
            f"{model_name}_{training_size}_{best_uncertainty_metric_by_model[model_name]}": scores
            for model_name, scores in best_scores_by_model.items()
        }
        data[evaluation_metric] = new_metric_data

    # Flip values for metrics that are lower to be better - ASO expects better model scores to be higher
    for metric in [
        "id_ece",
        "id_sce",
        "id_ace",
        "ood_ece",
        "ood_sce",
        "ood_ace",
        "id_coverage_width",
        "ood_coverage_width",
    ]:
        for model, scores in data[metric].items():
            data[metric][model] = -1 * np.array(scores)

    # Write stdout to file
    num_jobs = psutil.cpu_count(logical=True)

    # Filter metrics
    found_metrics = list(
        filter(lambda metric: not metric.startswith("_"), found_metrics)
    )

    if args.dataset == "clinc_plus":
        found_metrics = list(
            filter(lambda metric: not metric.startswith("ood"), found_metrics)
        )

    with open(
        f"{args.output_dir}/{args.dataset}_significance_testing.txt", "w"
    ) as out_file:
        for metric in tqdm(found_metrics):
            out_file.write(metric + "\n")
            result_df = multi_aso(
                dict(data[metric]),
                return_df=True,
                num_jobs=num_jobs,
                show_progress=False,
            )
            out_file.write(result_df.to_string() + "\n\n")
