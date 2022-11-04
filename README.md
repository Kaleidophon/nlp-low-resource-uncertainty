# Exploring Predictive Uncertainty and Calibration in NLP: A Study on the Impact of Method & Data Scarcity

### *Warning: The documentation below is still work-in-progress, but will be completed soon (I promise!)* 

This repository contains the code of the EMNLP 2022 paper [*"Exploring Predictive Uncertainty and Calibration in NLP: A Study on the Impact of Method & Data Scarcity"*](https://arxiv.org/abs/2210.15452) by
Dennis Ulmer, Jes Frellsen und Christian Hardmeier. 

This repository builds on the code in [``nlp-uncertainty-zoo``](https://github.com/Kaleidophon/nlp-uncertainty-zoo), which is also one of the requirements of this project.
 
If you found this repository useful, please cite our work:

    @article{ulmer2022exploring,
      title={Exploring Predictive Uncertainty and Calibration in NLP: A Study on the Impact of Method \& Data Scarcity},
      author={Ulmer, Dennis and Frellsen, Jes and Hardmeier, Christian},
      journal={arXiv preprint arXiv:2210.15452},
      year={2022}
    }


## Repository structure

* `data/`: Folder containing all the data used in the paper.
  * `raw`: Folders containing the raw datasets, i.e. `clinc`, `danplus` and `finnish_ud` (see Data section below for information on how to acquire them).
  * `processed`: Processed datasets using the same folder structure as `raw`. Produced by running the processing scripts located in `scripts/`.
* `img/`: Containing all the plots used in the work. 
  * `qualitative/`: Containing plots used for the qualitative analyses, corresponding to `dan+` and `finnish_ud`.
  * `scatter_plots/`: Scatter plots depicting the relationships between calibration and uncertainty quality and task performance.
  * `time/`: Plots that show the development of token / sequence Kendall's tau correlations over the training time.
* `results/`: All the results that are used in the paper.
  * `significance_testing/`: ASO values for different metrics and datasets between the models.
  * `subsampling_verification/`: Perplexity values obtained by the n-gram model for dataset splits, ood split and subsampled training splits.
  * The result of the directory contains files that are of the form `<dataset_name>_<size>_<model_name>_<run>_<timestamp>`. There are four different files per model run:
    * `_uncertainty.csv`: Uncertainty measurements per instance in the test set.
    * `_uncertainty.pkl`: Uncertainty measurements per instance as a pickle file.
    * `_scores.pkl`: Dictionary of all the final model scores.
    * `_stats.csv`: Evaluation scores per instance in the test set.
* `scripts/`
  * `check_subsampling_and_ood.py`: Check the frequencies of labels and types in subsampled / OOD splits. 
  * `get_ngram_ppl.py`: Compute the perplexity values for different (subsampled) splits. This requires a local installation of [SRILM](http://www.speech.sri.com/projects/srilm/).
  * `get_total_runtime_and_emissions.py`: Calculate the total runtime and emissions over model runs.
  * `preprocess_clinc.py`: Preprocess the CLINC (Plus) dataset.
  * `preprocess_danish.py`: Preprocess the DanPlus dataset.
  * `preprocess_finnish.py`: Preprocess the FinnishUD dataset.
* `src/`:
  * `clinc_config.py`: Model configs for CLINC (Plus).
  * `config.py`: General project config. 
  * `danplus_config.py`: Model configs for Dan+.
  * `data.py`: Define `DatasetBuilders` for used datasets.
  * `finnish_ud_config.py`: Model configs for Finnish UD.
  * `uncertainty_evaluation.py`: Define how uncertainty and calibration is being evaluated.
* `sweeps/`: Includes the information used for Weights & Biases hyperparameter sweeps.
* `format_results.py`: Format results in Latex for table 2 in the paper. 
* `hyperparameter_search.py`: Run a hyperparameter search. Called by `sweep.py`.
* `plot_metrics_over_time.py`: Plot the development of uncertainty metrics over time.
* `qualitative_analysis.py`: Produce plots for the qualitative analysis of `DanPlus` and `FinnishUD`.
* `run_experiments.py`: Run an experiment.
* `significance_testing.py`: Perform significance tests.
* `sweep.py`: Use the wandb agent to start a sweep. Calls `hyperparameter_search.py`.
* `visualize_results.py`: Create scatter plots of results, comparing task and uncertainty / calibration performance.

## Replication

First of all, install the repository requirements using ``pip``:
    
    pip3 install -r requirements.txt

Next, we will walk you through the different steps necessary to replicate the findings of the paper.

### Data

In order to replicate our findings, download the Clinc Plus, Dan+ and Finnish UD dataset from the links below:

  * **Clinc Plus**: https://github.com/clinc/oos-eval
  * **Dan+**: https://github.com/bplank/DaNplus
  * **Finnish UD**: 
    * Training, validation and test split: https://github.com/UniversalDependencies/UD_Finnish-TDT
    * OOD test set: https://github.com/UniversalDependencies/UD_Finnish-OOD

Then, place them in the ``data/raw/`` directory, in folders called ``clinc``, ``clinc_plus``, ``danplus`` and ``finnish_ud``.

You can then preprocess the data by running the following commands:

    cd scripts
    python3 process_clinc.py
    python3 process_danplus.py
    python3 process_finnish_ud.py

This should produce files in the `data/processed/` directory and the subfolders with the same name, populating them with 
files called ``train.csv``, ``val.csv``, ``test.csv`` as well as ``oo(d/s)_test.csv``.

### Setting secret.py

For experimental tracking via [Weights & Biases](https://github.com/wandb/wandb) as well as carbon emission tracking via [``codecarbon``](https://github.com/mlco2/codecarbon), it is possible to add some settings in a file called ``secret.py``. 

The actually secret file used for the paper is not added to the repository for security reasons, but you can add your own
information by filling out the fields in ``secret_template.py`` and then renaming it to ``secret.py``. 
Nevertheless, the code is supposed to run even without this step.

### Hyperparameter Search

Hyperparameter search is conducted through [Weights & Biases' sweep functionality](https://wandb.ai/site/sweeps). 
For that purpose, hyperparameter search ranges for every model and dataset can be found in `sweeps/<dataset_name>/sweep_<model_name>_<dataset_name>.yaml`.

All the logic for the hyperparameter search is defined in `hyperparameter.search.py`, which is called through running `sweep.py`.
To run a hyperparameter search, run something similar to the command below:

    python3 sweep.py sweeps/clinc_plus/sweep_lstm_clinc_plus.yaml 30

which will perform a hyperparameter search for the LSTM on Clinc Plus for 30 trials in total.

### Model Training

Model training is performed by running `run_experiments.py`. The most important arguments are the following:
  * `--dataset`: Specify the name of the dataset the model should be trained on. Can be `dan+`, `finnish_ud` or `clinc_plus`.
  * `--model`: Specify the name of the model to be trained. Can be any of the models defined in [``nlp-uncertainty-zoo``](https://github.com/Kaleidophon/nlp-uncertainty-zoo).
  * `--device`: Define the device the model should be trained on for PyTorch, e.g. `cpu` or `cuda`.
  * `--training-size`: Define the size the training set should be subsampled into.
  * `--runs`: Number of runs per model. In the paper, five different seeds per model were trained.

### Validation of Subsets / OOD Test Sets

This part involves two parts: Measuring perplexities of in- and out-of-distribution splits as well as plotting some of their properties.
For the first part, make sure to provide a local installation of [SRILM](http://www.speech.sri.com/projects/srilm/). Then run 

      cd scripts/
      python3 get_ngram_ppl.py

Where results will be saved to `results/subsampling_verification/<dataset_name>_results.txt`. For the second part, run 

      cd scripts
      python3 check_subsampling_and_ood.py

and resulting plots will be saved as `img/{label/seq/token}_freqs_{id_ood/subsampled}.pdf`.

### Significance testing

Significance testing is performed using the Almost Stochastic Order (ASO) test, as implemented in [deep-significance](https://github.com/Kaleidophon/deep-significance).
To execute the significance testing, it is required to have the corresponding `<dataset_name>_<dataset_size>_<model_name>_<run>_<timestamp>_scores.pkl` files in the `results/` folder.
Then, run 

    python3 significance_testing.py --dataset <dataset_name> --training-sizes <dataset_size>

and the corresponding results will be saved into `results/significance_testing/<dataset_name>_significance_testing.txt`.

### Figures

To re-run the hyperparameter search, or to train the different models and regenerate results, please refer to the sections
[Hyperparameter Search](#hyperparameter-search) and [Model Training](#model-training) above. We now list below the figures in the paper, and the necessary scripts and exact arguments to reproduce them:

* **Table 2**: Run `python3 format_results.py` specifying the dataset name and training set size. For the paper, these were
  * `python3 format_results.py --dataset clinc_plus --training-sizes 15000`
  * `python3 format_results.py --dataset finnish_ud --training-sizes 10000`
  * `python3 format_results.py --dataset danplus --training-sizes 4000`
  * Significant results were highlighted manually. To reproduce them, run 
    * `python3 significance_testing.py --dataset clinc_plus --training-sizes 15000`
    * `python3 significance_testing.py --dataset finnish_ud --training-sizes 10000`
    * `python3 significance_testing.py --dataset danplus --training-sizes 4000`
* **Figure 2**: This requires running `visualize_results.py` specifying the dataset name. For the paper, these were
  * `python3 visualize_results.py --dataset clinc_plus`
  * `python3 visualize_results.py --dataset finnish_ud`
  * `python3 visualize_results.py --dataset danplus`
  * Afterwards, results are found in `img/scatter_plots/`.
* **Figure 3**: This script is slightly more involved, since it includes a lot of different parameters. We report the exact parameters below:
  * **Figure 3 (b)**: `python3 plot_metrics_over_time.py --dataset dan+ --models lstm lstm_ensemble ddu_bert --target kendalls_tau_seq --target-name "Sequence-level Kendall's tau" --metrics ood_predictive_entropy --step-cutoff 2500 --identifier selection`
  * **Figure 3 (a)**: `python3 plot_metrics_over_time.py --dataset dan+ --models lstm lstm_ensemble ddu_bert --target kendalls_tau_token --target-name "Token-level Kendall's tau" --metrics ood_predictive_entropy --step-cutoff 2500 --identifier selection`
* **Figure 4**:
  * **Figure 4 (a)**: `python3 qualitative_analysis.py --dataset dan+ --metrics predictive_entropy --training-sizes 4000 --top-n 40 --normalize`
  * **Figure 4 (b)**: `python3 qualitative_analysis.py --dataset finnish_ud --metrics mutual_information predictive_entropy --models variational_lstm sngp_bert lstm_ensemble variational_bert --training-sizes 10000 --top-n 40`
  * Result will be saved in `img/qualitative/<dataset_name>/`. Note that sequences will be randomly sampled from the test set every time.
* **Figures 5 - 10**: These figures are all plotted and saved into `img/` by running the commands below:
  * `cd scripts`
  * `python3 check_subsampling_and_ood.py`
* **Table 3**: These results are saved in `results/subsampling_verification/<dataset_name>_results.txt`. To reproduce them, run
  * `cd scripts/`
  * `python3 get_ngram_ppl.py`
  * Be aware that this requires a functioning installation of [SRILM](http://www.speech.sri.com/projects/srilm/) on your system.
* **Table 4**: Hyperparameter search ranges are defined in `sweeps/<dataset_name>/sweep_<model_name>_<dataset_name>.yaml`.
* **Table 5**: All used hyperparameters can also be found in `src/<dataset_name>_config.py`.
* **Figure 11 - 14**: Same as Figure 2. 
* **Figure 15 + 16**: Same as Figure 3. We report the exact arguments for the script below:
  * **Figure 15 (a)**: `python3 plot_metrics_over_time.py --dataset dan+ --models lstm lstm_ensemble st_tau_lstm variational_lstm bayesian_lstm variational_bert ddu_bert sngp_bert --target kendalls_tau_token --target-name "Token-level Kendall's tau" --metrics ood_predictive_entropy --step-cutoff 2500 --identifier all`
  * **Figure 15 (b)**:`python3 plot_metrics_over_time.py --dataset finnish_ud --models lstm lstm_ensemble st_tau_lstm variational_lstm bayesian_lstm variational_bert ddu_bert sngp_bert --target kendalls_tau_token --target-name "Token-level Kendall's tau" --metrics ood_predictive_entropy --step-cutoff 7000 --identifier all`
  * **Figure 16 (a)**: `python3 plot_metrics_over_time.py --dataset clinc_plus --models lstm lstm_ensemble bayesian_lstm variational_bert ddu_bert --target kendalls_tau_seq --target-name "Sequence-level Kendall's tau" --metrics ood_predictive_entropy --step-cutoff 7000 --identifier all`
  * **Figure 16 (b)**: `python3 plot_metrics_over_time.py --dataset dan+ --models lstm lstm_ensemble st_tau_lstm variational_lstm bayesian_lstm variational_bert ddu_bert sngp_bert --target kendalls_tau_seq --target-name"Sequence-level Kendall's tau" --metrics ood_predictive_entropy --step-cutoff 2500 --identifier all`
  * **Figure 16 (c)**: `python3 plot_metrics_over_time.py --dataset finnish_ud --models lstm lstm_ensemble st_tau_lstm variational_lstm bayesian_lstm variational_bert ddu_bert sngp_bert --target kendalls_tau_seq --target-name "Sequence-level Kendall's tau" --metrics ood_predictive_entropy --step-cutoff 7000 --identifier all`
* **Figure 17 + 18**: Same as Figure 4.
