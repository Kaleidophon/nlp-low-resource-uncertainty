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

In order to replicate our findings, download the Clinc Plus, Dan+ and Finnish UD dataset (@TODO: Add links), 
and place them in the ``data/raw/`` directory, in folders called ``clinc``, ``clinc_plus``, ``danplus`` and ``finnish_ud``.

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

@TODO

### Model Training

@TODO

### Validation of Subsets / OOD Test Sets

@TODO


### Significance testing

@TODO

### Figures

We list below the figures in the paper, and the necessary scripts to reproduce them:

* *Table 2*: Run `python3 format_results.py` specifying the dataset name and training set size. For the paper, these were
    * `python3 format_results.py --dataset clinc_plus --training-sizes 15000`
    * `python3 format_results.py --dataset finnish_ud --training-sizes 10000`
    * `python3 format_results.py --dataset danplus --training-sizes 4000`
* *Figure 2*:
* *Figure 3*:
* *Figure 4*:
* *Figures 5 - 10*:
* *Table 3*:  
* *Figure 12-15*: 
* *Figure 16-17*: 
* *Figure 18-19*: 
