# Exploring Predictive Uncertainty and Calibration in NLP: A Study on the Impact of Method & Data Scarcity

### *Warning: The documentation below is still work-in-progress, but will be completed soon (I promise!)* 

This repository contains the code of the EMNLP 2022 paper *"Exploring Predictive Uncertainty and Calibration in NLP: A Study on the Impact of Method & Data Scarcity"* by
Dennis Ulmer, Jes Frellsen und Christian Hardmeier (@TODO: Add paper link). 

This repository builds on the code in [``nlp-uncertainty-zoo``](https://github.com/Kaleidophon/nlp-uncertainty-zoo), which is also one of the requirements of this project.
 
If you found this repository useful, please cite our work:

    @TODO: Add citation

## Repository structure

@TODO

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

For experimental tracking via Weights & Biases (@TODO: Link) as well as carbon emission tracking via ``codecarbon``
(@TODO: Link), it is possible to add some settings in a file called ``secret.py``. 

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
