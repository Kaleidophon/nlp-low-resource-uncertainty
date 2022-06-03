"""
Define all training and model parameters used for the Dan+ dataset.
"""

# EXT
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import transformers


FINNISH_UD_MODEL_PARAMS = {
    "lstm": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 0.001337,
        "lr": 0.2205,
        "num_training_steps": 20955,  # Changed from 55 in original
        "validation_interval": 381,
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "init_weight": 0.5848,  # Hacky way to include this for replication, this prob. won't be used anywhere else
        "optimizer_class": optim.SGD,
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": range(13 * 381, 54 * 381, 381),
        },
        "num_layers": 2,
        "hidden_size": 350,
        "input_size": 650,
        "dropout": 0.1392,
        "vocab_size": 50105,
        "output_size": 16,
        "is_sequence_classifier": False,
    },
    "lstm_ensemble": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 0.001337,
        "lr": 0.2205,
        "num_training_steps": 20955,  # Changed from 55 in original
        "validation_interval": 381,
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "init_weight": 0.5848,  # Hacky way to include this for replication, this prob. won't be used anywhere else
        "optimizer_class": optim.SGD,
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": range(13 * 381, 54 * 381, 381),
        },
        "num_layers": 2,
        "hidden_size": 350,
        "input_size": 650,
        "dropout": 0.1392,
        "vocab_size": 50105,
        "output_size": 16,
        "ensemble_size": 10,
        "is_sequence_classifier": False,
    },
    "bayesian_lstm": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 0.0332,
        "lr": 0.1161,
        "num_training_steps": 20955,  # Changed from 55 in original
        "validation_interval": 381,
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "optimizer_class": optim.SGD,
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": range(13 * 381, 54 * 381, 381),
        },
        "num_layers": 2,
        "hidden_size": 650,
        "input_size": 650,
        "dropout": 0.2527,
        "vocab_size": 50105,
        "output_size": 16,
        "prior_sigma_1": 0.9794,
        "prior_sigma_2": 0.7785,
        "prior_pi": 1,
        "posterior_mu_init": -0.1655,
        "posterior_rho_init": -7.0,
        "num_predictions": 10,
        "is_sequence_classifier": False,
    },
    "st_tau_lstm": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 0.003096,
        "lr": 0.01592,
        "num_training_steps": 20955,  # Changed from 55 in original
        "validation_interval": 381,
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "optimizer_class": optim.SGD,
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": range(13 * 381, 54 * 381, 381),
        },
        "num_layers": 2,
        "hidden_size": 350,
        "input_size": 650,
        "dropout": 0.3942,
        "vocab_size": 50105,
        "output_size": 16,
        "num_predictions": 10,
        "num_centroids": 5,
        "is_sequence_classifier": False,
    },
    # Taken from  https://github.com/yaringal/BayesianRNN/blob/master/LM_code/main_new_dropout_SOTA.lua
    "variational_lstm": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 0.01953,
        "lr": 0.7817,
        "num_training_steps": 20955,  # Changed from 55 in original
        "validation_interval": 381,
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "init_weight": 0.1097,  # Hacky way to include this for replication, this prob. won't be used anywhere else
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": range(13 * 381, 54 * 381, 381),
        },
        "num_layers": 2,
        "hidden_size": 350,
        "input_size": 650,
        "embedding_dropout": 0.1519,  # dropout_x, Large model Gal & Ghrahramani (2016)
        "layer_dropout": 0.3923,  # dropout_i / dropout_o, Large model Gal & Ghrahramani (2016)
        "time_dropout": 0.1646,  # dropout_h, Large model Gal & Ghrahramani (2016)
        "vocab_size": 50105,
        "output_size": 16,
        "num_predictions": 10,  # Changed from 1000 because that's just excessive
        "is_sequence_classifier": False,
    },
    "ddu_bert": {
        "bert_name": "TurkuNLP/bert-base-finnish-cased-v1",
        "ignore_indices": [-100, 0, 102, 103, 104],
        "batch_size": 64,
        "sequence_length": 128,
        "lr": 0.001825,
        "num_training_steps": 7620,
        "validation_interval": 381,
        "weight_decay": 0.09439,
        "grad_clip": 10,
        "optimizer_class": optim.Adam,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_kwargs": {
            # Warmup prob: 0.1
            "num_warmup_steps": 762,
            "num_training_steps": 7620,
        },
        "output_size": 16,
        "is_sequence_classifier": False,
        "projection_size": None,
        "spectral_norm_upper_bound": 0.941,
    },
    "variational_bert": {
        "bert_name": "TurkuNLP/bert-base-finnish-cased-v1",
        "batch_size": 64,
        "sequence_length": 128,
        "lr": 0.00003483,
        "num_training_steps": 7620,
        "validation_interval": 381,
        "grad_clip": 10,
        "dropout": 0.4364,
        "weight_decay": 0.09927,
        "optimizer_class": optim.Adam,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_kwargs": {
            # Warmup prob: 0.1
            "num_warmup_steps": 762,
            "num_training_steps": 7620,
        },
        "output_size": 16,
        "num_predictions": 10,
        "is_sequence_classifier": False,
    },
    "due_bert": {
        "bert_name": "TurkuNLP/bert-base-finnish-cased-v1",
        "batch_size": 64,
        "sequence_length": 128,
        "lr": 0.05,
        "num_training_steps": 7620,
        "validation_interval": 381,
        "optimizer_class": optim.Adam,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_kwargs": {
            # Warmup prob: 0.1
            "num_warmup_steps": 762,
            "num_training_steps": 7620,
        },
        "output_size": 16,
        "num_predictions": 10,
        "num_inducing_points": 200,
        "num_inducing_samples": 10000,
        "spectral_norm_upper_bound": 0.99,
        "kernel_type": "Matern32",
        "is_sequence_classifier": False,
    },
    "sngp_bert": {
        "bert_name": "TurkuNLP/bert-base-finnish-cased-v1",
        "batch_size": 32,
        "sequence_length": 128,
        "lr": 5e-5 * 32 / 128,
        "weight_decay": 0.01,
        "num_training_steps": 7620,
        "validation_interval": 381,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_kwargs": {
            # Warmup prob: 0.1
            "num_warmup_steps": 762,
            "num_training_steps": 7620,
        },
        "output_size": 16,
        "last_layer_size": 768,
        "spectral_norm_upper_bound": 0.96,
        "ridge_factor": 0.001,
        "scaling_coefficient": 0.999,
        "beta_length_scale": 2,
        "gp_mean_field_factor": 0.1,
        "num_predictions": 10,
        "is_sequence_classifier": False,
    },
}
