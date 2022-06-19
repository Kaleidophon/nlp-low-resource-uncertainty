"""
Define all training and model parameters used for the CLINC dataset.
"""

# EXT
import torch
import torch.optim.lr_scheduler as scheduler
import torch.optim as optim
import transformers

CLINC_MODEL_PARAMS = {
    "lstm": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 0.001337,
        "lr": 0.4712,
        "num_training_steps": 23375,
        "validation_interval": 425,
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "init_weight": 0.283,  # Hacky way to include this for replication, this prob. won't be used anywhere else
        "optimizer_class": optim.SGD,
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": range(13 * 425, 54 * 425, 425),
        },
        "num_layers": 3,
        "hidden_size": 650,
        "input_size": 650,
        "dropout": 0.3379,
        "vocab_size": 30522,
        "output_size": 151,
        "is_sequence_classifier": True,
    },
    "lstm_ensemble": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 0.001337,
        "lr": 0.4712,
        "num_training_steps": 23375,
        "validation_interval": 425,
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "init_weight": 0.283,  # Hacky way to include this for replication, this prob. won't be used anywhere else
        "optimizer_class": optim.SGD,
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": range(13 * 425, 54 * 425, 425),
        },
        "num_layers": 3,
        "hidden_size": 650,
        "input_size": 650,
        "dropout": 0.3379,
        "vocab_size": 30522,
        "output_size": 151,
        "ensemble_size": 5,
        "is_sequence_classifier": True,
    },
    "bayesian_lstm": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 0.001337,
        "lr": 0.1704,
        "num_training_steps": 23375,
        "validation_interval": 425,
        # "early_stopping_pat": 10,
        "grad_clip": 5,
        "optimizer_class": optim.SGD,
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": range(7 * 425, 20 * 425, 425),
        },
        "num_layers": 2,
        "hidden_size": 650,
        "input_size": 650,
        "dropout": 0.3379,
        "vocab_size": 30522,
        "output_size": 151,
        "prior_sigma_1": 0.9851,
        "prior_sigma_2": 0.5302,
        "prior_pi": 1,
        "posterior_mu_init": -0.005537,
        "posterior_rho_init": -7,
        "num_predictions": 10,
        "is_sequence_classifier": True,
    },
    "st_tau_lstm": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 0.003967,
        "lr": 0.0003063,
        "num_training_steps": 23375,
        "validation_interval": 425,
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "optimizer_class": optim.Adam,
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": range(13 * 425, 54 * 425, 425),
        },
        "num_layers": 2,
        "hidden_size": 650,
        "input_size": 650,
        "dropout": 0.1562,
        "vocab_size": 30522,
        "output_size": 151,
        "num_predictions": 10,
        "num_centroids": 2,
        "is_sequence_classifier": True,
    },
    # Taken from  https://github.com/yaringal/BayesianRNN/blob/master/LM_code/main_new_dropout_SOTA.lua
    "variational_lstm": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 0.0009555,
        "lr": 0.0021297,
        "num_training_steps": 23375,
        "validation_interval": 425,
        "grad_clip": 10,
        "init_weight": 0.25,
        "optimizer_class": optim.Adam,
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": range(13 * 425, 54 * 425, 425),
        },
        "num_layers": 2,
        "hidden_size": 650,
        "input_size": 650,
        "embedding_dropout": 0.1265,  # dropout_x, Large model Gal & Ghrahramani (2016)
        "layer_dropout": 0.1655,  # dropout_i / dropout_o, Large model Gal & Ghrahramani (2016)
        "time_dropout": 0.1758,  # dropout_h, Large model Gal & Ghrahramani (2016)
        "vocab_size": 30522,
        "output_size": 151,
        "num_predictions": 10,  # Changed from 1000 because that's just excessive
        "is_sequence_classifier": True,
    },
    "variational_bert": {
        "bert_name": "bert-base-uncased",
        "batch_size": 32,
        "sequence_length": 128,
        "weight_decay": 0.01591,
        "lr": 0.0002981,
        "num_training_steps": 8500,
        "validation_interval": 425,
        "grad_clip": 10,
        "dropout": 0.2382,
        "optimizer_class": optim.Adam,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_kwargs": {
            # Warmup prob: 0.1
            "num_warmup_steps": 850,
            "num_training_steps": 8500,
        },
        "output_size": 151,
        "num_predictions": 10,
        "is_sequence_classifier": True,
    },
    "sngp_bert": {
        "bert_name": "bert-base-uncased",
        "batch_size": 32,
        "sequence_length": 128,
        "lr": 0.00002112,
        "weight_decay_beta": 0.006236,
        "weight_decay": 0,
        "num_training_steps": 8500,
        "validation_interval": 425,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_kwargs": {
            # Warmup prob: 0.1
            "num_warmup_steps": 850,
            "num_training_steps": 8500,
        },
        "output_size": 151,
        "spectral_norm_upper_bound": 0.9175,
        "ridge_factor": 0.001,
        "scaling_coefficient": 0.999,
        "beta_length_scale": 1.501,
        "kernel_amplitude": 0.01851,
        "num_predictions": 10,
        "is_sequence_classifier": True,
    },
    "due_bert": {
        "bert_name": "bert-base-uncased",
        "batch_size": 32,
        "lr": 5e-3,
        "num_training_steps": 8500,
        "validation_interval": 425,
        "optimizer_class": optim.Adam,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_kwargs": {
            # Warmup prob: 0.1
            "num_warmup_steps": 850,
            "num_training_steps": 8500,
        },
        "output_size": 151,
        "num_predictions": 10,
        "num_inducing_points": 20,
        "num_inducing_samples": 10000,
        "spectral_norm_upper_bound": 0.9175,
        "kernel_type": "Matern32",
        "is_sequence_classifier": True,
    },
    "ddu_bert": {
        "bert_name": "bert-base-uncased",
        "ignore_indices": [-100, 0, 101, 102, 103],
        "batch_size": 32,
        "sequence_length": 128,
        "lr": 0.003077,
        "num_training_steps": 8500,
        "validation_interval": 425,
        "grad_clip": 10,
        "optimizer_class": optim.Adam,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_kwargs": {
            # Warmup prob: 0.1
            "num_warmup_steps": 850,
            "num_training_steps": 8500,
        },
        "output_size": 28996,
        "projection_size": 64,
        "is_sequence_classifier": True,
        "spectral_norm_upper_bound": 0.9753,
    },
}
