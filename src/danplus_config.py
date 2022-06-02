"""
Define all training and model parameters used for the Dan+ dataset.
"""

# EXT
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import transformers


DANPLUS_MODEL_PARAMS = {
    "lstm": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 0.001357,
        "lr": 0.4931,
        "num_training_steps": 7480,
        "validation_interval": 136,
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "init_weight": 0.5848,  # Hacky way to include this for replication, this prob. won't be used anywhere else
        "optimizer_class": optim.SGD,
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": range(13 * 136, 54 * 136, 136),
        },
        "num_layers": 2,
        "hidden_size": 350,
        "input_size": 650,
        "dropout": 0.223,
        "vocab_size": 52000,
        "output_size": 20,
        "is_sequence_classifier": False,
    },
    "lstm_ensemble": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 0.001357,
        "lr": 0.4931,
        "num_training_steps": 7480,
        "validation_interval": 136,
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "init_weight": 0.5848,  # Hacky way to include this for replication, this prob. won't be used anywhere else
        "optimizer_class": optim.SGD,
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": range(13 * 136, 54 * 136, 136),
        },
        "num_layers": 2,
        "hidden_size": 350,
        "input_size": 650,
        "dropout": 0.223,
        "vocab_size": 52000,
        "output_size": 20,
        "ensemble_size": 10,
        "is_sequence_classifier": False,
    },
    "bayesian_lstm": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 0.003016,
        "lr": 0.1114,
        "num_training_steps": 7480,
        "validation_interval": 136,
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "optimizer_class": optim.SGD,
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": range(13 * 136, 54 * 136, 136),
        },
        "num_layers": 2,
        "hidden_size": 650,
        "input_size": 650,
        "dropout": 0.3868,
        "vocab_size": 52000,
        "output_size": 20,
        "prior_sigma_1": 0.7664,
        "prior_sigma_2": 0.851,
        "prior_pi": 1,
        "posterior_mu_init": -0.0425,
        "posterior_rho_init": -6,
        "num_predictions": 10,
        "is_sequence_classifier": False,
    },
    "st_tau_lstm": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 0.001189,
        "lr": 0.1979,
        "num_training_steps": 7480,
        "validation_interval": 136,
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "optimizer_class": optim.SGD,
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": range(13 * 136, 54 * 136, 136),
        },
        "num_layers": 2,
        "hidden_size": 350,
        "input_size": 650,
        "dropout": 0.1867,
        "vocab_size": 52000,
        "output_size": 20,
        "num_predictions": 10,
        "num_centroids": 30,
        "is_sequence_classifier": False,
    },
    # Taken from  https://github.com/yaringal/BayesianRNN/blob/master/LM_code/main_new_dropout_SOTA.lua
    "variational_lstm": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 1e-7,
        "lr": 0.3031,
        "num_training_steps": 7480,
        "validation_interval": 136,
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "init_weight": 0.1097,  # Hacky way to include this for replication, this prob. won't be used anywhere else
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": range(13 * 136, 54 * 136, 136),
        },
        "num_layers": 2,
        "hidden_size": 350,
        "input_size": 650,
        "embedding_dropout": 0.1207,  # dropout_x, Large model Gal & Ghrahramani (2016)
        "layer_dropout": 0.1594,  # dropout_i / dropout_o, Large model Gal & Ghrahramani (2016)
        "time_dropout": 0.1281,  # dropout_h, Large model Gal & Ghrahramani (2016)
        "vocab_size": 52000,
        "output_size": 20,
        "num_predictions": 10,  # Changed from 1000 because that's just excessive
        "is_sequence_classifier": False,
    },
    "ddu_bert": {
        "bert_name": "alexanderfalk/danbert-small-cased",
        "ignore_indices": [-100, 0, 1, 2, 3],
        "batch_size": 32,
        "sequence_length": 128,
        "lr": 0.00006168,
        "weight_decay": 0.1868,
        "num_training_steps": 2720,
        "validation_interval": 136,
        "grad_clip": 10,
        "optimizer_class": optim.Adam,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_kwargs": {
            # Warmup prob: 0.1
            "num_warmup_steps": 272,
            "num_training_steps": 2720,
        },
        "output_size": 20,
        "is_sequence_classifier": False,
        "projection_size": None,
        "spectral_norm_upper_bound": 0.9211,
    },
    "variational_bert": {
        "bert_name": "alexanderfalk/danbert-small-cased",
        "batch_size": 32,
        "sequence_length": 128,
        "lr": 0.00009742,
        "weight_decay": 0.02731,
        "num_training_steps": 2720,
        "validation_interval": 136,
        "grad_clip": 10,
        "dropout": 0.4362,
        "optimizer_class": optim.Adam,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_step_or_epoch": "epoch",
        "scheduler_kwargs": {
            # Warmup prob: 0.1
            "num_warmup_steps": 272,
            "num_training_steps": 2720,
        },
        "output_size": 20,
        "num_predictions": 10,
        "is_sequence_classifier": False,
    },
    "due_bert": {
        "bert_name": "alexanderfalk/danbert-small-cased",
        "batch_size": 32,
        "sequence_length": 128,
        "lr": 0.05,
        "num_training_steps": 2720,
        "validation_interval": 136,
        "optimizer_class": optim.Adam,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_step_or_epoch": "epoch",
        "scheduler_kwargs": {
            # Warmup prob: 0.1
            "num_warmup_steps": 272,
            "num_training_steps": 2720,
        },
        "output_size": 20,
        "num_predictions": 10,
        "num_inducing_points": 200,
        "num_inducing_samples": 10000,
        "spectral_norm_upper_bound": 0.99,
        "kernel_type": "Matern32",
        "is_sequence_classifier": False,
    },
    "sngp_bert": {
        "bert_name": "alexanderfalk/danbert-small-cased",
        "batch_size": 32,
        "sequence_length": 128,
        "lr": 0.00004815,
        "length_scale": 2,
        "weight_decay": 0.01,
        "num_training_steps": 2720,
        "validation_interval": 136,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_step_or_epoch": "epoch",
        "scheduler_kwargs": {
            # Warmup prob: 0.1
            "num_warmup_steps": 272,
            "num_training_steps": 2720,
        },
        "output_size": 20,
        "last_layer_size": 768,
        "spectral_norm_upper_bound": 0.99,
        "ridge_factor": 0.001,
        "scaling_coefficient": 0.999,
        "beta_length_scale": 1.098,
        "gp_mean_field_factor": 0.2894,
        "num_predictions": 10,
        "is_sequence_classifier": False,
    },
}
