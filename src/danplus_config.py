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
        "weight_decay": 0,
        "lr": 0.3031,
        "num_epochs": 60,  # Changed from 55 in original
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "init_weight": 0.1097,  # Hacky way to include this for replication, this prob. won't be used anywhere else
        "optimizer_class": optim.SGD,
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_step_or_epoch": "epoch",
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": torch.LongTensor(range(13, 54, 1)),
        },
        "num_layers": 2,
        "hidden_size": 350,
        "input_size": 650,
        "dropout": 0.275,
        "vocab_size": 52000,
        "output_size": 20,
        "is_sequence_classifier": False,
    },
    "lstm_ensemble": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 0,
        "lr": 0.3031,
        "num_epochs": 60,  # Changed from 55 in original
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "init_weight": 0.1097,  # Hacky way to include this for replication, this prob. won't be used anywhere else
        "optimizer_class": optim.SGD,
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_step_or_epoch": "epoch",
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": torch.LongTensor(range(13, 54, 1)),
        },
        "num_layers": 2,
        "hidden_size": 350,
        "input_size": 650,
        "dropout": 0.275,
        "vocab_size": 52000,
        "output_size": 20,
        "ensemble_size": 10,
        "is_sequence_classifier": False,
    },
    "bayesian_lstm": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 0,
        "lr": 0.3031,
        "num_epochs": 60,  # Changed from 55 in original
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "optimizer_class": optim.SGD,
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_step_or_epoch": "epoch",
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": torch.LongTensor(range(13, 54, 1)),
        },
        "num_layers": 2,
        "hidden_size": 650,
        "input_size": 650,
        "dropout": 0.275,
        "vocab_size": 52000,
        "output_size": 20,
        "prior_sigma_1": 0.1,
        "prior_sigma_2": 0.002,
        "prior_pi": 1,
        "posterior_mu_init": 0,
        "posterior_rho_init": -6.0,
        "num_predictions": 10,
        "is_sequence_classifier": False,
    },
    "st_tau_lstm": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 0,
        "lr": 0.3031,
        "num_epochs": 60,  # Changed from 55 in original
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "optimizer_class": optim.SGD,
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_step_or_epoch": "epoch",
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": torch.LongTensor(range(13, 54, 1)),
        },
        "num_layers": 2,
        "hidden_size": 350,
        "input_size": 650,
        "dropout": 0.275,
        "vocab_size": 52000,
        "output_size": 20,
        "num_predictions": 10,
        "num_centroids": 20,
        "is_sequence_classifier": False,
    },
    # Taken from  https://github.com/yaringal/BayesianRNN/blob/master/LM_code/main_new_dropout_SOTA.lua
    "variational_lstm": {
        "batch_size": 32,
        "sequence_length": 35,
        "early_stopping": True,
        "weight_decay": 1e-7,
        "lr": 0.3031,
        "num_epochs": 60,
        # "early_stopping_pat": 10,
        "grad_clip": 10,
        "init_weight": 0.1097,  # Hacky way to include this for replication, this prob. won't be used anywhere else
        "scheduler_class": scheduler.MultiStepLR,
        "scheduler_step_or_epoch": "epoch",
        "scheduler_kwargs": {
            "gamma": 0.8695,  # 1 / 1.15; in the Zaremba implementation you divide by gamma,
            "milestones": torch.LongTensor(range(13, 54, 1)),
        },
        "num_layers": 2,
        "hidden_size": 350,
        "input_size": 650,
        "embedding_dropout": 0.3,  # dropout_x, Large model Gal & Ghrahramani (2016)
        "layer_dropout": 0.2645,  # dropout_i / dropout_o, Large model Gal & Ghrahramani (2016)
        "time_dropout": 0.2923,  # dropout_h, Large model Gal & Ghrahramani (2016)
        "vocab_size": 52000,
        "output_size": 20,
        "num_predictions": 10,  # Changed from 1000 because that's just excessive
        "is_sequence_classifier": False,
    },
    "ddu_bert": {
        "bert_name": "alexanderfalk/danbert-small-cased",
        "batch_size": 32,
        "sequence_length": 128,
        "lr": 0.05,
        "num_epochs": 20,
        "grad_clip": 10,
        "optimizer_class": optim.Adam,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_step_or_epoch": "epoch",
        "scheduler_kwargs": {
            # Warmup prob: 0.1
            "num_warmup_steps": 20 * 0.1,
            "num_training_steps": 20,
        },
        "output_size": 20,
        "is_sequence_classifier": False,
        "spectral_norm_upper_bound": 0.99,
    },
    "variational_bert": {
        "bert_name": "alexanderfalk/danbert-small-cased",
        "batch_size": 32,
        "sequence_length": 128,
        "lr": 0.05,
        "num_epochs": 20,
        "grad_clip": 10,
        "dropout": 0.2,
        "optimizer_class": optim.Adam,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_step_or_epoch": "epoch",
        "scheduler_kwargs": {
            # Warmup prob: 0.1
            "num_warmup_steps": 20 * 0.1,
            "num_training_steps": 20,
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
        "num_epochs": 20,
        "optimizer_class": optim.Adam,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_step_or_epoch": "epoch",
        "scheduler_kwargs": {
            # Warmup prob: 0.1
            "num_warmup_steps": 20 * 0.1,
            "num_training_steps": 20,
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
        "lr": 5e-3,
        "length_scale": 2,
        "weight_decay": 0.1,
        "num_epochs": 20,
        "scheduler_class": transformers.get_linear_schedule_with_warmup,
        "scheduler_step_or_epoch": "epoch",
        "scheduler_kwargs": {
            # Warmup prob: 0.1
            "num_warmup_steps": 20 * 0.1,
            "num_training_steps": 20,
        },
        "output_size": 20,
        "last_layer_size": 768,
        "spectral_norm_upper_bound": 0.99,
        "ridge_factor": 0.001,
        "scaling_coefficient": 0.999,
        "beta_length_scale": 2,
        "gp_mean_field_factor": 0.1,
        "num_predictions": 10,
        "is_sequence_classifier": False,
    },
}
