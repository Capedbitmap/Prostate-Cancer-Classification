"""
Configuration file for the Prostate Cancer Classification project.
"""

import os

# Dataset Configuration
DATASET_CONFIG = {
    "gdrive_zip_link": "1Jiz3Ij2NbbhGml2jce31gboJwDDct73P",
    "local_zip_name": "SICAPv2.zip",
    "base_dir": "SICAPv2/SICAPv2",
    "images_folder": "images",
    "masks_folder": "masks",
    "partition_dir": "partition",
}

# Model Configuration
MODEL_CONFIG = {
    "num_classes": 5,
    "num_seg_classes": 4,
    "use_transformer": True,
    "backbone": "efficientnet_b0",
    "feature_dim": 1280,
    "num_attention_heads": 8,
    "dropout_rate": 0.3,
}

# Training Configuration
TRAINING_CONFIG = {
    "epochs": 25,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "specialist_epochs": 15,
    "specialist_batch_size": 8,
    "use_amp": True,  # Automatic Mixed Precision
    "num_workers": 2,
}

# Loss Configuration
LOSS_CONFIG = {
    "alpha": 0.4,  # Multi-label loss weight
    "beta": 0.4,   # Cribriform loss weight
    "gamma": 0.2,  # Segmentation loss weight
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    "low_res_size": (224, 224),
    "high_res_size": (384, 384),
    "horizontal_flip_prob": 0.5,
    "vertical_flip_prob": 0.5,
    "rotation_degrees": 30,
    "color_jitter": {
        "brightness": 0.3,
        "contrast": 0.3,
        "saturation": 0.3,
        "hue": 0.2,
    },
}

# Stain Normalization Configuration
STAIN_NORM_CONFIG = {
    "target_means": [0.65, 0.70, 0.29],
    "target_stds": [0.15, 0.15, 0.10],
    "method": "macenko",
}

# Evaluation Configuration
EVAL_CONFIG = {
    "threshold": 0.5,
    "mask_percentage_threshold": 0.05,
    "cross_validation_folds": 4,
}

# Output Configuration
OUTPUT_CONFIG = {
    "save_dir": "SICAPv2_results",
    "model_checkpoint": "best_multitask_model.pth",
    "specialist_checkpoint": "cribriform_specialist.pth",
    "results_file": "enhanced_results_summary.txt",
    "visualizations": {
        "predictions": "multilabel_predictions.png",
        "misclassifications": "cribriform_misclassifications.png",
    },
}

# Class Labels
CLASS_LABELS = {
    "label_map": {"nc": 0, "g3": 1, "g4": 2, "g4c": 3, "g5": 4},
    "class_names": ["NC", "G3", "G4", "G4C", "G5"],
    "inverse_label_map": {0: "nc", 1: "g3", 2: "g4", 3: "g4c", 4: "g5"},
}

# Device Configuration
DEVICE_CONFIG = {
    "use_cuda": True,
    "cuda_device": 0,
    "fallback_to_cpu": True,
}
