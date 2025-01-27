import os

CONFIG = {
    "training": {
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "loss_function": "categorical_crossentropy"
    },
    "data": {
        "dataset_path": '/kaggle/input/lung-tumor-full-nod/dataset_lung.xlsx',
        "train_path": '/kaggle/input/lung-tumor-full-nod/train',
        "train_split": 0.8,
        "validation_split": 0.1,
        "test_split": 0.1,
        "shuffle": True,
        "random_seed": 42,
    },

    "model": {
        "input_shape": (128, 128, 1),
        "num_classes": 2,
        "architecture": "custom",  # 'custom' o 'pretrained'
        "pretrained_model": "ResNet50",  # Ignorato se 'architecture' è 'custom'
    },
    "logging": {
        "log_file": "training.log",
        "log_level": "INFO",
    },
    "output": {
        "save_model_path": "models",
        "save_history_path": "./results/training_history.json",
    },
    "general":{
        "seed": 42
    }
}
