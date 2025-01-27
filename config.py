import os


LOCAL_DATA_PATH = "Data/dataset_lung.xlsx"
LOCAL_TRAIN_PATH = "Data/Train"

KAGGLE_DATA_PATH = "/kaggle/input/lung-tumor-full-nod/dataset_lung.xlsx"
KAGGLE_TRAIN_PATH = "/kaggle/input/lung-tumor-full-nod/Data/Train"


CONFIG = {
    "training": {
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "loss_function": "categorical_crossentropy"
    },
    "data": {
        "dataset_path": LOCAL_DATA_PATH,
        "train_path": LOCAL_TRAIN_PATH,
        "train_split": 0.8,
        "validation_split": 0.1,
        "test_split": 0.1,
        "shuffle": True,
    },
    "preprocessing": {
        "resize": (128, 128),
        "normalization_type": "z-score"
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
        "save_path": "results",
    },

    "general":{
        "seed": 42
    }
}


## COSE DA CAMBIARE PER RUNNARE IL CODICE IN LOCALE
# dataset_path e train_path