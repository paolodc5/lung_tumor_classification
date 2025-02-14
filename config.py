

# ENV = detect_environment()

# Configurazione automatica dei percorsi
# if ENV == "kaggle":
#     DATA_PATH = "/kaggle/input/lung-tumor-full-nod/dataset_lung.xlsx"
#     TRAIN_PATH = "/kaggle/input/lung-tumor-full-nod/Data/Train"
# else:
#     DATA_PATH = "Data/dataset_lung.xlsx"
#     TRAIN_PATH = "Data/Train"




LOCAL_DATA_PATH = "Data/dataset_lung.xlsx"
LOCAL_TRAIN_PATH = "Data/Train"

KAGGLE_DATA_PATH = "/kaggle/input/lung-tumor-full-nod/dataset_lung.xlsx"
KAGGLE_TRAIN_PATH = "/kaggle/input/lung-tumor-full-nod/Data/Train"

CONFIG = {
    "training": {
        "epochs": 150,
        "batch_size": 32,
        "learning_rate": 0.0001,
        "optimizer": "adamW",
        "loss_function": "binary_crossentropy",
        "augmentation": True,
        "callbacks":{
            "early_stopping": {
                "patience": 20,
                "monitor": "val_loss",
                "mode": "min",
            },
            "reduce_lr": {
                "patience": 20,
                "monitor": "val_loss",
                "factor": 0.1,
                "min_lr": 1e-6,
            }
        }
    },
    "data": {
        "dataset_path": KAGGLE_DATA_PATH,
        "train_path": KAGGLE_TRAIN_PATH,
        "train_split": 0.7,
        "validation_split": 0.1,
        "test_split": 0.2,
        "shuffle": True,
        "class_weights": [0.6,2],
    },
    "preprocessing": {
        "resize": (224, 224),
        "normalization_type": "None",
        "clipping_range": (-1000, 3000),
        "median_filter_size": 3,
        "clahe_clip_limit": 7.0,
        "clahe_tile_grid_size": (20, 20),
        "morph_kernel_size": 2,
        "pipeline":{
            "cropping": True,
            "median_filtering": True,
            "he": False,
            "clahe": True,
            "opening": False,
            "closing": False,
            "convert_to_rgb": True
        }
    },
    "model": {
        "input_shape": (224, 224, 3),
        "output_shape": 1,
        "backbone": "convnext_small",
        "preprocess_input": True,
        "type": "custom",
    },
    "logging": {
        "log_file": "training.txt",
        "log_level": "INFO",
    },
    "output": {
        "save_model_path": "models",
        "save_path": "results",
    },
    "general":{
        "seed": 42
    },
}





