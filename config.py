

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
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "adamW",
        "loss_function": "binary_crossentropy"
    },
    "data": {
        "dataset_path": KAGGLE_DATA_PATH,
        "train_path": KAGGLE_TRAIN_PATH,
        "train_split": 0.8,
        "validation_split": 0.1,
        "test_split": 0.1,
        "shuffle": True,
        "class_weights": [0.65990028, 2.06347439] # used only if categorical_focal_crossentropy is the loss
    },
    "preprocessing": {
        "resize": (224, 224),
        "normalization_type": "None",
        "clipping_range": (-1000, 3000),
        "median_filter_size": 3,
        "clahe_clip_limit": 5.0,
        "clahe_tile_grid_size": (5, 5),
        "morph_kernel_size": 3,
        "pipeline":{
            "cropping": True,
            "median_filtering": True,
            "clahe": True,
            "he": False,
            "opening": False,
            "closing": False,
            "convert_to_rgb": True
        }
    },
    "model": {
        "input_shape": (224, 224, 1),
        "num_classes": 2,
        "backbone": "convnext_small",
        "preprocess_input": False
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
    "info":"only_median_filtering_and_cropping"
}




