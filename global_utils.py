import numpy as np
import random
import tensorflow as tf
from config import CONFIG
from logging_utils import app_logger
import os


def global_library_setup():
    seed = CONFIG['general']['seed']

    tf.get_logger().setLevel('ERROR')

    gpus = tf.config.list_physical_devices('GPU')
    app_logger.info(f"Numero di GPU disponibili: {len(gpus)}")

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    app_logger.info(f"Seed globale impostato per la riproducibilità: {seed}")


def detect_environment():
    """Rileva l'ambiente di esecuzione: Kaggle o locale."""
    if os.path.exists("/kaggle"):
        return "kaggle"
    return "local"
