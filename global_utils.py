import json

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

    app_logger.info(f"Versione di tensorflow: {tf.__version__}")

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    app_logger.info(f"Seed globale impostato per la riproducibilità: {seed}")


def detect_environment():
    """Rileva l'ambiente di esecuzione: Kaggle o locale."""
    if os.path.exists("/kaggle"):
        return "kaggle"
    return "local"


def convert_dict_to_json(d, folder_path=CONFIG["output"]["save_path"], file_name="output_dict.json"):
    """
    Converte un dizionario in un file JSON e lo salva nella cartella specificata.

    :param d: Dizionario da convertire.
    :param folder_path: Percorso della cartella dove salvare il file JSON.
    :param file_name: Nome del file JSON (default: "output.json").
    """
    try:
        # Assicurati che la cartella esista
        os.makedirs(folder_path, exist_ok=True)

        # Percorso completo del file
        file_path = os.path.join(folder_path, file_name)

        # Scrivi il dizionario in formato JSON nel file
        with open(file_path, 'w') as json_file:
            json.dump(d, json_file, indent=4)

        print(f"JSON salvato correttamente in {file_path}")
        return file_path
    except Exception as e:
        print(f"Errore durante il salvataggio del JSON: {e}")
        raise


