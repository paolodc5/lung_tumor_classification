from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

from config import CONFIG


def get_callbacks(config_dict=CONFIG['training']['callbacks']):
    """
    Metodo per addestrare il modello con i dati forniti da DataLoader.
    :param model: Modello Keras da addestrare.
    :param train_generator: DataLoader per il training.
    :param val_generator: DataLoader per la validazione.
    """
    patience = config_dict['early_stopping']['patience']
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    return [early_stopping]


def get_augmentation(images, seed=CONFIG['general']['seed']):
    augmentation_layer = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
        tf.keras.layers.RandomRotation(0.2, fill_mode='constant', fill_value=0, seed=seed),
        tf.keras.layers.RandomContrast(factor=0.3, seed=seed),
    ])


    images = augmentation_layer(images, training=True)
    images = tf.cast(images, tf.uint8)
    images = images.numpy()

    return images














