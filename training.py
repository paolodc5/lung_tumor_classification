from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping



def get_callbacks():
    """
    Metodo per addestrare il modello con i dati forniti da DataLoader.
    :param model: Modello Keras da addestrare.
    :param train_generator: DataLoader per il training.
    :param val_generator: DataLoader per la validazione.
    """

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    return [early_stopping]







