import os
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from logging_utils import app_logger, configure_keras_logging
from config import CONFIG
from data_loader_class import DataLoader
from model import build_model






def fit_model(model, train_generator, val_generator, train_steps, val_steps):
    """
    Metodo per addestrare il modello con i dati forniti da DataLoader.
    :param model: Modello Keras da addestrare.
    :param train_generator: DataLoader per il training.
    :param val_generator: DataLoader per la validazione.
    """
    # Parametri di configurazione
    epochs = CONFIG['training']['epochs']
    model_save_path = CONFIG['output']['save_model_path']

    # Creazione delle directory se non esistono
    os.makedirs(model_save_path, exist_ok=True)

    # Callbacks per il salvataggio del modello e early stopping
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(model_save_path, 'best_model.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # Conversione dei DataLoader in generatori


    # Training del modello
    app_logger.info("Inizio del training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=[early_stopping],
        verbose=1
    )

    app_logger.info("Training completato.")
    return history


if __name__ == '__main__':


    configure_keras_logging(app_logger)
    # Inizializzazione dei DataLoader
    train_loader = DataLoader(split='train')
    val_loader = DataLoader(split='val')

    training_steps = len(train_loader)
    val_steps = len(val_loader)

    train_generator = iter(train_loader)  # Converts DataLoader into an iterator yielding (x, y)
    val_generator = iter(val_loader)

    # Creazione del modello
    model = build_model()

    # Addestramento del modello
    history = fit_model(model,
                        train_generator,
                        val_generator,
                        training_steps,
                        val_steps)

    # Salvataggio del modello finale
    final_model_path = os.path.join(CONFIG['output']['save_model_path'], 'final_model.h5')
    model.save(final_model_path)
    app_logger.info(f"Modello finale salvato in {final_model_path}")
