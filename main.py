import os
from logging_utils import app_logger, configure_keras_logging
from config import CONFIG
from data_loader_class import DataLoader
from model import build_model
from evaluation_utils import evaluate_results
from training import get_callbacks
from global_utils import global_library_setup, convert_dict_to_json


if __name__ == '__main__':

    global_library_setup()
    configure_keras_logging(app_logger)
    convert_dict_to_json(CONFIG, file_name="config_dict.json") # Saves the configuration of the current run


    # Inizializzazione dei DataLoader
    train_loader = DataLoader(split='train')
    val_loader = DataLoader(split='val')

    # Creazione del modello
    model = build_model()

    # creazione directory di salvataggio se non esistono
    os.makedirs(CONFIG['output']['save_model_path'], exist_ok=True)

    # Gestione callbacks
    callb = get_callbacks()


    # Addestramento del modello
    app_logger.info("Inizio del training...")
    history = model.fit(
        iter(train_loader),
        validation_data=iter(val_loader),
        epochs=CONFIG['training']['epochs'],
        steps_per_epoch=len(train_loader),
        validation_steps=len(val_loader),
        callbacks=callb,
        verbose=1,
    )
    app_logger.info("Training completato.")



    # Salvataggio del modello finale
    final_model_path = os.path.join(CONFIG['output']['save_model_path'], 'final_model.keras')
    model.save(final_model_path)
    app_logger.info(f"Modello finale salvato in {final_model_path}")

    evaluate_results(history, model)
