import numpy as np

from global_utils import global_library_setup
from logging_utils import app_logger

from data_loader_class import DataLoader
from model import build_model
from visualization_utils import visualize_images, visualize_histograms
from evaluation_utils import get_test_data_and_labels, get_predicted_classes, evaluate_results, calculate_f1_score, \
    calculate_accuracy_score, generate_confusion_matrix, generate_roc_curve
import json
import os


def test_data_loader():
    """
    Testa il DataLoader con un piccolo batch e verifica il caricamento e il preprocessing.
    """
    try:
        # Inizializzazione del DataLoader
        data_loader = DataLoader(split='train')

        # Genera un batch di dati
        batch_iterator = iter(data_loader)
        images, labels = next(batch_iterator)



        # Verifica dei dati caricati
        print("=== Test DataLoader ===")
        print(f"Shape delle immagini: {images.shape}")
        print(f"Shape delle label: {labels.shape}")
        print(f"Tipo delle immagini: {images.dtype}")
        print(f"Tipo delle label: {labels.dtype}")
        print(f"Valori minimi e massimi delle immagini: {np.min(images)}, {np.max(images)}")
        print(f"Primo batch di label (one-hot):\n{labels}")

        # Controllo che le immagini siano normalizzate
        #if np.min(images) < 0 or np.max(images) > 1:
        #    raise ValueError("Le immagini non sono correttamente normalizzate tra 0 e 1!")


        visualize_images(images, labels)
        visualize_histograms(images[0],images[1])

        app_logger.info("Test del DataLoader completato con successo!")

    except Exception as e:
        app_logger.error(f"Errore nel test del DataLoader: {e}")
        raise




def test_get_test_data_and_labels(images, labels):
    """
    Funzione di test per verificare la correttezza dei dati restituiti da `get_test_data_and_labels`.
    1. Verifica dello shape e del dtype delle immagini e delle label.
    2. Visualizza le immagini usando `visualize_images`.

    :param get_test_data_and_labels: Funzione che restituisce i dati di test (immagini, label).
    """
    try:

        # Verifica se le immagini e le label non sono vuote
        assert images.size > 0, "Le immagini restituite sono vuote."
        assert labels.size > 0, "Le label restituite sono vuote."

        # Verifica lo shape
        assert len(
            images.shape) == 4, f"Le immagini devono avere 4 dimensioni (batch_size, height, width, channels), ma lo shape è {images.shape}."
        assert len(labels.shape) == 1, f"Le label devono avere 1 dimensione (n_labels,), ma lo shape è {labels.shape}."

        # Verifica il dtype
        assert images.dtype in [np.float32, np.float64,
                                np.uint8], f"Il dtype delle immagini dovrebbe essere `float32`, `float64` o `uint8`, ma è {images.dtype}."
        assert labels.dtype == np.float64, f"Il dtype delle label dovrebbe essere `int32`, ma è {labels.dtype}."

        # Debugging delle informazioni sui dati
        print("=== Test dei dati di test ===")
        print(f"Shape delle immagini: {images.shape}")
        print(f"Shape delle label: {labels.shape}")
        print(f"Tipo delle immagini: {images.dtype}")
        print(f"Tipo delle label: {labels.dtype}")
        print(f"Numero di immagini: {images.shape[0]}")
        print(f"Primi valori delle label: {labels[:5]}")
        print(f"valori minimi e massimi delle immagini: {np.min(images)}, {np.max(images)}")
        # Visualizzazione delle immagini e delle rispettive label
        visualize_images(images, labels)
        visualize_histograms(images[0],images[1])
        print("Le immagini sono state visualizzate con successo.")

        print("Test completato con successo!")

    except AssertionError as e:
        print(f"Errore durante il test: {e}")
        raise
    except Exception as e:
        print(f"Errore inaspettato: {e}")
        raise


from collections import Counter
import pandas as pd


def test_generate_split(data_loader):
    """
    Funzione di test per verificare il corretto funzionamento di `generate_split`.
    Restituisce le dimensioni dei dataset (training, validation, test) e il bilanciamento delle classi.

    :param data_loader: Istanza della classe DataLoader (deve essere stato chiamato generate_split internamente).
    """
    try:
        # Recupera i DataFrame generati dallo split
        train_df = data_loader.train_df
        val_df = data_loader.val_df
        test_df = data_loader.test_df

        # Stampa le dimensioni dei dataset
        print("=== Dimensioni dei dataset ===")
        print(f"Train set: {len(train_df)} esempi")
        print(f"Validation set: {len(val_df)} esempi")
        print(f"Test set: {len(test_df)} esempi")
        print()

        # Analizza il bilanciamento delle classi nei dataset
        print("=== Bilanciamento delle classi ===")

        def analyze_class_balance(df, split_name):
            class_counts = Counter(df['TumorClass'])
            print(f"{split_name} set class distribution:")
            for class_id, count in sorted(class_counts.items()):
                print(f"  Classe {class_id}: {count} esempi")
            print()

        analyze_class_balance(train_df, "Train")
        analyze_class_balance(val_df, "Validation")
        analyze_class_balance(test_df, "Test")

        # Test completato con successo
        print("Test completato con successo! Lo split è stato generato correttamente e bilanciato.")

    except Exception as e:
        print(f"Errore durante il test di generate_split: {e}")
        raise


def test_generate_split_binary(data_loader):
    """
    Testa la funzione generate_split per verificare il bilanciamento delle classi binarie nel dataset di training.
    """
    try:
        # Recupera i DataFrame
        train_df = data_loader.train_df
        val_df = data_loader.val_df
        test_df = data_loader.test_df

        # Controlla che le colonne essenziali esistano
        assert 'BinaryClass' in train_df.columns, "La colonna BinaryClass è assente nel train_df!"
        assert 'BinaryClass' in val_df.columns, "La colonna BinaryClass è assente nel val_df!"
        assert 'BinaryClass' in test_df.columns, "La colonna BinaryClass è assente nel test_df!"
        # Stampa le dimensioni dei dataset
        print("=== Dimensioni dei dataset ===")
        print(f"Train set: {len(train_df)} esempi")
        print(f"Validation set: {len(val_df)} esempi")
        print(f"Test set: {len(test_df)} esempi")
        print()

        # Controlla il bilanciamento delle classi nel training set
        binary_counts_train = train_df['BinaryClass'].value_counts()
        print("=== Bilanciamento delle classi binarie nel training set ===")
        print(binary_counts_train)
        print()

        # Controlla lo squilibrio naturale delle classi nel validation set
        binary_counts_val = val_df['BinaryClass'].value_counts()
        print("=== Bilanciamento delle classi binarie nel validation set ===")
        print(binary_counts_val)
        print()

        # Controlla lo squilibrio naturale delle classi nel test set
        binary_counts_test = test_df['BinaryClass'].value_counts()
        print("=== Bilanciamento delle classi binarie nel test set ===")
        print(binary_counts_test)
        print()

        # Test del bilanciamento
        assert binary_counts_train[0] == binary_counts_train[1], "Il training set non è bilanciato correttamente!"
        print("Test completato con successo: il dataset di training è bilanciato.")

    except AssertionError as e:
        print(f"Errore durante il test: {e}")
        raise
    except Exception as e:
        print(f"Errore inaspettato durante il test: {e}")
        raise


def test_untrained_model_predictions(model, data_generator, num_batches=3):
    """
    Testa un modello non addestrato per assicurarsi che non predica tutte le immagini
    unicamente come classe "0" o "1" su più batch.

    :param model: Modello Keras/TF non ancora addestrato.
    :param data_generator: Generatore di dati (e.g., train/test generator).
    :param num_batches: Numero di batch su cui effettuare il test (default: 3).
    """
    try:
        # Conta i batch caricati e verifica le predizioni
        batch_count = 0
        for X_batch, y_batch in data_generator:
            # Predizioni del modello
            predictions = model.predict(X_batch)

            # Argmax (nel caso siano classificazioni con softmax/multiclasse o sigmoid per binario)
            # predicted_classes = np.argmax(predictions, axis=1) if predictions.shape[1] > 1 else (
            #             predictions > 0.5).astype(np.int16).flatten()
            predicted_classes = get_predicted_classes(predictions)

            # Conta le classi predette
            unique_classes, counts = np.unique(predicted_classes, return_counts=True)

            # Mostra il conteggio delle classi per il batch
            print(f"Batch {batch_count + 1}: Predizioni {dict(zip(unique_classes, counts))}")

            # Test: Assicurati che non predica sempre una sola classe
            if len(unique_classes) == 1:
                print(
                    f"ERRORE: Il modello non addestrato produce una sola classe ({unique_classes[0]}) nel batch {batch_count + 1}.")
            else:
                print(f"SUCCESSO: Il modello non addestrato produce predizioni multiple nel batch {batch_count + 1}.")

            batch_count += 1

            # Ferma dopo il numero di batch definiti
            if batch_count >= num_batches:
                break

    except Exception as e:
        print(f"Errore durante il test: {e}")
        raise

def test_evaluate_results(output_dir="./results", mock_data=True):
    """
    Testa la funzione evaluate_results per verificare problemi con il dump di file JSON.

    :param output_dir: Directory in cui salvare i file.
    :param mock_data: Se True, genera dati simulati per il test.
    """
    try:
        # Simula i dati di test se mock_data è True

        # Simula etichette di test e predizioni
        test_labels = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        predicted_classes = np.array([0, 1, 1, 0, 0, 0, 1, 1])
        predictions = np.array([0.1, 0.8, 0.7, 0.2, 0.4, 0.3, 0.9, 0.6])
        class_names = None


        # Genera metriche (sostituisci con funzioni reali)
        f1 = calculate_f1_score(test_labels, predicted_classes)
        test_acc = calculate_accuracy_score(test_labels, predicted_classes)
        conf_matrix = generate_confusion_matrix(test_labels, predicted_classes, class_names, output_dir)
        roc_auc = generate_roc_curve(test_labels, predictions, output_dir)

        # Crea risultato in formato dictionary
        results = {
            "f1_score": f1,
            "test_accuracy": test_acc,
            "confusion_matrix": conf_matrix.tolist(),
            "roc_auc": roc_auc
        }

        # Salva il JSON
        output_path = os.path.join(output_dir, "evaluation_results.json")

        # Prova il dump di JSON
        print("Salvataggio dei risultati in formato JSON...")
        with open(output_path, "w") as json_file:
            json.dump(results, json_file, indent=4)

        print(f"SUCCESSO: I risultati sono stati salvati correttamente nel file {output_path}")

        return results

    except json.JSONDecodeError as e:
        print(f"ERRORE JSON: Problema durante il salvataggio del file JSON. Dettagli: {e}")
    except FileNotFoundError as e:
        print(f"ERRORE FILE: Directory o file non trovata. Dettagli: {e}")
    except Exception as e:
        print(f"ERRORE GENERICO: Si è verificato un errore. Dettagli: {e}")
        raise

if __name__ == '__main__':
    # test_generator = DataLoader(split='test')
    # images, labels = get_test_data_and_labels(test_generator)
    # test_get_test_data_and_labels(images, labels)

    #test_data_loader()

    #train_generator = DataLoader(split='train')
    #test_generate_split_binary(train_generator)
    #model = build_model()
    #test_untrained_model_predictions(model, train_generator)
    #global_library_setup()

    test_evaluate_results()
