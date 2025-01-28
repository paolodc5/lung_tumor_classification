import numpy as np
from logging_utils import app_logger

from data_loader_class import DataLoader
from visualization_utils import visualize_images, visualize_histograms
from training_utils import get_test_data_and_labels

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






if __name__ == '__main__':
    test_generator = DataLoader(split='test')
    images, labels = get_test_data_and_labels(test_generator)
    test_get_test_data_and_labels(images, labels)

    #test_data_loader()


