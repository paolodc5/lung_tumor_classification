import numpy as np
from logging_utils import app_logger

from data_loader_class import DataLoader
from visualization_utils import visualize_images, visualize_histograms


def test_data_loader():
    """
    Testa il DataLoader con un piccolo batch e verifica il caricamento e il preprocessing.
    """
    try:
        # Inizializzazione del DataLoader
        data_loader = DataLoader(split='test')

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

if __name__ == '__main__':
    test_data_loader()
