import numpy as np
import os
import pandas as pd
import nrrd
from logging_utils import app_logger
from data_processor import DataProcessor
from sklearn.model_selection import train_test_split
from config import CONFIG


class DataLoader:
    def __init__(self, split='train'):
        """
        Classe per gestire il caricamento dei dati e il preprocessing on-the-fly.
        :param split: Tipo di dataset (train test validation)

        """
        self.dataset_path = CONFIG['data']['dataset_path']
        self.train_path = CONFIG['data']['train_path']
        self.batch_size = CONFIG['training']['batch_size']
        self.target_image_size = CONFIG['preprocessing']['resize']
        self.preprocess_fn = DataProcessor
        self.test_size = CONFIG['data']['test_split']
        self.val_size = CONFIG['data']['validation_split']
        self.random_state = CONFIG['general']['seed']
        self.norm_type = CONFIG['preprocessing']['normalization_type']

        # Carica il dataset e dividi in train, test, validation
        full_dataset = pd.read_excel(self.dataset_path)
        train_val_df, self.test_df = train_test_split(full_dataset, test_size=self.test_size, random_state=self.random_state,
                                                      stratify=full_dataset['TumorClass'])
        self.train_df, self.val_df = train_test_split(train_val_df, test_size=self.val_size / (1 - self.test_size),
                                                      random_state=self.random_state, stratify=train_val_df['TumorClass'])

        # Seleziona lo split
        if split == 'train':
            self.dataset = self.train_df
        elif split == 'val':
            self.dataset = self.val_df
        elif split == 'test':
            self.dataset = self.test_df
        else:
            error = ValueError("Split must be 'train', 'val', or 'test'")
            app_logger.error(error)
            raise error

        self.num_samples = len(self.dataset)
        self.on_epoch_end()

    def on_epoch_end(self):
        """Mescola il dataset a fine epoca."""
        self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        """Numero totale di batch."""
        return int(np.ceil(self.num_samples / self.batch_size))

    def __iter__(self):
        """Iteratore per generare batch."""
        while True:  # Ciclo infinito per continuare a fornire batch
            for start_idx in range(0, self.num_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, self.num_samples)
                batch_rows = self.dataset.iloc[start_idx:end_idx]
                yield self._load_batch(batch_rows)

    def _load_batch(self, batch_rows):
        """Carica e preprocessa un singolo batch."""
        images = []
        labels = []

        for _, row in batch_rows.iterrows():
            # Full_slice
            fullslice_name = row['Full_slice']
            if isinstance(fullslice_name, str):
                fullslice_path = os.path.join(self.train_path, fullslice_name)
                if os.path.isfile(fullslice_path):
                    fullslice_data, _ = nrrd.read(fullslice_path)
                    fullslice_data = np.resize(fullslice_data, self.target_image_size + (1,))
                    images.append(fullslice_data)

            # Label
            labels.append(row['TumorClass'])

        images = np.array(images)
        labels = np.array(labels)

        # Preprocessing (se definito)
        if self.preprocess_fn:
            processor = self.preprocess_fn(images)
            processor.clip_values(min_val=-1000, max_val=3000)  # Esempio di clipping
            processor.normalize(norm_type=self.norm_type)  # Normalizzazione Min-Max
            images = processor.data

        # This is for binary tasks
        labels = np.where(labels < 4, 0, 1)

        return images, labels
    




