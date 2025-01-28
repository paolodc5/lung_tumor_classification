import numpy as np
import os
import pandas as pd
import nrrd
from logging_utils import app_logger
from data_processor import DataProcessor, resize_image
from sklearn.model_selection import train_test_split
from config import CONFIG
from sklearn.utils import resample


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
        self.test = 0

        # Salvataggio/caricamento degli split
        self.split_files = {
            'train': 'split/train_split.csv',
            'val': 'split/val_split.csv',
            'test': 'split/test_split.csv',
        }

        os.makedirs('split',exist_ok=True)


        if all(os.path.exists(file) for file in self.split_files.values()):
            # Carica gli split esistenti dai file
            self.train_df = pd.read_csv(self.split_files['train'])
            self.val_df = pd.read_csv(self.split_files['val'])
            self.test_df = pd.read_csv(self.split_files['test'])
        else:
            self.generate_split()

        # Seleziona lo split
        if split == 'train':
            self.dataset = self.train_df.copy()
        elif split == 'val':
            self.dataset = self.val_df.copy()
        elif split == 'test':
            self.dataset = self.test_df.copy()
            self.test = 1 # flag
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
        if self.test:
            app_logger.debug("Entrato nel blocco test_df")
            # Per il test, un'iterazione senza ciclo infinito
            for start_idx in range(0, self.num_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, self.num_samples)
                batch_rows = self.dataset.iloc[start_idx:end_idx]
                yield self._load_batch(batch_rows)
        else:
            app_logger.debug("Entrato nel blocco train/val")
            # Per train/val, loop infinito
            while True:
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
                    fullslice_data = resize_image(fullslice_data, self.target_image_size)
                    #fullslice_data = fullslice_data[...,np.newaxis]
                    images.append(fullslice_data)
                    app_logger.debug(f"sono dentro il _load_batch e dtype è {fullslice_data.dtype}")
            # Label
            labels.append(row['TumorClass'])

        images = np.array(images)
        labels = np.array(labels)

        # Preprocessing (se definito)
        if self.preprocess_fn:
            processor = self.preprocess_fn(images)
            processor.apply_pipeline()
            images = processor.data

        # This is for binary tasks
        labels = np.where(labels < 4, 0, 1).astype(np.float64)

        return images, labels


    def generate_split(self):
        # Genera nuovi split e salvali
        app_logger.info("Generating new split...")
        full_dataset = pd.read_excel(self.dataset_path)

        # Codifica le classi per il task binario
        full_dataset['BinaryClass'] = np.where(full_dataset['TumorClass'] < 4, 0, 1)

        # Divisione del dataset in training/validation/test con stratificazione basata sulla classe binaria
        train_val_df, self.test_df = train_test_split(
            full_dataset,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=full_dataset['BinaryClass']
        )
        self.train_df, self.val_df = train_test_split(
            train_val_df,
            test_size=self.val_size / (1 - self.test_size),
            random_state=self.random_state,
            stratify=train_val_df['BinaryClass']
        )

        # Oversampling per compensare le classi minoritarie nel dataset di training
        app_logger.info("Applying binary oversampling to balance the training data.")
        majority_class = self.train_df['BinaryClass'].value_counts().idxmax()
        max_count = self.train_df['BinaryClass'].value_counts().max()

        balanced_train_dfs = []
        for class_id, group in self.train_df.groupby('BinaryClass'):
            if len(group) < max_count:
                # Oversampling della classe minoritaria
                oversampled_group = resample(
                    group,
                    replace=True,  # Campionamento con duplicati
                    n_samples=max_count,  # Porta al numero massimo della classe maggioritaria
                    random_state=self.random_state
                )
                balanced_train_dfs.append(oversampled_group)
            else:
                # Mantieni la classe maggioritaria invariata
                balanced_train_dfs.append(group)

        # Concatenazione dei dataframe bilanciati
        self.train_df = pd.concat(balanced_train_dfs, axis=0).reset_index(drop=True)

        # Shuffle per mescolare le classi
        self.train_df = self.train_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        # Salva gli split
        self.train_df.to_csv(self.split_files['train'], index=False)
        self.val_df.to_csv(self.split_files['val'], index=False)
        self.test_df.to_csv(self.split_files['test'], index=False)







# def generate_split_multi(self):
#     # Genera nuovi split e salvali
#     app_logger.info("Generating new split...")
#     full_dataset = pd.read_excel(self.dataset_path)
#
#     # Divisione del dataset in training/validation/test con stratificazione
#     train_val_df, self.test_df = train_test_split(
#         full_dataset,
#         test_size=self.test_size,
#         random_state=self.random_state,
#         stratify=full_dataset['TumorClass']
#     )
#     self.train_df, self.val_df = train_test_split(
#         train_val_df,
#         test_size=self.val_size / (1 - self.test_size),
#         random_state=self.random_state,
#         stratify=train_val_df['TumorClass']
#     )
#
#     # Oversampling per compensare le classi minoritarie nel dataset di training
#     app_logger.info("Applying oversampling to balance the training data.")
#     majority_class = self.train_df['TumorClass'].value_counts().idxmax()
#     max_count = self.train_df['TumorClass'].value_counts().max()
#
#     balanced_train_dfs = []
#     for class_id, group in self.train_df.groupby('TumorClass'):
#         if len(group) < max_count:
#             # Oversampling della classe minoritaria
#             oversampled_group = resample(
#                 group,
#                 replace=True,  # Campionamento con duplicati
#                 n_samples=max_count,  # Porta al numero massimo della classe maggioritaria
#                 random_state=self.random_state
#             )
#             balanced_train_dfs.append(oversampled_group)
#         else:
#             # Mantieni la classe maggioritaria invariata
#             balanced_train_dfs.append(group)
#
#     # Concatenazione dei dataframe bilanciati
#     self.train_df = pd.concat(balanced_train_dfs, axis=0).reset_index(drop=True)
#
#     # Shuffle per mescolare le classi
#     self.train_df = self.train_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
#
#     # Salva gli split
#     self.train_df.to_csv(self.split_files['train'], index=False)
#     self.val_df.to_csv(self.split_files['val'], index=False)
#     self.test_df.to_csv(self.split_files['test'], index=False)






 # def generate_split_no_resampling(self):
 #        # Genera nuovi split e salvali
 #        app_logger.info("Generating new split...")
 #        full_dataset = pd.read_excel(self.dataset_path)
 #        train_val_df, self.test_df = train_test_split(
 #            full_dataset,
 #            test_size=self.test_size,
 #            random_state=self.random_state,
 #            stratify=full_dataset['TumorClass']
 #        )
 #        self.train_df, self.val_df = train_test_split(
 #            train_val_df,
 #            test_size=self.val_size / (1 - self.test_size),
 #            random_state=self.random_state,
 #            stratify=train_val_df['TumorClass']
 #        )
 #        # Salva gli split
 #        self.train_df.to_csv(self.split_files['train'], index=False)
 #        self.val_df.to_csv(self.split_files['val'], index=False)
 #        self.test_df.to_csv(self.split_files['test'], index=False)

