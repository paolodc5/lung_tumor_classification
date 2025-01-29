import numpy as np

from config import CONFIG
from logging_utils import app_logger
from PIL import Image
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import cv2

from visualization_utils import visualize_images, visualize_histograms


class DataProcessor:
    def __init__(self, data):
        """
        This class normalizes data
        :param data: Numpy array of images of type (n_images, height, width, channels)
        """

        if not isinstance(data, np.ndarray):
            error = TypeError(f"Data to be normalized is not a numpy array instead it is {type(data)}")
            app_logger.error(str(error))
            raise error

        self.data = data.copy()
        self.range = CONFIG['preprocessing']['clipping_range']


    def clip_values(self):
        """
        Clips the values of the data between min_val and max_val.
        :param min_val: Minimum value for the clipping range
        :param max_val: Maximum value for the clipping range
        """
        # Verifica che min_val sia minore di max_val
        min_val = self.range[0]
        max_val = self.range[1]
        app_logger.debug(f"Clipping values between {min_val} and {max_val}.")

        if min_val >= max_val:
            error = ValueError("min_val must be less than max_val for clipping range")
            app_logger.error(str(error))
            raise error
        # Clip the values
        self.data = np.clip(self.data, min_val, max_val)
        # app_logger.info(f"Values clipped to range [{min_val}, {max_val}].")


    def pre_normalize(self):
            """
            Normalizza i dati tra 0 e 255 e li converte in interi a 8 bit (uint8).
            """
            # Trova il valore minimo e massimo nei dati
            data_min = np.min(self.data)
            data_max = np.max(self.data)

            if data_max > data_min:
                # Normalizza i dati sulla scala 0-255
                self.data = (self.data - data_min) / (data_max - data_min)*255

            else:
                # Se i valori sono costanti, assegna direttamente 0
                self.data = np.zeros_like(self.data)

            # Converte i dati in interi a 8 bit
            self.data = self.data.astype(np.uint8)


    def normalize(self, norm_type=CONFIG['preprocessing']['normalization_type']):
        try:
            if norm_type == "min-max":
                # Normalizzazione Min-Max: scala l'immagine tra 0 e 1
                img_min = np.min(self.data)
                img_max = np.max(self.data)
                if img_max > img_min:  # Evitiamo la divisione per zero
                    self.data = (self.data - img_min) / (img_max - img_min)
                else:
                    raise ValueError("Image max is less or equal than image min")

            elif norm_type == "z-score":
                # Normalizzazione Z-Score: utilizza media e deviazione standard
                mean = np.mean(self.data)
                std = np.std(self.data)
                app_logger.debug(f"Image mean: {mean}, Image std: {std}")
                if std > 0:  # Evitiamo la divisione per zero
                    self.data = (self.data - mean) / std
                else:
                    raise ValueError("Image std is less or equal than zero")
            elif norm_type == "None":
                pass
            else:
                raise ValueError(f"'{norm_type}' is not a supported scaling type.")
            


            # Se il tipo è valido, esegui la normalizzazione
            # app_logger.info(f"Normalizing data with {norm_type}.")
            return f"Data normalized using {norm_type}."

        except ValueError as e:
            # Log dell'errore
            app_logger.error(str(e))
            app_logger.error("I did not normalize the images")
            raise

    @staticmethod
    def median_filtering(data, filter_size=CONFIG['preprocessing']['median_filter_size']):
        """
        Apply a median filter to a batch of images efficiently using OpenCV.
        :param data: Numpy array of shape (n_batch, h, w, 1).
        :param filter_size: Size of the median filter.
        :return: Filtered images with the same shape as input.
        """
        if len(data.shape) < 4 or data.shape[-1] != 1:
            raise ValueError("Data must have shape (n_batch, height, width, 1) for median filtering.")

        # Use OpenCV for batch processing
        filtered_data = np.empty_like(data)
        for i in range(data.shape[0]):
            # cv2.medianBlur expects a 2D single-channel image
            filtered_data[i, ..., 0] = cv2.medianBlur(data[i, ..., 0].astype(np.uint8), filter_size)
        return filtered_data


    @staticmethod
    def he(data):
        """
        Apply histogram equalization to a batch of images.
        :param data: Numpy array of shape (n_batch, h, w, 1)
        :return: Images after histogram equalization with the same shape as input.
        """
        equalized_data = np.empty_like(data)
        for i in range(data.shape[0]):
            # Convert image to uint8 (required for cv2.equalizeHist)
            equalized_data[i, ..., 0] = cv2.equalizeHist(data[i, ..., 0].astype(np.uint8))

        return equalized_data

    @staticmethod
    def clahe(data,
              clip_limit=CONFIG['preprocessing']['clahe_clip_limit'],
              tile_grid_size=CONFIG['preprocessing']['clahe_tile_grid_size']):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a batch of images.
        :param data: Numpy array of shape (n_batch, h, w, 1).
        :param clip_limit: Threshold for contrast limiting.
        :param tile_grid_size: Size of the grid for the CLAHE algorithm.
        :return: Images after applying CLAHE with the same shape as input.
        """
        # Create a CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        equalized_data = np.empty_like(data)

        for i in range(data.shape[0]):
            # Convert the image to uint8 if necessary
            #img_uint8 = cv2.normalize(data[i, ..., 0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            # Apply CLAHE
            equalized_data[i, ..., 0] = clahe.apply(data[i, ..., 0].astype(np.uint8))

        return equalized_data


    @staticmethod
    def convert_to_rgb(data):
        """
        Converte un array di immagini in formato grayscale in RGB.
        Ripete il singolo canale su 3 (R, G, B).

        :param data: Array numpy di immagini (n_images, height, width, 1) o (n_images, height, width).
        :return: Array numpy di immagini (n_images, height, width, 3).
        """
        # Assicurati che l'input abbia almeno 3 dimensioni
        if len(data.shape) == 3:
            # Aggiungi un asse per simulare il canale
            images = data[..., np.newaxis]

        if data.shape[-1] != 1:
            raise ValueError("Le immagini devono essere in formato grayscale con 1 canale.")

        # Ripeti il canale singolo 3 volte per creare un'immagine RGB
        images_rgb = np.repeat(data, 3, axis=-1)

        return images_rgb


    @staticmethod
    def opening(data, kernel_size=CONFIG['preprocessing']['morph_kernel_size']):
        """
        Applica l'operazione di apertura (opening) a un batch di immagini.

        :param data: Numpy array di immagini (n_batch, h, w, 1).
        :param kernel_size: Dimensione del kernel strutturante.
        :return: Immagini processate con opening.
        """
        if len(data.shape) < 4 or data.shape[-1] != 1:
            raise ValueError("Data must have shape (n_batch, height, width, 1) for morphological operations.")

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opened_data = np.empty_like(data)

        for i in range(data.shape[0]):
            opened_data[i, ..., 0] = cv2.morphologyEx(data[i, ..., 0], cv2.MORPH_OPEN, kernel)

        return opened_data


    @staticmethod
    def closing(data, kernel_size=CONFIG['preprocessing']['morph_kernel_size']):
        """
        Applica l'operazione di chiusura (closing) a un batch di immagini.

        :param data: Numpy array di immagini (n_batch, h, w, 1).
        :param kernel_size: Dimensione del kernel strutturante.
        :return: Immagini processate con closing.
        """
        if len(data.shape) < 4 or data.shape[-1] != 1:
            raise ValueError("Data must have shape (n_batch, height, width, 1) for morphological operations.")

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed_data = np.empty_like(data)

        for i in range(data.shape[0]):
            closed_data[i, ..., 0] = cv2.morphologyEx(data[i, ..., 0], cv2.MORPH_CLOSE, kernel)

        return closed_data


    def apply_pipeline(self):
        self.clip_values()
        self.pre_normalize()
        self.data = self.median_filtering(self.data)
        self.data = self.he(self.data)
        self.data = self.clahe(self.data)
        self.data = self.opening(self.data)
        self.data = self.closing(self.data)
        self.normalize()
        self.data = self.convert_to_rgb(self.data)


# Fuori dalla classe
def resize_image(image, target_size):
    """
    Ridimensiona un'immagine utilizzando OpenCV.
    :param image: Numpy array dell'immagine (h, w) o (h, w, 1).
    :param target_size: Tuple con la dimensione desiderata (height, width).
    :return: Immagine ridimensionata.
    """
    if image.ndim == 2:  # Per immagini con un singolo canale
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        return resized[..., np.newaxis]  # Aggiunge nuovamente il canale
    else:  # Per immagini in scala di grigi senza canale
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)




if __name__ == '__main__':
    image_path = r"C:\Users\dicia\OneDrive - Politecnico di Milano\_BSPMI\MI\Labs\Lab3\mano.jpg"

    img = Image.open(image_path)
    img_array = np.array(img)
    img_array = img_array[np.newaxis, ..., np.newaxis]
    img_array = np.concatenate((img_array, img_array, img_array), axis=0)
    print(img_array.shape)


    proc = DataProcessor(img_array)
    print(proc.data.shape)
    print(proc.data.dtype)

    proc.apply_pipeline()

    # img_filt = ndi.median_filter(img_array, size=10)

    visualize_images([img_array[2,...],proc.data[1,...]], indices=[0, 1], n_images=2)
    visualize_histograms(img_array, proc.data)
    #plt.imshow(proc.data[0,...], cmap='gray')
    #plt.show()
    print(proc.data.shape)
    print(proc.data.dtype)

