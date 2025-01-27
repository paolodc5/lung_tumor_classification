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
    def median_filtering(data):
        """
        Apply a median filter to a batch of images.
        :param data: Numpy array of shape (n_batch, h, w, 1)
        :return: Filtered images with the same shape as input.
        """
        filtered_data = np.empty_like(data)
        for i in range(data.shape[0]):
            # Apply median filter to each image individually
            filtered_data[i, ..., 0] = ndi.median_filter(data[i, ..., 0], size=5)
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
            #img_uint8 = cv2.normalize(data[i, ..., 0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            #equalized_data[i, ..., 0] = cv2.equalizeHist(img_uint8)
            pass
        return equalized_data

    def apply_pipeline(self):
        #self.data = self.median_filtering(self.data)
        #self.data = self.he(self.data)
        self.clip_values()
        self.normalize()




if __name__ == '__main__':
    image_path = r"C:\Users\dicia\OneDrive - Politecnico di Milano\_BSPMI\MI\Labs\Lab3\mano.jpg"

    img = Image.open(image_path)
    img_array = np.array(img)
    print(img_array.shape)

    proc = DataProcessor(img_array)
    proc.apply_pipeline()

    visualize_images([img_array, proc.data], indices=[0, 1], n_images=2)
    visualize_histograms(img_array, proc.data)
    plt.show()


