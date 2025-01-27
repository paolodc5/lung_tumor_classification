import numpy as np
from logging_utils import app_logger
from PIL import Image
import matplotlib.pyplot as plt

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



    def clip_values(self, min_val=-1000, max_val=3000):
        """
        Clips the values of the data between min_val and max_val.
        :param min_val: Minimum value for the clipping range
        :param max_val: Maximum value for the clipping range
        """
        # Verifica che min_val sia minore di max_val
        if min_val >= max_val:
            error = ValueError("min_val must be less than max_val for clipping range")
            app_logger.error(str(error))
            raise error
        # Clip the values
        self.data = np.clip(self.data, min_val, max_val)
        # app_logger.info(f"Values clipped to range [{min_val}, {max_val}].")


    def normalize(self, norm_type='standard'):
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



if __name__ == '__main__':
    image_path = r"C:\Users\dicia\OneDrive - Politecnico di Milano\_BSPMI\MI\Labs\Lab3\mano.jpg"

    img = Image.open(image_path)
    img_array = np.array(img)

    proc = DataProcessor(img_array)
    proc.clip_values(4,35)
    proc.normalize(norm_type='min-max')

    min_max_norm = proc.data

    proc.normalize(norm_type='z-score')
    z_score_norm = proc.data


    plt.imshow(proc.original_data, cmap='gray')
    plt.show()

    visualize_histograms(proc.original_data, min_max_norm)
    visualize_histograms(proc.original_data, z_score_norm)