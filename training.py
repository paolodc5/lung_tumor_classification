from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

from config import CONFIG


def get_callbacks(config_dict=CONFIG['training']['callbacks']):
    """
    Metodo per addestrare il modello con i dati forniti da DataLoader.
    :param model: Modello Keras da addestrare.
    :param train_generator: DataLoader per il training.
    :param val_generator: DataLoader per la validazione.
    """
    patience = config_dict['early_stopping']['patience']
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    return [early_stopping]


def get_augmentation(images, seed=CONFIG['general']['seed']):
    augmentation_layer = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
        tf.keras.layers.RandomRotation(0.2, fill_mode='constant', fill_value=0, seed=seed),
        tf.keras.layers.RandomContrast(factor=0.3, seed=seed),
        RandomShear(shear_range=0.2, seed=seed)
    ])


    images = augmentation_layer(images, training=True)
    images = tf.cast(images, tf.uint8)
    images = images.numpy()

    return images




class RandomShear(tf.keras.layers.Layer):
    def __init__(self, shear_range, seed=None, **kwargs):
        """
        Inizializza il layer di trasformazione Random Shear.

        Args:
            shear_range (float): Valore massimo (sia negativo che positivo) del parametro di shear in radianti.
        """
        super(RandomShear, self).__init__(**kwargs)
        self.shear_range = shear_range
        self.seed = seed

    def build(self, input_shape):
        # Costruzione del layer, qui puoi definire i pesi (se necessario)
        pass

    def call(self, inputs, training=None):
        """
        Applica il Random Shear sulle immagini di input.

        Args:
            inputs: Tensore di input (batch di immagini).
            training: Flag che indica se il layer viene eseguito in fase di training.

        Returns:
            Tensore modificato con shear casuale.
        """
        # Otteniamo batch size, altezza e larghezza
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]

        # Genera un valore casuale di shear per ogni immagine
        shear_angles = tf.random.uniform(
            shape=(batch_size,),
            minval=-self.shear_range,
            maxval=self.shear_range,
            seed=self.seed
        )

        # Funzione per applicare lo shear
        def apply_shear(image, shear_angle):
            # Creiamo la matrice di trasformazione nel formato richiesto da TensorFlow
            transformation_matrix = [
                1.0, tf.tan(shear_angle), 0.0,  # Prima riga della trasformazione
                0.0, 1.0, 0.0,  # Seconda riga della trasformazione
                0.0, 0.0  # Padding per compatibilità
            ]

            # Applica la trasformazione
            image = tf.raw_ops.ImageProjectiveTransformV3(
                images=image[None, ...],  # Aggiungiamo una dimensione per batch singolo
                transforms=[transformation_matrix],  # Trasforma come lista (1x8)
                output_shape=[height, width],
                interpolation="BILINEAR",
                fill_mode="CONSTANT",
                fill_value=0
            )[0]

            return image

        # Applichiamo shear per ogni immagine del batch
        sheared_images = tf.map_fn(
            lambda x: apply_shear(x[0], x[1]),
            (inputs, shear_angles),
            dtype=inputs.dtype
        )
        return sheared_images



if __name__ == "__main__":
    # Creazione del layer di shear
    random_shear_layer = RandomShear(shear_range=0.5)  # Valore massimo di shear ±0.5 radianti

    # Creiamo un'immagine di input casuale (batch di una singola immagine)
    input_image = tf.random.uniform(shape=(1, 224, 224, 3))  # Batch con 1 immagine (224x224 con 3 canali)

    # Applichiamo la trasformazione durante la fase di training
    output_image = random_shear_layer(input_image, training=True)

    print("Forma immagine d'input:", input_image.shape)
    print("Forma immagine trasformata:", output_image.shape)









