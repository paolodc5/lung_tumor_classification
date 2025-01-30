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
    monitor_value = config_dict['early_stopping']['monitor']
    early_stopping = EarlyStopping(
        monitor=monitor_value,
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    return [early_stopping]


def get_augmentation(images, seed=CONFIG['general']['seed']):
    augmentation_layer = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
        tf.keras.layers.RandomRotation(0.2, fill_mode='constant', fill_value=0, seed=seed),
        RandomContrast(lower=0.8, upper=1.2, seed=seed),
        RandomShear(shear_range=0.2, seed=seed),
        RandomBrightness(max_delta=0.3, seed=seed)
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



class RandomBrightness(tf.keras.layers.Layer):
    def __init__(self, max_delta, seed=None, **kwargs):
        """
        Layer per applicare Random Brightness con normalizzazione dei valori tra 0 e 1,
        e successiva riconversione a uint8 (0-255).

        Args:
            max_delta (float): Valore massimo di variazione positiva o negativa.
                               Deve essere compreso tra 0 e 1.
            seed (int, opzionale): Seed per generare numeri casuali.
        """
        super(RandomBrightness, self).__init__(**kwargs)
        self.max_delta = max_delta
        self.seed = seed

    def call(self, inputs, training=True):
        """
        Aggiunge casualmente random brightness alle immagini, normalizza in [0, 1] e torna in uint8.

        Args:
            inputs: Batch di immagini (tensore).
            training: Flag per indicare se siamo in fase di training.

        Returns:
            Tensore `uint8` con luminosità modificata.
        """
        if not training:
            return tf.cast(inputs, tf.uint8)  # Ritorna le immagini originali in uint8 in modalità inference

        # Normalizza l'immagine da [0, 255] a [0, 1]
        inputs = tf.cast(inputs, tf.float32) / 255.0

        # Genera un valore casuale per la variazione di luminosità
        delta = tf.random.uniform(
            shape=[],
            minval=-self.max_delta,
            maxval=self.max_delta,
            seed=self.seed
        )

        # Modifica la luminosità aggiungendo il delta
        brightened = tf.clip_by_value(inputs + delta, 0.0, 1.0)

        # Riporta i valori a [0, 255] e converte in uint8
        return tf.cast(brightened * 255.0, tf.uint8)



class RandomContrast(tf.keras.layers.Layer):
    def __init__(self, lower, upper, seed=None, **kwargs):
        """
        Layer per applicare Random Contrast con normalizzazione dei valori tra 0 e 1,
        e successiva riconversione a uint8 (0-255).

        Args:
            lower (float): Limite inferiore del fattore di contrasto (minore di 1 diminuisce il contrasto).
            upper (float): Limite superiore del fattore di contrasto (maggiore di 1 aumenta il contrasto).
            seed (int, opzionale): Seed per generare numeri casuali.
        """
        super(RandomContrast, self).__init__(**kwargs)
        self.lower = lower
        self.upper = upper
        self.seed = seed

    def call(self, inputs, training=True):
        """
        Aggiunge casualmente random contrast alle immagini, normalizza in [0, 1] e torna in uint8.

        Args:
            inputs: Batch di immagini (tensore).
            training: Flag per indicare se siamo in fase di training.

        Returns:
            Tensore `uint8` con contrasto modificato.
        """
        if not training:
            return tf.cast(inputs, tf.uint8)  # Nessuna modifica in inferenza

        # Normalizza l'immagine da [0, 255] a [0, 1]
        inputs = tf.cast(inputs, tf.float32) / 255.0

        # Genera un valore casuale per il fattore di contrasto
        contrast_factor = tf.random.uniform(
            shape=[],
            minval=self.lower,
            maxval=self.upper,
            seed=self.seed
        )

        # Calcola il valore medio su ciascun canale (per ogni immagine nel batch)
        means = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)

        # Modifica il contrasto
        contrasted = tf.clip_by_value((inputs - means) * contrast_factor + means, 0.0, 1.0)

        # Riporta i valori a [0, 255] e converte in uint8
        return tf.cast(contrasted * 255.0, tf.uint8)





if __name__ == "__main__":
    # Creazione del layer di shear
    random_shear_layer = RandomShear(shear_range=0.5)  # Valore massimo di shear ±0.5 radianti

    # Creiamo un'immagine di input casuale (batch di una singola immagine)
    input_image = tf.random.uniform(shape=(1, 224, 224, 3))  # Batch con 1 immagine (224x224 con 3 canali)

    # Applichiamo la trasformazione durante la fase di training
    output_image = random_shear_layer(input_image, training=True)

    print("Forma immagine d'input:", input_image.shape)
    print("Forma immagine trasformata:", output_image.shape)









