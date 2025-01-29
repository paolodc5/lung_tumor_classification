import tensorflow as tf
import numpy as np
import cv2


class GradCAM(tf.keras.Model):
    def __init__(self, model, layer_name):
        """
        Inizializza GradCAM con un modello e il layer convoluzionale da analizzare.

        Args:
            model: Modello Keras con backbone + classificatore.
            layer_name: Nome del layer convoluzionale da cui estrarre le feature maps.
        """
        super(GradCAM, self).__init__()
        self.model = model
        self.layer_name = layer_name
        self.grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output, model.output]
        )

    def compute_heatmap(self, img_array, class_idx=None):
        """
        Calcola la heatmap Grad-CAM per l'immagine data.

        Args:
            img_array: Immagine preprocessata (shape [1, H, W, C]).
            class_idx: Indice della classe target (se None, usa la classe predetta).

        Returns:
            Heatmap normalizzata tra 0 e 1.
        """
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])  # Usa la classe predetta
            loss = predictions[:, class_idx]  # Logit della classe target

        # Calcola i gradienti della loss rispetto alle attivazioni convoluzionali
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Global Average Pooling

        # Moltiplica i gradienti per la feature map
        conv_outputs = conv_outputs[0]  # Rimuove dimensione batch
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

        # Normalizza tra 0 e 1
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1

        return heatmap

    def overlay_heatmap(self, heatmap, img_path, alpha=0.4):
        """
        Sovrappone la heatmap all'immagine originale.

        Args:
            heatmap: Heatmap normalizzata.
            img_path: Path dell'immagine originale.
            alpha: Peso della heatmap sulla sovrapposizione.

        Returns:
            Immagine con heatmap sovrapposta.
        """
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # Converti heatmap in colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

        # Sovrapposizione con alpha blending
        superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

        return superimposed_img
