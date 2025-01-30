from tensorflow import keras as tfk
import tensorflow.keras.layers as tfkl
from config import CONFIG
from logging_utils import app_logger
import tensorflow.keras.models as tfkm


backbone_dict = {
    "efficientnetB2": [tfk.applications.EfficientNetB2, tfk.applications.efficientnet.preprocess_input],
    "Resnet50": [tfk.applications.ResNet50, tfk.applications.resnet.preprocess_input],
    "Resnet101V2": [tfk.applications.ResNet101V2, tfk.applications.resnet_v2.preprocess_input],
    "convnext_small": [tfk.applications.ConvNeXtSmall, tfk.applications.convnext.preprocess_input]
}

loss_dict = {
    "binary_crossentropy": tfk.losses.BinaryCrossentropy(),
    "categorical_crossentropy": tfk.losses.CategoricalCrossentropy(),
    "categorical_focal_crossentropy": tfk.losses.CategoricalFocalCrossentropy(alpha=CONFIG['data']['class_weights']),
    "sparse_categorical_crossentropy": tfk.losses.SparseCategoricalCrossentropy(from_logits=True),
}


lr = CONFIG['training']['learning_rate']
preprocess_function = backbone_dict[CONFIG['model']['backbone']][1]
metrics_train = ['accuracy', tfk.metrics.AUC(name="auc"),
                 tfk.metrics.Precision(),
                 tfk.metrics.Recall()]

if CONFIG['training']['loss_function'] in loss_dict.keys(): loss_fns = loss_dict[CONFIG['training']['loss_function']]
else: raise ValueError("Loss function not found in the loss dictionary.")


def build_model(backbone=backbone_dict[CONFIG['model']['backbone']][0],
                input_shape=CONFIG['model']['input_shape'],
                output_shape=CONFIG['model']['output_shape'],
                pooling='avg',
                output_activation='sigmoid' if CONFIG['model']['output_shape'] == 1 else 'softmax',
                loss_fn = loss_fns,
                optimizer=tfk.optimizers.AdamW(learning_rate=lr),
                metrics=metrics_train,
                preprocess_input=CONFIG['model']['preprocess_input'],
                plot=True):

    # Input layer
    inputs = tfk.Input(shape=input_shape, name='overall_input_layer')



    if preprocess_input: input_prep = preprocess_function(inputs)
    else: input_prep = inputs


    # Defining the backbone and calling it
    backbone = backbone(weights="imagenet",include_top=False,input_shape=(input_shape[0],input_shape[1],3),pooling=pooling)
    backbone.trainable = False
    for layer in backbone.layers[-10:]:
        layer.trainable = True

    x = backbone(input_prep)

    x = tfkl.Dropout(0.2, name='dropout')(x)
    x = tfkl.BatchNormalization(name='batch_norm')(x)

    x = tfkl.Dense(256, activation='relu', name='dense_1')(x)
    x = tfkl.Dropout(0.2, name='dropout_2')(x)
    x = tfkl.BatchNormalization(name='batch_norm_2')(x)

    outputs = tfkl.Dense(output_shape, activation=output_activation, name='output')(x)

    # Model definition and compiling
    tl_model = tfk.Model(inputs=inputs, outputs=outputs, name='model')
    tl_model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)
    app_logger.info("Model correctly compiled")
    if plot:
      tl_model.summary(expand_nested=True, show_trainable=True)

    return tl_model






import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras import Model

import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras import Model

class EnhancedCNN(Model):
    def __init__(self, input_shape, output_shape):
        super(EnhancedCNN, self).__init__()

        # 📌 **Feature Extraction (Convolutional Layers)**
        self.conv1 = tfkl.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.conv2 = tfkl.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.pool1 = tfkl.MaxPooling2D((2, 2))

        self.conv3 = tfkl.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.conv4 = tfkl.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool2 = tfkl.MaxPooling2D((2, 2))

        self.conv5 = tfkl.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.conv6 = tfkl.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.pool3 = tfkl.MaxPooling2D((2, 2))

        self.conv7 = tfkl.Conv2D(256, (3, 3), activation='relu', padding='same')
        self.conv8 = tfkl.Conv2D(256, (3, 3), activation='relu', padding='same')
        self.pool4 = tfkl.MaxPooling2D((2, 2))

        # 📌 **Flatten & Fully Connected Layers**
        self.flatten = tfkl.Flatten()
        self.dense1 = tfkl.Dense(512, activation='relu')
        self.dropout1 = tfkl.Dropout(0.5)
        self.dense2 = tfkl.Dense(256, activation='relu')
        self.dropout2 = tfkl.Dropout(0.5)
        self.output_layer = tfkl.Dense(output_shape, activation='sigmoid' if output_shape == 1 else 'softmax')

    def call(self, inputs, training=False):
        """ Definisce il forward pass del modello. """
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.pool4(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.output_layer(x)

    def build(self, input_shape):
        """ 📌 Costruisce il modello con un input shape corretto """
        super(EnhancedCNN, self).build(input_shape)




def build_enhanced_model(input_shape=CONFIG['model']['input_shape'],
                output_shape=CONFIG['model']['output_shape'],
                loss_fn=loss_fns,
                optimizer=tfk.optimizers.AdamW(learning_rate=CONFIG['training']['learning_rate']),
                metrics=metrics_train,
                plot=True):
    """
    Costruisce e compila l'Enhanced CNN.

    :param input_shape: Dimensione dell'input (es. (224, 224, 3)).
    :param output_shape: Numero di classi in output.
    :param loss_fn: Funzione di loss scelta.
    :param optimizer: Ottimizzatore (default: AdamW).
    :param metrics: Metriche da monitorare.
    :param plot: Se True, stampa il summary del modello.
    :return: Modello compilato pronto per il training.
    """

    model = EnhancedCNN(input_shape=input_shape, output_shape=output_shape)
    dummy_input = tf.keras.Input(shape=input_shape)
    _ = model(dummy_input)

    # Compila il modello
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)
    app_logger.info("Enhanced model correctly compiled")

    if plot:
        model.build((None, input_shape[0], input_shape[1], 3))
        model.summary(expand_nested=True, show_trainable=True)

    return model


if __name__ == '__main__':
    model = build_enhanced_model()
