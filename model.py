import tensorflow as tf
import keras as tfk
from keras import layers as tfkl

from config import CONFIG
from logging_utils import app_logger

backbone_dict = {
    "efficientnetB2": tfk.applications.EfficientNetB2,
    "Resnet50": tfk.applications.ResNet50,
    "Resnet101V2": tfk.applications.ResNet101V2,
    "ConvNextSmall": tfk.applications.ConvNeXtSmall,
}

lr = CONFIG['training']['learning_rate']
# input_shape = CONFIG['model']['input_shape']

def build_model(backbone=backbone_dict["ConvNextSmall"],
                augmentation=None,
                input_shape=CONFIG['model']['input_shape'],
                output_shape=1,
                pooling='avg',
                loss_fn = tfk.losses.BinaryCrossentropy(),
                optimizer=tfk.optimizers.AdamW(lr),
                metrics=['accuracy'],
                plot=True):


    # Input layer
    inputs = tfk.Input(shape=input_shape, name='input_layer')
    back_adapt = tfkl.Conv2D(3, (3,3), padding='same')(inputs)
    back_adapt = tfkl.LayerNormalization()(back_adapt)


    # Defining the backbone and calling it
    backbone = backbone(weights="imagenet",
                        include_top=False,
                        input_shape=(input_shape[0],input_shape[1],3),
                        pooling=pooling)
    backbone.trainable = False
    x = backbone(back_adapt)

    x = tfkl.Dropout(0.3, name='dropout')(x)
    x = tfkl.BatchNormalization(name='batch_norm')(x)

    x = tfkl.Dense(256, activation='relu', name='dense_1')(x)
    x = tfkl.Dropout(0.3, name='dropout_2')(x)
    x = tfkl.BatchNormalization(name='batch_norm_2')(x)

    outputs = tfkl.Dense(output_shape, activation='sigmoid', name='dense')(x)

    # Model definition and compiling
    tl_model = tfk.Model(inputs=inputs, outputs=outputs, name='model')
    tl_model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)
    app_logger.info("Model correctly compiled")

    if plot:
      tl_model.summary(expand_nested=True, show_trainable=True)

    return tl_model







if __name__ == '__main__':
    model = build_model()
