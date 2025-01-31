import tensorflow as tf
from tensorflow import keras as tfk
import tensorflow.keras.layers as tfkl
from config import CONFIG
from logging_utils import app_logger
from tensorflow.keras import layers, models

backbone_dict = {
    "efficientnetB2": [tfk.applications.EfficientNetB2, tfk.applications.efficientnet.preprocess_input],
    "resnet50": [tfk.applications.ResNet50, tfk.applications.resnet.preprocess_input],
    "Resnet101V2": [tfk.applications.ResNet101V2, tfk.applications.resnet_v2.preprocess_input],
    "convnext_small": [tfk.applications.ConvNeXtSmall, tfk.applications.convnext.preprocess_input],
    "resnet50v2": [tfk.applications.ResNet50V2, tfk.applications.resnet_v2.preprocess_input],
    "resnet101": [tfk.applications.ResNet101, tfk.applications.resnet.preprocess_input],

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


def build_model_2(backbone=backbone_dict[CONFIG['model']['backbone']][0],
                input_shape=CONFIG['model']['input_shape'],
                output_shape=CONFIG['model']['output_shape'],
                pooling=None,
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
    x = tfkl.Conv2D(256, 1, activation='relu', name='conv_adapt')(x)
    x = pyramid_pooling_module(x)
    x = tfkl.GlobalAveragePooling2D()(x)

    outputs = tfkl.Dense(output_shape, activation=output_activation, name='output')(x)

    # Model definition and compiling
    tl_model = tfk.Model(inputs=inputs, outputs=outputs, name='model')
    tl_model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)
    app_logger.info("Model correctly compiled")
    if plot:
      tl_model.summary(expand_nested=True, show_trainable=True)

    return tl_model



# Residual Block
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(x) if stride > 1 or x.shape[-1] != filters else x
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

# Pyramid Pooling Module (PPM)

def pyramid_pooling_module(x, pool_sizes=[1,4,7]):
    shape = tf.keras.backend.int_shape(x)  # Ottieni la forma di input
    height, width = shape[1], shape[2]  # Altezza e larghezza dello spazio

    pooled_outputs = [x]
    print(f"Shape prima di MaxPooling (pool_size={pool_sizes[0]}): {x.shape}")

    for size in pool_sizes:
        # Pooling
        pooled = layers.MaxPooling2D(pool_size=(height // size, width // size),
                                         strides=(height // size, width // size), padding='same')(x)
        pooled = layers.Conv2D(256, 1, activation='relu')(pooled)
        print(f"Shape dopo MaxPooling (pool_size={size}): {pooled.shape}")
        # Upsampling con Conv2DTranspose
        pooled = layers.Conv2DTranspose(512, kernel_size=3, strides=(height // size, width // size), padding='same')(
            pooled)
        print(f"Shape dopo Conv2D (512, 1): {pooled.shape}")  # Log della shape dopo la convoluzione

        # Aggiungi il risultato alla lista
        pooled_outputs.append(pooled)
    # Concatenazione dei risultati
    return layers.Concatenate()(pooled_outputs)



# MResNet Model
def MResNet(input_shape=CONFIG['model']['input_shape'],
            num_classes=CONFIG['model']['output_shape'],
            output_activation='sigmoid' if CONFIG['model']['output_shape'] == 1 else 'softmax',
            loss_fn = loss_fns,
            metrics=metrics_train,
            optimizer=tfk.optimizers.AdamW(learning_rate=lr),):

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    # Residual Blocks
    for filters in [64, 128, 256, 512]:
        x = residual_block(x, filters)

    # PPM Module
    x = pyramid_pooling_module(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation=output_activation)(x)

    model = models.Model(inputs, outputs)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)
    app_logger.info("Model correctly compiled")

    return model


# Create model


if __name__ == '__main__':
    # model = build_model()
    # model = MResNet()
    # model.summary()
    model = build_model_2()
    # l = model.get_layer(CONFIG["model"]["backbone"])
    # for layer in l.layers:
    #     layer.trainable = True
    # model.summary()