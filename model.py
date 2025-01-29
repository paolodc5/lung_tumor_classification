from tensorflow import keras as tfk
import tensorflow.keras.layers as tfkl
from config import CONFIG
from logging_utils import app_logger


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
metrics_train = ['accuracy', tfk.metrics.AUC(name="auc"), tfk.metrics.Precision(), tfk.metrics.Recall()]

if CONFIG['training']['loss_function'] in loss_dict.keys(): loss_fns = loss_dict[CONFIG['training']['loss_function']]
else: raise ValueError("Loss function not found in the loss dictionary.")


def build_model(backbone=backbone_dict[CONFIG['model']['backbone']][0],
                input_shape=CONFIG['model']['input_shape'],
                output_shape=CONFIG['model']['output_shape'],
                pooling='avg',
                output_activation='sigmoid',
                loss_fn = loss_fns,
                optimizer=tfk.optimizers.AdamW(lr),
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

    x = tfkl.Dropout(0.3, name='dropout')(x)
    x = tfkl.BatchNormalization(name='batch_norm')(x)

    x = tfkl.Dense(256, activation='relu', name='dense_1')(x)
    x = tfkl.Dropout(0.3, name='dropout_2')(x)
    x = tfkl.BatchNormalization(name='batch_norm_2')(x)

    outputs = tfkl.Dense(output_shape, activation=output_activation, name='output')(x)

    # Model definition and compiling
    tl_model = tfk.Model(inputs=inputs, outputs=outputs, name='model')
    tl_model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)
    app_logger.info("Model correctly compiled")
    if plot:
      tl_model.summary(expand_nested=True, show_trainable=True)

    return tl_model





class CustomModel(tfk.Model):
    def __init__(self, backbone, input_shape, output_shape, preprocess_function, pooling='avg',
                 output_activation='sigmoid', seed=42):
        super(CustomModel, self).__init__()

        # Salviamo i parametri principali
        self.preprocess_function = preprocess_function
        self.seed = seed

        # Data augmentation layers
        self.augmentation = tfk.Sequential([
            tfkl.RandomFlip("horizontal_and_vertical", name="random_flip", seed=seed),
            tfkl.RandomRotation(0.3, name="random_rotation", seed=seed),
            tfkl.RandomSharpness(0.3, value_range=(0, 255), name="random_sharpness", seed=seed)
        ], name="augmentation_pipeline")

        # Backbone (base model)
        self.backbone = backbone(weights="imagenet",
                                 include_top=False,
                                 input_shape=(input_shape[0], input_shape[1], 3),
                                 pooling=pooling)
        self.backbone.trainable = False
        for layer in self.backbone.layers[-30:]:
            layer.trainable = True

        # Additional layers after the backbone
        self.dropout1 = tfkl.Dropout(0.3, name='dropout_1')
        self.batch_norm1 = tfkl.BatchNormalization(name='batch_norm_1')
        self.dense1 = tfkl.Dense(256, activation='relu', name='dense_1')

        self.dropout2 = tfkl.Dropout(0.3, name='dropout_2')
        self.batch_norm2 = tfkl.BatchNormalization(name='batch_norm_2')
        self.output_layer = tfkl.Dense(output_shape, activation=output_activation, name='output')

    def call(self, inputs, training=False):
        # Preprocessing step
        if self.preprocess_function:
            x = self.preprocess_function(inputs)
        else:
            x = inputs

        # Augmentation (applied during training only)
        if training:
            x = self.augmentation(x)

        # Backbone processing
        x = self.backbone(x, training=training)

        # Additional layers
        x = self.dropout1(x, training=training)
        x = self.batch_norm1(x, training=training)
        x = self.dense1(x)

        x = self.dropout2(x, training=training)
        x = self.batch_norm2(x, training=training)
        outputs = self.output_layer(x)

        return outputs






if __name__ == '__main__':
    model = build_model()
