import keras as tfk
from keras import layers as tfkl

from config import CONFIG
from logging_utils import app_logger

backbone_dict = {
    "efficientnetB2": [tfk.applications.EfficientNetB2, tfk.applications.efficientnet.preprocess_input],
    "Resnet50": [tfk.applications.ResNet50, tfk.applications.resnet.preprocess_input],
    "Resnet101V2": [tfk.applications.ResNet101V2, tfk.applications.resnet_v2.preprocess_input],
    "ConvNextSmall": [tfk.applications.ConvNeXtSmall, tfk.applications.convnext.preprocess_input]
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

if CONFIG['training']['loss_function'] in loss_dict.keys():
    loss_fns = loss_dict[CONFIG['training']['loss_function']]
else:
    raise ValueError("Loss function not found in the loss dictionary.")



# tfkl.Resizing(224, 224, name="resize"),  # Resize images to 224x224
# tfkl.RandomShear(0.2, name="random_shear"),  # Apply shear transformation

def get_augmentation_pipeline():
    return tfk.Sequential([
        tfkl.RandomFlip("horizontal_and_vertical", name="random_flip"),  # Flip images both horizontally and vertically
        tfkl.RandomRotation(0.2, name="random_rotation"),  # Randomly rotate images (20% of 360 degrees)
    ], name="augmentation_pipeline")





def build_model(backbone=backbone_dict[CONFIG['model']['backbone']][0],
                augmentation=get_augmentation_pipeline(),
                input_shape=CONFIG['model']['input_shape'],
                output_shape=1,
                pooling='avg',
                output_activation='sigmoid',
                loss_fn = loss_fns,
                optimizer=tfk.optimizers.AdamW(lr),
                metrics=metrics_train,
                preprocess_input=CONFIG['model']['preprocess_input'],
                seed=CONFIG['general']['seed'],
                plot=False):


    # Input layer
    inputs = tfk.Input(shape=input_shape, name='input_layer')

    if preprocess_input:
        input_prep = preprocess_function(inputs)
    else:
        input_prep = inputs


    # Augmentation directly integrated as layers
    # augmented = tfkl.RandomFlip("horizontal_and_vertical", name="random_flip",seed=seed)(input_prep)  # Random flips
    # augmented = tfkl.RandomRotation(0.3, name="random_rotation",seed=seed)(augmented)  # Random rotations
    # augmented = tfkl.RandomShear(x_factor=0.3,y_factor=0.3,seed=seed)(augmented) # Random Shear
    # augmented = tfkl.RandomSharpness(0.3, value_range=(0, 255), name="random_sharpness",seed=seed)(augmented)
    augmented = inputs


    # Defining the backbone and calling it
    backbone = backbone(weights="imagenet",
                        include_top=False,
                        input_shape=(input_shape[0],input_shape[1],3),
                        pooling=pooling)
    backbone.trainable = False
    for layer in backbone.layers[-30:]:
        layer.trainable = True


    x = backbone(augmented)

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






if __name__ == '__main__':
    model = build_model()
