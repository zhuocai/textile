"""
Model inspired by: https://github.com/natasasdj/anomalyDetection
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    LeakyReLU,
    Activation,
    concatenate,
    Flatten,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

# Preprocessing parameters
RESCALE = 1.0 / 255
SHAPE = (256, 256)
PREPROCESSING_FUNCTION = None
PREPROCESSING = None
VMIN = 0.0
VMAX = 1.0
DYNAMIC_RANGE = VMAX - VMIN


def build_model(color_mode):
    # set channels
    if color_mode == "grayscale":
        channels = 1
    elif color_mode == "rgb":
        channels = 3
    img_dim = (*SHAPE, channels)

    # input
    input_img = Input(shape=img_dim)

    # encoder
    encoding_dim = 64  # 128
    # layer1
    x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(
        input_img
    )
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)


    # layer2
    # added ---------------------------------------------------------------------------
    x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x1 = x
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    # ---------------------------------------------------------------------------------


    # layer3
    x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    # layer4
    # added ---------------------------------------------------------------------------
    x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    # ---------------------------------------------------------------------------------

    # layer5
    x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    # layer6
    # added ---------------------------------------------------------------------------
    x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    # ---------------------------------------------------------------------------------

    x = Flatten()(x)
    x = Dense(encoding_dim, kernel_regularizer=regularizers.l2(1e-6))(x)
    x = LeakyReLU(alpha=0.1)(x)
    # encoded = x

    # decoder
    x = Reshape((4, 4, encoding_dim // 16))(x)
    x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)

    ## added ---------------------------------------------------------------------------
    x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)
    # ---------------------------------------------------------------------------------

    x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)

    ## added ---------------------------------------------------------------------------
    x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)
    # ---------------------------------------------------------------------------------

    x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    #x2 = x
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)

    ## added ---------------------------------------------------------------------------
    x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x2 = x
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)
    # ---------------------------------------------------------------------------------

    x = Conv2D(
        img_dim[2], (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6)
    )(x)
    x = BatchNormalization()(x)
    x = Activation("sigmoid")(x)
    decoded = x

    output = x1 - x2
    # model
    # autoencoder = Model(input_img, decoded)
    autoencoder = Model(input_img, output)
    return autoencoder
