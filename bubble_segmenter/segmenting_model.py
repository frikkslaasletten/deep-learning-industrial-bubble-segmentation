from tensorflow.keras import layers
from tensorflow import keras
import keras_tuner as kt
from config import *

def bce_dice_loss(y_true, y_pred):
    return keras.losses.BinaryCrossentropy()(y_true, y_pred) + keras.losses.Dice()(y_true, y_pred)

def conv_block(x, size):
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(size, 3, padding="same", use_bias=False)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(size, 3, padding="same", use_bias=False)(x)
    return x

def get_unet_hypermodel(img_size, overwrite=False):
    tuner = kt.BayesianOptimization(
        UnetHypermodel(img_size),
        objective="val_loss",
        max_trials=20,
        executions_per_trial=2,
        directory="keras_tuner",
        overwrite=overwrite,
        project_name=PROJECT_NAME
    )
    return tuner

class UnetHypermodel(kt.HyperModel):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size

    def build(self, hp):
        # Setup
        block_amount = hp.Int(
            name="block_amount",
            min_value=BLOCK_AMOUNT_MIN,
            max_value=BLOCK_AMOUNT_MAX,
            step=BLOCK_AMOUNT_STEP)
        base_channel_amount = hp.Int(
            name="base_channel_amount",
            min_value=BLOCK_CHANNEL_AMOUNT_MIN,
            max_value=BLOCK_CHANNEL_AMOUNT_MAX,
            step=BLOCK_CHANNEL_AMOUNT_STEP)
        channels_list = [base_channel_amount * 2**i for i in range(block_amount)]
        residuals = []

        # Input
        inputs = keras.Input(shape=(self.img_size, self.img_size, 3,))
        x = layers.Conv2D(channels_list[0], 3, activation='relu', padding='same')(inputs)

        # Downscaling
        for size in channels_list:
            x = conv_block(x, size)
            residuals.append(x)
            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Bottleneck
        x = conv_block(x, channels_list[-1] * 2)
        x = keras.layers.SpatialDropout2D(hp.Int(
            name="dropout",
            min_value=int(DROPOUT_MIN*10),
            max_value=int(DROPOUT_MAX*10),
            step=int(DROPOUT_STEP*10))/10.0)(x)
        residuals.reverse()

        # Upscaling
        for i, size in enumerate(reversed(channels_list)):
            x = keras.layers.Conv2DTranspose(size, 3, strides=2, padding="same")(x)
            residual = residuals[i]
            x = layers.Concatenate()([x, residual])
            x = conv_block(x, size)
        outputs = layers.Conv2D(1, 1, activation="sigmoid", padding="same")(x)

        # Model
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer="adam",
            loss=bce_dice_loss,
            metrics=[keras.metrics.BinaryIoU()]
        )
        return model