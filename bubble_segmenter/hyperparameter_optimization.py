from tensorflow import keras

from data_load import get_tvt_split
from segmenting_model import get_unet_hypermodel
from config import IMG_SIZE, EPOCHS_HP, OVERWRITE

# Datasets
train_dataset, validate_dataset = get_tvt_split("bubble_gun_images", IMG_SIZE)

# Get tuner
tuner = get_unet_hypermodel(IMG_SIZE, overwrite=OVERWRITE)

# Search for best hyperparameters
history = tuner.search(
    train_dataset,
    validation_data=validate_dataset,
    epochs=EPOCHS_HP,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)],
)