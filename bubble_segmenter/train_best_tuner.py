from tensorflow import keras
import matplotlib.pyplot as plt

from data_load import get_tvt_split
from segmenting_model import get_unet_hypermodel, bce_dice_loss
from config import IMG_SIZE, EPOCHS_MAIN, LOAD, MODEL_NAME

# Dataset
train_dataset, validate_dataset = get_tvt_split("bubble_gun_images", IMG_SIZE)

# Model
if LOAD:
    model = keras.models.load_model(
        f"{MODEL_NAME}.keras",
        custom_objects={"bce_dice_loss": bce_dice_loss}
    )
else:
    tuner = get_unet_hypermodel(IMG_SIZE)
    best_hp = tuner.get_best_hyperparameters()[0]
    model = tuner.hypermodel.build(best_hp)

# Train the model and save it
history = model.fit(
    train_dataset,
    validation_data=validate_dataset,
    epochs=EPOCHS_MAIN,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)],
)
model.save(f"{MODEL_NAME}.keras")

# Graph plot
plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.ylim([0,1])
plt.savefig("results/history.png")
plt.close()