from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

from data_load import get_tvt_split
from config import IMG_SIZE, MODEL_NAME
from segmenting_model import bce_dice_loss

# Dataset
tr, va = get_tvt_split("gassco_images", IMG_SIZE, augment=False)
dataset = tr.concatenate(va).unbatch().batch(1)

# Load model
model = keras.models.load_model(
    f"{MODEL_NAME}.keras",
    custom_objects={"bce_dice_loss": bce_dice_loss}
)

# Result analysis
model.evaluate(dataset)

# Overlay plot
results = model.predict(dataset)
for i, (result, (image, _)) in enumerate(zip(results, dataset)):
    mask = (result[..., 0] > 0.5).astype(np.uint8)
    img = image.numpy().copy().squeeze()
    img = (img * 255).astype(np.uint8)
    binary_mask = mask.astype(bool)
    overlay = img.copy()
    overlay[binary_mask] = [0, 255, 0]
    plt.figure()
    plt.imshow(img)
    plt.imshow(overlay, alpha=0.3)
    plt.axis("off")
    plt.title("Segmentation Overlay")
    plt.savefig(f"results/overlay{i}.png")
    plt.close()


