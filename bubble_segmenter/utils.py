import numpy as np
import matplotlib.pyplot as plt

def save_dataset(dataset, folder, name):
    for i, (image, mask) in enumerate(dataset):
        # Only takes the first element from each batch
        mask = (mask[0].numpy().squeeze() > 0.5).astype(np.uint8)
        img = image[0].numpy().copy().squeeze()
        img = (img * 255).astype(np.uint8)
        binary_mask = mask.astype(bool)
        overlay = img.copy()
        overlay[binary_mask] = [0, 255, 0]
        plt.figure()
        plt.imshow(img)
        plt.imshow(overlay, alpha=0.3)
        plt.axis("off")
        plt.title("Segmentation Overlay")
        plt.savefig(f"{folder}/{name}{i}.png")
        plt.close()