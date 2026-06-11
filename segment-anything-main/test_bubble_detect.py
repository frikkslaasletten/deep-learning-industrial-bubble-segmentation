import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

# Paths
image_path = r"BILDE ADRESSE"
sam_checkpoint = sam_checkpoint = r".\sam_vit_h_4b8939.pth"

# Load image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam.to(device)

predictor = SamPredictor(sam)
predictor.set_image(image)

# POINT INSIDE THE OBJECT (x, y)
input_point = np.array([[426, 377]])
input_label = np.array([1])  # 1 = foreground

masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

# Pick best mask (highest score)
best_mask = masks[scores.argmax()]

# Visualize
plt.imshow(image)
plt.imshow(best_mask, alpha=0.5)
plt.axis("off")
plt.show()