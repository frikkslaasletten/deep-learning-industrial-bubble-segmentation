import torch
from PIL import Image
import numpy as np
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from torch.utils.data import DataLoader
import vid_to_images

# Config
# PROMPT: The prompt SAM3 uses to figure out what to segment
PROMPT = "translucent reflective white soapbubble bubble soap bubbles"
# MASK_AMOUNT: The limit on how many masks SAM3 can generates per image
MASK_AMOUNT = 1
#IMAGE_SIZE: The resolution of the input and output images
IMAGE_SIZE = (256,256)
#TARGET_FPS: How many frames should be extracted per second of the videos
TARGET_FPS = 3

# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model, confidence_threshold=0.01)

# Get input videos and turn them into a pytorch dataset
frames, vid_indexes = vid_to_images.video_to_frames("input_vids", resize=IMAGE_SIZE, target_fps=TARGET_FPS)
frame_amount = len(frames)
dl = DataLoader(frames, batch_size=1)

# Loop over the dataset
for frame_i, image in enumerate(dl):

    # Figures out which video the image belongs to
    cur_vid_index = -1
    for vid_index in vid_indexes:
        if vid_index > frame_i:
            cur_vid_index += 1

    # Clear gpu memory to prevent crashes
    torch.cuda.empty_cache()

    # Change dimensions for 
    image = image.permute(0, 3, 1, 2)
    image = image[0]

    # Feed image to model
    inference_state = processor.set_image(image)

    # Prompt the model with text
    output = processor.set_text_prompt(state=inference_state, prompt=PROMPT)

    # Get the masks, bounding boxes, and scores
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    sorted_indices = torch.argsort(scores, descending=True)

    # Reorder everything
    scores = scores[sorted_indices]
    masks = masks[sorted_indices]
    boxes = boxes[sorted_indices]

    print(f"Predicted masks: {len(masks)}")
    print(image.shape)

    # Format image to png
    image = image.permute(1,2,0)
    image = image.numpy()
    image = (image*255).astype(np.uint8)
    image = Image.fromarray(image)
    original_image = image
    image = image.convert("RGBA")

    for mask_i, mask in enumerate(masks):
        # Convert mask to image format
        mask_array = mask.squeeze().cpu().numpy()
        mask_img = Image.fromarray((mask_array * 255).astype(np.uint8))

        # Overlay mask on original image
        overlay = Image.new("RGBA", image.size, (0, 255, 0, 0))
        alpha = Image.fromarray((mask_array * 60).astype(np.uint8))
        overlay.putalpha(alpha)
        blended = Image.alpha_composite(image, overlay)

        # Save overlay and mask
        mask_img.save(f"/workspace/sam3/results/mask/mask_{cur_vid_index}_{frame_i}_{mask_i}.png")
        blended.save(f"/workspace/sam3/results/overlay/overlay_{cur_vid_index}_{frame_i}_{mask_i}.png")
        if mask_i >= MASK_AMOUNT-1:
            break
    # If mask was found, save original
    if len(masks) != 0:
        original_image.save(f"/workspace/sam3/results/original/original_{cur_vid_index}_{frame_i}.png")

    print(f"{frame_i} out of {frame_amount}")