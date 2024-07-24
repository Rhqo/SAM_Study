import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from segment_anything.build_sam import sam_model_registry
from segment_anything.predictor import SamPredictor

IMAGE_PATH = '../image.jpg'
image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# define model & target
sam = sam_model_registry["default"](checkpoint="../sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)
predictor.set_image(image_bgr)

# define box, mask
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


input_boxes = torch.tensor([
    [100, 140, 320, 310],
    [80, 80, 150, 135],
], device=predictor.device)

transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image_rgb.shape[:2])

# note several name changed; predict -> predict_torch, box -> boxes
masks, _, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
)

print(masks.shape)

plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
for mask in masks:
    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
for box in input_boxes:
    show_box(box.cpu().numpy(), plt.gca())
plt.axis('off')
plt.show()