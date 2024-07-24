import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything.build_sam import sam_model_registry
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator

# define annotation
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


image_bgr = cv2.imread('../image.jpg')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# define model
# if you want simple, just write model
sam = sam_model_registry['default'](checkpoint='../sam_vit_h_4b8939.pth')
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)

masks = mask_generator.generate(image_rgb)
print(masks)

plt.figure(figsize=(10,10))
plt.imshow(image_rgb)
show_anns(masks)
plt.axis('off')
plt.show()