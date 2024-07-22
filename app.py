from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import supervision as sv

IMAGE_PATH = './image.jpg'
image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# define model
sam = sam_model_registry["default"](checkpoint="./sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)

# get result
sam_result = mask_generator.generate(image_rgb)
print(sam_result[0].keys())

# show result (colored)
mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
detections = sv.Detections.from_sam(sam_result=sam_result)
annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
sv.plot_images_grid(
    images=[image_bgr, annotated_image],
    grid_size=(1, 2),
    titles=['source image', 'segmented image']
)

## show result (boolean)
# masks = [
#     mask['segmentation']
#     for mask
#     in sorted(sam_result, key=lambda x: x['area'], reverse=True)
# ]
# sv.plot_images_grid(
#     images=masks,
#     grid_size=(32, int(len(masks) / 8)),
#     size=(16, 16)
# )

