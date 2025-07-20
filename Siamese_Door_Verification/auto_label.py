import torch
import cv2
import os
import json
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

# COCO class ID for refrigerator = 72
TARGET_CLASS_ID = 72

# Load pretrained model (Faster R-CNN)
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # adjust threshold
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

predictor = DefaultPredictor(cfg)

image_dir = "all_images_door/"
image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))]

annotations = []
images = []

image_id = 0
ann_id = 0

for file in image_files:
    path = os.path.join(image_dir, file)
    image = cv2.imread(path)
    outputs = predictor(image)

    boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    classes = outputs["instances"].pred_classes.cpu().numpy()

    h, w = image.shape[:2]
    images.append({
        "id": image_id,
        "file_name": file,
        "width": w,
        "height": h
    })

    for i, cls in enumerate(classes):
        if cls == TARGET_CLASS_ID:
            x1, y1, x2, y2 = boxes[i]
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,  # Custom ID for "door"
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "area": float((x2 - x1) * (y2 - y1)),
                "iscrowd": 0
            })
            ann_id += 1

    image_id += 1

# Final COCO-style dictionary
coco_json = {
    "images": images,
    "annotations": annotations,
    "categories": [{"id": 1, "name": "door"}]
}

# Save
with open("auto_labels.json", "w") as f:
    json.dump(coco_json, f, indent=4)
