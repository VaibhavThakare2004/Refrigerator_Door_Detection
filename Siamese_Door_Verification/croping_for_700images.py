import os
import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# -------- CONFIGURE MODEL --------
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only "door"
cfg.MODEL.WEIGHTS = r"C:\Users\vaibh\Desktop\Work\Intenship_Work\09-06-25\Detectron\output\model_final.pth"  # path to your trained model_final.pth
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # confidence threshold
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)

# -------- INPUT & OUTPUT FOLDERS --------
input_folder = r"C:\Users\vaibh\Desktop\Work\Intenship_Work\09-06-25\Detectron\all_images_door"
output_folder = r"C:\Users\vaibh\Desktop\Work\Intenship_Work\09-06-25\Detectron\cropped_doors_700"
os.makedirs(output_folder, exist_ok=True)

# -------- PROCESS IMAGES --------
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for img_file in image_files:
    img_path = os.path.join(input_folder, img_file)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image {img_file}")
        continue

    outputs = predictor(img)
    instances = outputs["instances"]

    if not instances.has("pred_boxes") or len(instances) == 0:
        print(f"No doors detected in {img_file}")
        continue

    pred_boxes = instances.pred_boxes
    scores = instances.scores

    for i, box in enumerate(pred_boxes):
        score = scores[i].item()
        if score < 0.5:
            continue
        
        x1, y1, x2, y2 = box.int().tolist()
        cropped = img[y1:y2, x1:x2]

        base_name, ext = os.path.splitext(img_file)
        save_path = os.path.join(output_folder, f"{base_name}_door{i+1}{ext}")

        cv2.imwrite(save_path, cropped)
        print(f"Saved cropped door to {save_path}")

print("Door cropping completed!")
