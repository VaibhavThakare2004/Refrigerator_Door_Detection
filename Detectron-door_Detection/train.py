import os
import torch
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo

# 🧠 Step 1: Register COCO Dataset
register_coco_instances(
    "door_dataset",
    {},
    r"C:\Users\vaibh\Desktop\Work\Intenship_Work\17-06-25\all_images_door_coco\dataset.json\dataset.json",
    r"C:/Users/vaibh/Desktop/Work/Intenship_Work/17-06-25/all_images_door"
)

# 🛠️ Step 2: Set up Configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("door_dataset",)
cfg.DATASETS.TEST = ()  # No test set
cfg.DATALOADER.NUM_WORKERS = 0

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Use pre-trained weights
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000    # Change depending on dataset size
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only "door"

# Save output model and logs here
cfg.OUTPUT_DIR = "./output_door"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# 🚀 Step 3: Train
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
