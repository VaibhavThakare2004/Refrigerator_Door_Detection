from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import torch
import os
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
import shutil
from werkzeug.utils import secure_filename

app = FastAPI()

# Configuration
UPLOAD_FOLDER = 'static/uploads'
CROP_FOLDER = 'static/crops'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROP_FOLDER, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Siamese Network definition
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

    def forward_once(self, x):
        output = self.backbone(x)
        output = self.fc(output)
        output = F.normalize(output, p=2, dim=1)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.on_event("startup")
async def load_models():
    global detection_predictor, verification_model, device
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = "door_detection_model.pth"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    detection_predictor = DefaultPredictor(cfg)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verification_model = SiameseNetwork().to(device)
    verification_model.load_state_dict(torch.load("siamese_door_model.pth", map_location=device))
    verification_model.eval()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/detection", response_class=HTMLResponse)
async def detection_page(request: Request):
    return templates.TemplateResponse("detection.html", {"request": request})

@app.get("/verification", response_class=HTMLResponse)
async def verification_page(request: Request):
    return templates.TemplateResponse("verification.html", {"request": request})

@app.post("/detect", response_class=HTMLResponse)
async def detect_doors(
    request: Request,
    file: UploadFile = File(...),
    threshold: float = Form(0.7)
):
    if not 0.5 <= threshold <= 0.95:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": "Threshold must be between 0.5 and 0.95"
        })
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = cv2.imread(file_path)
    if image is None:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": "Invalid image file"
        })

    for f in os.listdir(CROP_FOLDER):
        os.remove(os.path.join(CROP_FOLDER, f))

    outputs = detection_predictor(image)
    instances = outputs["instances"]
    boxes = instances.pred_boxes.tensor.cpu().numpy()
    scores = instances.scores.cpu().numpy()

    detected_doors = []
    for i, box in enumerate(boxes):
        if scores[i] >= threshold:
            x1, y1, x2, y2 = box.astype(int)
            crop = image[y1:y2, x1:x2]
            crop = cv2.resize(crop, (300, 300))
            crop_filename = f"door_crop_{i+1}.jpg"
            crop_path = os.path.join(CROP_FOLDER, crop_filename)
            cv2.imwrite(crop_path, crop)
            detected_doors.append({
                "filename": crop_filename,
                "confidence": float(scores[i]),
                "box": [int(x1), int(y1), int(x2), int(y2)]
            })

    return templates.TemplateResponse("detection_results.html", {
        "request": request,
        "original_image": filename,
        "detected_doors": detected_doors,
        "count": len(detected_doors),
        "threshold": threshold
    })

@app.post("/verify", response_class=HTMLResponse)
async def verify_doors(
    request: Request,
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    filename1 = secure_filename(file1.filename)
    filename2 = secure_filename(file2.filename)

    file1_path = os.path.join(UPLOAD_FOLDER, filename1)
    file2_path = os.path.join(CROP_FOLDER, filename2)

    with open(file1_path, "wb") as buffer:
        shutil.copyfileobj(file1.file, buffer)

    with open(file2_path, "wb") as buffer:
        shutil.copyfileobj(file2.file, buffer)

    if not filename2.startswith("door_crop_"):
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": "Second image must be one of the detected door crops"
        })

    def load_image(image_path):
        img = Image.open(image_path).convert("RGB")
        return transform(img).unsqueeze(0).to(device)

    try:
        img1 = load_image(file1_path)
        img2 = load_image(file2_path)
    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": f"Error loading images: {str(e)}"
        })

    with torch.no_grad():
        out1, out2 = verification_model(img1, img2)
        dist = F.pairwise_distance(out1, out2)

    threshold = 0.85
    is_same = dist.item() < threshold

    return templates.TemplateResponse("verification_results.html", {
        "request": request,
        "image1": filename1,
        "image2": filename2,
        "distance": float(dist.item()),
        "is_same_door": is_same,
        "threshold": threshold
    })

@app.get("/download-crops", response_class=HTMLResponse)
async def list_crops_for_download(request: Request):
    try:
        files = [
            f for f in os.listdir(CROP_FOLDER)
            if os.path.isfile(os.path.join(CROP_FOLDER, f))
        ]
        return templates.TemplateResponse("download_crops.html", {
            "request": request,
            "files": files
        })
    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": f"Error listing crops: {str(e)}"
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
