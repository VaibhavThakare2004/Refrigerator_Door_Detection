from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
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
from pathlib import Path
import zipfile
import io

app = FastAPI()

# Configuration
UPLOAD_FOLDER = 'static/uploads'
CROP_FOLDER = 'static/crops'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROP_FOLDER, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load models at startup
@app.on_event("startup")
async def load_models():
    global detection_predictor, verification_model, device
    
    # Load Detectron2 model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = "model_final.pth"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    detection_predictor = DefaultPredictor(cfg)
    
    # Load Siamese model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verification_model = SiameseNetwork().to(device)
    verification_model.load_state_dict(torch.load("siamese_door_model.pth", map_location=device))
    verification_model.eval()

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

# Image transformation for verification
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/detection", response_class=HTMLResponse)
async def detection_page(request: Request):
    return templates.TemplateResponse("detection.html", {"request": request})

@app.get("/verification", response_class=HTMLResponse)
async def verification_page(request: Request):
    return templates.TemplateResponse("verification.html", {"request": request})

@app.get("/download-crops")
async def download_crops():
    # Create a zip file in memory
    zip_filename = "detected_doors.zip"
    memory_file = io.BytesIO()
    
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for file in os.listdir(CROP_FOLDER):
            file_path = os.path.join(CROP_FOLDER, file)
            if os.path.isfile(file_path):
                zf.write(file_path, file)
    
    memory_file.seek(0)
    return FileResponse(
        memory_file,
        media_type="application/zip",
        filename=zip_filename
    )

@app.post("/detect", response_class=HTMLResponse)
async def detect_doors(
    request: Request,
    file: UploadFile = File(...),
    threshold: float = Form(0.7)
):
    # Validate threshold
    if not 0.5 <= threshold <= 0.95:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": "Threshold must be between 0.5 and 0.95"
        })
    
    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Read image
    image = cv2.imread(file_path)
    if image is None:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": "Invalid image file"
        })
    
    # Clear previous crops
    for f in os.listdir(CROP_FOLDER):
        os.remove(os.path.join(CROP_FOLDER, f))
    
    # Detect doors
    outputs = detection_predictor(image)
    instances = outputs["instances"]
    boxes = instances.pred_boxes.tensor.cpu().numpy()
    scores = instances.scores.cpu().numpy()
    
    # Crop and save detected doors (resized to 300x300)
    detected_doors = []
    for i, box in enumerate(boxes):
        if scores[i] >= threshold:
            x1, y1, x2, y2 = box.astype(int)
            crop = image[y1:y2, x1:x2]
            
            # Resize the crop to make it smaller and consistent
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
        "original_image": file.filename,
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
    # Save uploaded files
    file1_path = os.path.join(UPLOAD_FOLDER, secure_filename(file1.filename))
    file2_path = os.path.join(UPLOAD_FOLDER, secure_filename(file2.filename))
    
    with open(file1_path, "wb") as buffer:
        shutil.copyfileobj(file1.file, buffer)
    with open(file2_path, "wb") as buffer:
        shutil.copyfileobj(file2.file, buffer)
    
    # Verify that the second image is from the crops folder
    if not file2.filename.startswith("door_crop_"):
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": "Second image must be one of the detected door crops"
        })
    
    # Load and transform images
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
    
    # Get embeddings
    with torch.no_grad():
        out1, out2 = verification_model(img1, img2)
        dist = F.pairwise_distance(out1, out2)
    
    # Fixed threshold of 0.85
    threshold = 0.85
    is_same = dist.item() < threshold
    
    return templates.TemplateResponse("verification_results.html", {
        "request": request,
        "image1": file1.filename,
        "image2": file2.filename,
        "distance": float(dist.item()),
        "is_same_door": is_same,
        "threshold": threshold
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


