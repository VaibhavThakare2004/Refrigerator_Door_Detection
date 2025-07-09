# utils.py
import os
import requests
import torch

def download_from_drive(file_id, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename} from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url, stream=True)
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"{filename} downloaded successfully.")
    else:
        print(f"{filename} already exists.")

def load_models():
    model_final_id = "1xFk-eTvI5IFPywu3XnVgII855vgzKzQ8"
    siamese_model_id = "1c_sl55Q1tt5YIG9S2ExhybYymDItRdS3"

    model_final_file = "door_detection_model.pth"
    siamese_model_file = "siamese_door_model.pth"

    download_from_drive(model_final_id, model_final_file)
    download_from_drive(siamese_model_id, siamese_model_file)

    model_main = torch.load(model_final_file, map_location=torch.device("cpu"))
    model_main.eval()

    siamese_model = torch.load(siamese_model_file, map_location=torch.device("cpu"))
    siamese_model.eval()

    return model_main, siamese_model
