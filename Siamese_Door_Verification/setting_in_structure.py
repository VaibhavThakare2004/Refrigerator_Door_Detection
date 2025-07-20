import os
import shutil

# Path where all your images currently are (original + augmented)
source_folder = r"C:\Users\vaibh\Desktop\Work\Intenship_Work\09-06-25\Detectron\augmented_doors"

# Path where you want to create organized dataset folders by door ID
target_folder = r"C:\Users\vaibh\Desktop\Work\Intenship_Work\09-06-25\Detectron\door_dataset"

os.makedirs(target_folder, exist_ok=True)

# List all image files in the source folder
image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.png'))]

for filename in image_files:
    # Extract door ID from filename:
    # This example assumes filename format: door_001_orig.jpg or door_001_aug_1.jpg
    # You can adjust splitting logic if your naming is different
    door_id = filename.split('_')[0] + '_' + filename.split('_')[1]  # e.g. "door_001"
    
    # Create folder for this door inside target_folder if not exists
    door_folder = os.path.join(target_folder, door_id)
    os.makedirs(door_folder, exist_ok=True)
    
    # Move the image to the door's folder
    src_path = os.path.join(source_folder, filename)
    dst_path = os.path.join(door_folder, filename)
    
    # You can use shutil.move to move or shutil.copy2 to copy (copy recommended to keep original files)
    shutil.copy2(src_path, dst_path)

print("Images organized into folders by door ID.")
