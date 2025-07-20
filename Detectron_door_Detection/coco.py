import labelme2coco
import os

# Input folder where all .json and images are saved
labelme_folder = r"C:\Users\vaibh\Desktop\Work\Intenship_Work\17-06-25\all_images_door"

# Output COCO JSON path
output_json = r"C:\Users\vaibh\Desktop\Work\Intenship_Work\17-06-25\all_images_door_coco\dataset.json"

# Run conversion
labelme2coco.convert(labelme_folder, output_json)

print("✅ Conversion complete. COCO JSON saved.")
