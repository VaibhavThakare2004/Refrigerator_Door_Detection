import cv2
import os
import albumentations as A

# Folder with cropped door images (folder, not a file)
input_folder = r"C:\Users\vaibh\Desktop\Work\Intenship_Work\09-06-25\Detectron\cropped_doors_700"

# Output folder for augmented images
output_folder = "augmented_doors"
os.makedirs(output_folder, exist_ok=True)

# Augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7),
    A.GaussNoise(p=0.2)
])

num_augmentations = 5

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png'))]

for img_file in image_files:
    img_path = os.path.join(input_folder, img_file)
    image = cv2.imread(img_path)
    basename = os.path.splitext(img_file)[0]

    # Save original image (optional)
    cv2.imwrite(os.path.join(output_folder, f"{basename}_orig.jpg"), image)

    # Create augmented versions
    for i in range(num_augmentations):
        augmented = transform(image=image)
        aug_image = augmented['image']
        out_path = os.path.join(output_folder, f"{basename}_aug_{i+1}.jpg")
        cv2.imwrite(out_path, aug_image)

print("Augmentation done.")
