import os
import random
import itertools
import csv

door_dataset_folder = r"C:\Users\vaibh\Desktop\Work\Intenship_Work\09-06-25\Detectron\door_dataset"

# List all door folders
door_folders = [f for f in os.listdir(door_dataset_folder) if os.path.isdir(os.path.join(door_dataset_folder, f))]

pairs = []

# Generate positive pairs (same door)
for door in door_folders:
    door_path = os.path.join(door_dataset_folder, door)
    images = [img for img in os.listdir(door_path) if img.lower().endswith(('.jpg', '.png'))]
    
    # Create all unique pairs without repetition
    pos_pairs = list(itertools.combinations(images, 2))
    for (img1, img2) in pos_pairs:
        pairs.append({
            "img1": os.path.join(door_path, img1),
            "img2": os.path.join(door_path, img2),
            "label": 1
        })

# Generate negative pairs (different doors)
# To limit the number, generate random pairs instead of all combinations
num_neg_pairs = len(pairs)  # same number of negative pairs as positive pairs for balance

while len([p for p in pairs if p["label"] == 0]) < num_neg_pairs:
    door1, door2 = random.sample(door_folders, 2)  # pick two different doors
    door1_path = os.path.join(door_dataset_folder, door1)
    door2_path = os.path.join(door_dataset_folder, door2)
    
    img1 = random.choice([img for img in os.listdir(door1_path) if img.lower().endswith(('.jpg', '.png'))])
    img2 = random.choice([img for img in os.listdir(door2_path) if img.lower().endswith(('.jpg', '.png'))])
    
    pairs.append({
        "img1": os.path.join(door1_path, img1),
        "img2": os.path.join(door2_path, img2),
        "label": 0
    })

# Shuffle pairs
random.shuffle(pairs)

# Save to CSV file for easy loading later
csv_file = "door_pairs_dataset.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["img1", "img2", "label"])
    writer.writeheader()
    for pair in pairs:
        writer.writerow(pair)

print(f"Dataset pairs saved to {csv_file}")
