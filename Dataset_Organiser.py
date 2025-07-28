import os
import shutil
import random
from pathlib import Path

# Paths
source_dir = '/Users/geraldalanraja/Documents/Projects/COVID-19 Detection from Chest CT/Dataset'           
target_dir = '/Users/geraldalanraja/Documents/Projects/COVID-19 Detection from Chest CT/Dataset/Processed'    

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Classes
classes = ['COVID', 'non-COVID']

# Create target directory structure
for split in ['train', 'val', 'test']:
    for cls in classes:
        Path(os.path.join(target_dir, split, cls)).mkdir(parents=True, exist_ok=True)

# Move images for each class
for cls in classes:
    cls_path = os.path.join(source_dir, cls)
    images = os.listdir(cls_path)
    images = [f for f in images if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)

    total = len(images)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    # Move files
    for f in train_files:
        shutil.move(os.path.join(cls_path, f), os.path.join(target_dir, 'train', cls, f))

    for f in val_files:
        shutil.move(os.path.join(cls_path, f), os.path.join(target_dir, 'val', cls, f))

    for f in test_files:
        shutil.move(os.path.join(cls_path, f), os.path.join(target_dir, 'test', cls, f))

    # Remove original class folder if empty
    if not os.listdir(cls_path):
        os.rmdir(cls_path)

print("✅ Dataset successfully split into train/val/test and files moved.")
