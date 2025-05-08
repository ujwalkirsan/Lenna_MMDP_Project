import os
import json
import shutil
import random
import pickle
import re

# Configuration - set to True to respect original splits
RESPECT_ORIGINAL_SPLITS = False  # Set to True if you want train images to stay in train, val in val
RENAME_FILES = True  # Set to True to rename files to match their new split

# Set paths
train_dir = "/images/train2014"
val_dir = "/images/val2014"
refcocog_dir = "refcocog"

# Output directories
output_dir = "refcocog_small"
output_train_dir = os.path.join(output_dir, "train2014")
output_val_dir = os.path.join(output_dir, "val2014")
output_test_dir = os.path.join(output_dir, "test2014")
output_refcocog_dir = os.path.join(output_dir, "refcocog")

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_val_dir, exist_ok=True)
os.makedirs(output_test_dir, exist_ok=True)
os.makedirs(output_refcocog_dir, exist_ok=True)

# Set sizes
train_size = 1000
val_size = 200
test_size = 200
seed = 42
random.seed(seed)

print("Loading annotations...")
instances_file = os.path.join(refcocog_dir, "instances.json")
refs_google_file = os.path.join(refcocog_dir, "refs(google).p")
refs_umd_file = os.path.join(refcocog_dir, "refs(umd).p")

with open(instances_file, 'r') as f:
    instances_data = json.load(f)

with open(refs_google_file, 'rb') as f:
    refs_google = pickle.load(f)

with open(refs_umd_file, 'rb') as f:
    refs_umd = pickle.load(f)

# Get all image IDs and separate by original split
all_images = instances_data['images']
print(f"Total images in dataset: {len(all_images)}")

train_images = []
val_images = []

for img in all_images:
    if 'COCO_train2014_' in img['file_name']:
        train_images.append(img)
    elif 'COCO_val2014_' in img['file_name']:
        val_images.append(img)

print(f"Original split: {len(train_images)} train images, {len(val_images)} val images")

# Function to try multiple paths
def find_image(filename, primary_dir, fallback_dirs):
    primary_path = os.path.join(primary_dir, filename)
    if os.path.exists(primary_path):
        return primary_path
        
    for dir_path in fallback_dirs:
        test_path = os.path.join(dir_path, filename)
        if os.path.exists(test_path):
            return test_path
            
    return None

# Create splits based on configuration
if RESPECT_ORIGINAL_SPLITS:
    # Only use train images for training and val images for val/test
    random.shuffle(train_images)
    random.shuffle(val_images)
    
    selected_train_images = train_images[:train_size]
    half_val_size = val_size + test_size
    if len(val_images) < half_val_size:
        print(f"Warning: Not enough val images. Using {len(val_images)} instead of {half_val_size}")
        half_val_size = len(val_images)
    
    selected_val_images = val_images[:val_size]
    selected_test_images = val_images[val_size:val_size+test_size]
else:
    # Mix both train and val images in all splits
    all_image_list = all_images.copy()
    random.shuffle(all_image_list)
    
    selected_train_images = all_image_list[:train_size]
    selected_val_images = all_image_list[train_size:train_size+val_size]
    selected_test_images = all_image_list[train_size+val_size:train_size+val_size+test_size]

train_image_ids = set(img['id'] for img in selected_train_images)
val_image_ids = set(img['id'] for img in selected_val_images)
test_image_ids = set(img['id'] for img in selected_test_images)

print(f"Selected {len(train_image_ids)} train, {len(val_image_ids)} val, {len(test_image_ids)} test images")

# Create new instances data
new_instances = {
    'info': instances_data['info'],
    'licenses': instances_data['licenses'],
    'images': [],
    'annotations': [],
    'categories': instances_data['categories']
}

# Function to rename file if needed
def rename_for_split(filename, split):
    if not RENAME_FILES:
        return filename
    
    # Extract number from filename
    match = re.search(r'_(\d+)\.jpg$', filename)
    if not match:
        return filename
    
    img_number = match.group(1)
    new_filename = f"COCO_{split}2014_{img_number}.jpg"
    return new_filename

# Process each split
splits = {
    'train': {'images': selected_train_images, 'ids': train_image_ids, 'output_dir': output_train_dir},
    'val': {'images': selected_val_images, 'ids': val_image_ids, 'output_dir': output_val_dir},
    'test': {'images': selected_test_images, 'ids': test_image_ids, 'output_dir': output_test_dir}
}

image_id_to_new_filename = {}  # To track renamed files

for split_name, split_info in splits.items():
    print(f"Processing {split_name} split...")
    
    # Add images to new instances
    updated_images = []
    for img in split_info['images']:
        img_copy = img.copy()
        original_filename = img_copy['file_name']
        
        # Rename file if configured to do so
        new_filename = rename_for_split(original_filename, split_name)
        if new_filename != original_filename:
            img_copy['file_name'] = new_filename
            image_id_to_new_filename[img['id']] = new_filename
        
        updated_images.append(img_copy)
    
    new_instances['images'].extend(updated_images)
    
    # Filter annotations
    split_annotations = [ann for ann in instances_data['annotations'] if ann['image_id'] in split_info['ids']]
    new_instances['annotations'].extend(split_annotations)
    
    # Copy images
    print(f"Copying {len(split_info['images'])} images for {split_name} split...")
    copied_count = 0
    failed_count = 0
    
    for img in split_info['images']:
        original_filename = img['file_name']
        img_id = img['id']
        
        # Determine source directory
        fallback_dirs = []
        if 'COCO_train2014_' in original_filename:
            primary_dir = train_dir
            fallback_dirs = ["train2014", os.path.join("images", "train2014")]
        elif 'COCO_val2014_' in original_filename:
            primary_dir = val_dir
            fallback_dirs = ["val2014", os.path.join("images", "val2014")]
        else:
            print(f"Warning: Unknown source for {original_filename}")
            failed_count += 1
            continue
            
        # Find the image
        src_path = find_image(original_filename, primary_dir, fallback_dirs)
        
        if src_path:
            # Determine destination filename (renamed if needed)
            if img_id in image_id_to_new_filename:
                dst_filename = image_id_to_new_filename[img_id]
            else:
                dst_filename = original_filename
                
            dst_path = os.path.join(split_info['output_dir'], dst_filename)
            shutil.copy(src_path, dst_path)
            copied_count += 1
        else:
            print(f"Warning: File not found: {original_filename}")
            failed_count += 1
    
    print(f"Split {split_name}: Copied {copied_count} images, Failed to find {failed_count} images")

# Save new instances file
print("Saving new annotations...")
output_instances_file = os.path.join(output_refcocog_dir, "instances.json")
with open(output_instances_file, 'w') as f:
    json.dump(new_instances, f)

# Filter and update refs files
all_selected_ids = train_image_ids | val_image_ids | test_image_ids

# Update references with new filenames if needed
filtered_refs_google = []
for ref in refs_google:
    if ref['image_id'] in all_selected_ids:
        ref_copy = ref.copy()
        filtered_refs_google.append(ref_copy)

filtered_refs_umd = []
for ref in refs_umd:
    if ref['image_id'] in all_selected_ids:
        ref_copy = ref.copy()
        filtered_refs_umd.append(ref_copy)

# Save filtered refs files
output_refs_google_file = os.path.join(output_refcocog_dir, "refs(google).p")
with open(output_refs_google_file, 'wb') as f:
    pickle.dump(filtered_refs_google, f)

output_refs_umd_file = os.path.join(output_refcocog_dir, "refs(umd).p")
with open(output_refs_umd_file, 'wb') as f:
    pickle.dump(filtered_refs_umd, f)

print(f"Done! Created dataset with {train_size} train, {val_size} val, {test_size} test images.")
print(f"Output saved to: {output_dir}")
print(f"File naming: {'Renamed to match new split' if RENAME_FILES else 'Original filenames preserved'}")
print(f"Split strategy: {'Respected original train/val splits' if RESPECT_ORIGINAL_SPLITS else 'Mixed train/val images across splits'}")