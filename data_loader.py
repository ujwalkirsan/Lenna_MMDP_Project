import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class RefCOCOgDataset(Dataset):
    """
    Dataset class for RefCOCOg
    """
    def __init__(self, data_dir, split="val2014", transform=None):
        """
        Initialize the RefCOCOg dataset
        
        Args:
            data_dir: Directory containing the RefCOCOg dataset
            split: Dataset split (val2014, train2014, test2014)
            transform: Image transforms
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Load instances
        with open(os.path.join(data_dir, "instances.json"), "r") as f:
            all_instances = json.load(f)
        
        # Filter instances by split
        self.instances = [instance for instance in all_instances if instance["split"] == split]
        print(f"Loaded {len(self.instances)} instances for split {split}")
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Returns:
            image: Image tensor
            query: Text query
            gt_box: Ground truth bounding box [x, y, width, height]
            image_id: Image ID
            ann_id: Annotation ID
        """
        instance = self.instances[idx]
        
        # Get image
        image_id = instance["image_id"]
        image_path = os.path.join(self.data_dir, self.split, f"COCO_{self.split}_{image_id:012d}.jpg")
        image = Image.open(image_path).convert("RGB")
        
        # Get referring expression
        query = instance["sentence"]
        
        # Get ground truth box
        gt_box = instance["bbox"]  # [x, y, width, height]
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        return {
            "image": image,
            "query": query,
            "gt_box": torch.tensor(gt_box, dtype=torch.float),
            "image_id": image_id,
            "ann_id": instance["ann_id"]
        }


def collate_fn(batch):
    """
    Custom collate function for the dataloader
    """
    images = [item["image"] for item in batch]
    queries = [item["query"] for item in batch]
    gt_boxes = [item["gt_box"] for item in batch]
    image_ids = [item["image_id"] for item in batch]
    ann_ids = [item["ann_id"] for item in batch]
    
    return {
        "images": images,
        "queries": queries,
        "gt_boxes": torch.stack(gt_boxes),
        "image_ids": image_ids,
        "ann_ids": ann_ids
    }


def get_dataloader(data_dir, split="val2014", batch_size=1, transform=None, shuffle=False):
    """
    Get a dataloader for the RefCOCOg dataset
    
    Args:
        data_dir: Directory containing the RefCOCOg dataset
        split: Dataset split (val2014, train2014, test2014)
        batch_size: Batch size
        transform: Image transforms
        shuffle: Whether to shuffle the dataset
        
    Returns:
        dataloader: DataLoader for the dataset
    """
    dataset = RefCOCOgDataset(data_dir, split, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
    
    return dataloader


def process_refcocog_example(image_path, query, model):
    """
    Process a single RefCOCOg example
    
    Args:
        image_path: Path to the image
        query: Referring expression
        model: Lenna model
        
    Returns:
        boxes: Detected bounding boxes
        scores: Confidence scores
        image_with_boxes: Image with drawn bounding boxes
    """
    return model.process_image_with_query(image_path, query)