import argparse
import json
import os
from lenna import Lenna
import numpy as np
from tqdm import tqdm


def load_ground_truth(data_dir):
    """
    Load ground truth annotations from the RefCOCOg dataset
    
    Args:
        data_dir: Directory containing the RefCOCOg dataset
        
    Returns:
        ground_truth: Dictionary mapping annotation IDs to ground truth boxes
    """
    with open(os.path.join(data_dir, "instances.json"), "r") as f:
        instances = json.load(f)
    
    ground_truth = {}
    for instance in instances:
        ground_truth[instance["ann_id"]] = {
            "bbox": instance["bbox"],  # [x, y, width, height]
            "image_id": instance["image_id"],
            "sentence": instance["sentence"]
        }
    
    return ground_truth


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes
    
    Args:
        box1: First box in format [x1, y1, x2, y2]
        box2: Second box in format [x1, y1, x2, y2]
        
    Returns:
        iou: IoU between the boxes
    """
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
        
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate IoU
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_results(results_file, ground_truth, iou_thresholds=[0.5]):
    """
    Evaluate results on the RefCOCOg dataset
    
    Args:
        results_file: Path to the results file
        ground_truth: Ground truth annotations
        iou_thresholds: List of IoU thresholds for evaluation
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Load results
    with open(results_file, "r") as f:
        results = json.load(f)
    
    # Calculate accuracy for each IoU threshold
    metrics = {}
    
    for iou_threshold in iou_thresholds:
        correct = 0
        total = 0
        
        for result in results:
            ann_id = result["ann_id"]
            
            # Skip if no ground truth
            if ann_id not in ground_truth:
                continue
            
            # Get ground truth box
            gt_box = ground_truth[ann_id]["bbox"]  # [x, y, width, height]
            
            # Convert ground truth box from [x, y, width, height] to [x1, y1, x2, y2]
            gt_box_xyxy = [
                gt_box[0], 
                gt_box[1], 
                gt_box[0] + gt_box[2], 
                gt_box[1] + gt_box[3]
            ]
            
            # Get predicted box with highest score
            pred_boxes = result["predicted_boxes"]
            scores = result["scores"]
            
            if len(pred_boxes) > 0:
                best_idx = np.argmax(scores)
                pred_box = pred_boxes[best_idx]
                
                # Calculate IoU
                iou = calculate_iou(gt_box_xyxy, pred_box)
                
                if iou >= iou_threshold:
                    correct += 1
                    
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        metrics[f"accuracy@{iou_threshold}"] = accuracy
        print(f"Accuracy@{iou_threshold}: {accuracy:.4f} ({correct}/{total})")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Lenna model on RefCOCOg dataset")
    parser.add_argument("--data_dir", type=str, default="refcocog_small", help="Directory containing the RefCOCOg dataset")
    parser.add_argument("--results_file", type=str, default="results/results_val2014.json", help="Path to the results file")
    parser.add_argument("--output_file", type=str, default="results/metrics.json", help="Path to save metrics")
    args = parser.parse_args()
    
    # Load ground truth
    ground_truth = load_ground_truth(args.data_dir)
    
    # Evaluate results
    metrics = evaluate_results(args.results_file, ground_truth, iou_thresholds=[0.5, 0.7, 0.9])
    
    # Save metrics
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {args.output_file}")


if __name__ == "__main__":
    main()