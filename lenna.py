import os
import torch
import numpy as np
from PIL import Image, ImageDraw
import json
from transformers import AutoProcessor, AutoModelForCausalLM
from groundingdino.util.inference import load_model as load_grounding_dino
from groundingdino.util.inference import predict as grounding_dino_predict
from groundingdino.util.utils import clean_state_dict
import argparse
import cv2
from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor

class Lenna:
    def __init__(
        self,
        llava_model_path="llava-hf/llava-1.5-7b-hf",
        grounding_dino_config_path="groundingdino/config/GroundingDINO_SwinT_OGC.py",
        grounding_dino_checkpoint_path="groundingdino/weights/groundingdino_swint_ogc.pth",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize Lenna model with LLaVA and Grounding DINO components
        """
        self.device = device
        print(f"Using device: {self.device}")
        
        # Load LLaVA model
        print("Loading LLaVA model...")
        self.llava_tokenizer = AutoTokenizer.from_pretrained(llava_model_path, use_fast=False)
        self.llava_processor = AutoProcessor.from_pretrained(llava_model_path)
        self.llava_model = AutoModelForCausalLM.from_pretrained(
            llava_model_path, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        print("LLaVA model loaded!")
        
        # Load Grounding DINO model
        print("Loading Grounding DINO model...")
        self.grounding_dino_model = load_grounding_dino(
            config_path=grounding_dino_config_path,
            checkpoint_path=grounding_dino_checkpoint_path,
            device=self.device
        )
        print("Grounding DINO model loaded!")
        
        # Grounding DINO parameters
        self.box_threshold = 0.35
        self.text_threshold = 0.25
        
    def llava_get_text_embeds(self, text_prompt):
        """Get text embeddings from LLaVA for a given text prompt"""
        inputs = self.llava_processor(text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_embeds = self.llava_model.get_model().embed_text(inputs["input_ids"])
        return text_embeds
        
    def process_image_with_query(self, image_path, query):
        """
        Process an image with a natural language query to identify the referred objects
        
        Args:
            image_path: Path to the image
            query: Natural language query describing the target object/region
            
        Returns:
            boxes: Bounding boxes for the detected objects
            scores: Confidence scores for each box
            image_with_boxes: Image with drawn bounding boxes
        """
        # Load image
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        
        # Process with Grounding DINO
        boxes, logits, phrases = grounding_dino_predict(
            model=self.grounding_dino_model,
            image=image_np,
            caption=query,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        
        # Convert boxes to the right format [x1, y1, x2, y2]
        H, W, _ = image_np.shape
        boxes_scaled = boxes * torch.Tensor([W, H, W, H])
        boxes_xyxy = boxes_scaled.cpu().numpy()
        
        # Draw bounding boxes on the image
        image_with_boxes = image_pil.copy()
        draw = ImageDraw.Draw(image_with_boxes)
        
        for box, score, phrase in zip(boxes_xyxy, logits, phrases):
            x0, y0, x1, y1 = box
            draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
            draw.text((x0, y0 - 10), f"{phrase}: {score:.2f}", fill="red")
        
        return boxes_xyxy, logits.cpu().numpy(), image_with_boxes
    
    def process_refcocog_dataset(self, data_dir, output_dir="results", split="val2014"):
        """
        Process the RefCOCOg dataset
        
        Args:
            data_dir: Directory containing the RefCOCOg dataset
            output_dir: Directory to save results
            split: Dataset split (val2014, train2014, test2014)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load instances
        with open(os.path.join(data_dir, "instances.json"), "r") as f:
            instances = json.load(f)
        
        # Process each instance
        results = []
        
        for instance in tqdm(instances):
            if instance["split"] != split:
                continue
                
            # Get image path
            image_id = instance["image_id"]
            image_path = os.path.join(data_dir, split, f"COCO_{split}_{image_id:012d}.jpg")
            
            # Get referring expression
            query = instance["sentence"]
            
            try:
                # Process image with query
                boxes, scores, image_with_boxes = self.process_image_with_query(image_path, query)
                
                # Save image with boxes
                output_image_path = os.path.join(output_dir, f"{image_id}_{instance['ann_id']}.jpg")
                image_with_boxes.save(output_image_path)
                
                # Collect results
                result = {
                    "image_id": image_id,
                    "ann_id": instance["ann_id"],
                    "sentence": query,
                    "predicted_boxes": boxes.tolist(),
                    "scores": scores.tolist(),
                    "output_image_path": output_image_path
                }
                results.append(result)
                
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
        
        # Save results
        with open(os.path.join(output_dir, f"results_{split}.json"), "w") as f:
            json.dump(results, f)
        
        return results
    
    def evaluate_refcocog(self, results, ground_truth, iou_threshold=0.5):
        """
        Evaluate the model on the RefCOCOg dataset
        
        Args:
            results: List of prediction results
            ground_truth: Ground truth annotations
            iou_threshold: IoU threshold for considering a prediction correct
            
        Returns:
            accuracy: Accuracy of the model
        """
        correct = 0
        total = 0
        
        for result in results:
            # Find ground truth box
            gt_box = None
            for gt in ground_truth:
                if gt["ann_id"] == result["ann_id"]:
                    gt_box = gt["bbox"]  # [x, y, width, height]
                    break
            
            if gt_box is None:
                continue
                
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
                iou = self._calculate_iou(gt_box_xyxy, pred_box)
                
                if iou >= iou_threshold:
                    correct += 1
                    
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
        return accuracy
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes [x1, y1, x2, y2]"""
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


def main():
    parser = argparse.ArgumentParser(description="Lenna model for visual grounding")
    parser.add_argument("--data_dir", type=str, default="refcocog_small", help="Directory containing the RefCOCOg dataset")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--split", type=str, default="val2014", choices=["train2014", "val2014", "test2014"], 
                        help="Dataset split to process")
    args = parser.parse_args()
    
    # Initialize Lenna model
    lenna = Lenna()
    
    # Process RefCOCOg dataset
    results = lenna.process_refcocog_dataset(args.data_dir, args.output_dir, args.split)
    
    print(f"Processed {len(results)} images from {args.split}")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()