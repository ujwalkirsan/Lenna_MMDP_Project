import argparse
import os
from lenna import Lenna
from data_loader import get_dataloader
from tqdm import tqdm
import torch


def main():
    parser = argparse.ArgumentParser(description="Run Lenna model on RefCOCOg dataset")
    parser.add_argument("--data_dir", type=str, default="refcocog_small", help="Directory containing the RefCOCOg dataset")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--split", type=str, default="val2014", choices=["train2014", "val2014", "test2014"], 
                        help="Dataset split to process")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run the model on")
    parser.add_argument("--llava_model_path", type=str, default="llava-hf/llava-1.5-7b-hf", 
                        help="Path to the LLaVA model")
    parser.add_argument("--grounding_dino_config_path", type=str, 
                        default="groundingdino/config/GroundingDINO_SwinT_OGC.py", 
                        help="Path to the Grounding DINO config")
    parser.add_argument("--grounding_dino_checkpoint_path", type=str, 
                        default="groundingdino/weights/groundingdino_swint_ogc.pth", 
                        help="Path to the Grounding DINO checkpoint")
    parser.add_argument("--evaluate", action="store_true", help="Whether to evaluate the results")
    args = parser.parse_args()
    
    # Initialize Lenna model
    lenna = Lenna(
        llava_model_path=args.llava_model_path,
        grounding_dino_config_path=args.grounding_dino_config_path,
        grounding_dino_checkpoint_path=args.grounding_dino_checkpoint_path,
        device=args.device
    )
    
    # Process RefCOCOg dataset
    results = lenna.process_refcocog_dataset(args.data_dir, args.output_dir, args.split)
    
    print(f"Processed {len(results)} images from {args.split}")
    print(f"Results saved to {args.output_dir}")
    
    # Evaluate if requested
    if args.evaluate:
        from evaluate import load_ground_truth, evaluate_results
        
        # Load ground truth
        ground_truth = load_ground_truth(args.data_dir)
        
        # Evaluate results
        results_file = os.path.join(args.output_dir, f"results_{args.split}.json")
        metrics = evaluate_results(results_file, ground_truth, iou_thresholds=[0.5, 0.7, 0.9])
        
        # Save metrics
        import json
        metrics_file = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":
    main()