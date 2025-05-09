import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class TextGuidedObjectDetector:
    def __init__(self):
        """Initialize the text-guided object detection system with multiple models"""
        print("Initializing models... (this may take a moment)")
        
        # Initialize CLIP for text-image similarity matching
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Initialize BLIP-2 for image understanding and captioning
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        
        # Initialize OneFormer for semantic segmentation
        self.oneformer_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
        self.oneformer_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
        
        print("Models initialized successfully")
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model.to(self.device)
        self.blip_model.to(self.device)
        self.oneformer_model.to(self.device)

    def preprocess_image(self, image_path):
        """Load and preprocess image"""
        image = Image.open(image_path).convert("RGB")
        return image
    
    def analyze_image(self, image):
        """Generate a description of the image using BLIP-2"""
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        
        generated_ids = self.blip_model.generate(
            **inputs,
            max_length=50,
            num_beams=5,
            early_stopping=True
        )
        
        description = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return description
    
    def segment_image(self, image):
        """Perform semantic segmentation using OneFormer"""
        inputs = self.oneformer_processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.oneformer_model(**inputs)
        
        # Process semantic segmentation predictions
        segmentation = self.oneformer_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(image.height, image.width)]
        )[0]
        
        return segmentation.cpu().numpy()
    
    def extract_regions(self, image, segmentation):
        """Extract regions from segmentation map"""
        unique_segments = np.unique(segmentation)
        regions = []
        
        for segment_id in unique_segments:
            if segment_id == 0:  # Skip background
                continue
                
            # Create mask for this segment
            mask = (segmentation == segment_id).astype(np.uint8)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Extract region and segment ID
                regions.append({
                    'box': (x, y, x+w, y+h),
                    'segment_id': segment_id,
                    'mask': mask
                })
        
        return regions
    
    def extract_salient_regions(self, image):
        """Extract salient regions using color clustering and contour detection"""
        # Convert PIL image to numpy array
        img_np = np.array(image)
        
        # Convert to RGB if it's not
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
        # Resize for faster processing
        scale = 1.0
        if max(img_np.shape[:2]) > 1000:
            scale = 1000 / max(img_np.shape[:2])
            width = int(img_np.shape[1] * scale)
            height = int(img_np.shape[0] * scale)
            img_np = cv2.resize(img_np, (width, height))
        
        # Convert to LAB color space for better clustering
        lab_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        
        # Reshape for K-means
        pixels = lab_image.reshape(-1, 3)
        
        # Perform K-means clustering
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # Reshape labels back to image dimensions
        segmented_image = labels.reshape(img_np.shape[:2])
        
        # Extract regions
        regions = []
        for i in range(n_clusters):
            # Create binary mask for this cluster
            mask = (segmented_image == i).astype(np.uint8)
            
            # Apply morphological operations to clean up mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process each contour that's large enough
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filter out small regions
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Scale back to original size
                    if scale != 1.0:
                        x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                    
                    regions.append({
                        'box': (x, y, x+w, y+h),
                        'contour': contour,
                        'area': area
                    })
        
        return regions
    
    def match_text_to_regions(self, image, regions, text_query):
        """Match text query to image regions using CLIP"""
        matches = []
        
        if not regions:
            return matches
        
        # Process the text query
        text_inputs = self.clip_processor(text=[text_query], return_tensors="pt").to(self.device)
        text_features = self.clip_model.get_text_features(**text_inputs)
        
        # Process each region
        for region in regions:
            box = region['box']
            # Crop region from image
            region_img = image.crop(box)
            
            # Process region with CLIP
            try:
                image_inputs = self.clip_processor(images=region_img, return_tensors="pt").to(self.device)
                image_features = self.clip_model.get_image_features(**image_inputs)
                
                # Calculate similarity
                similarity = torch.nn.functional.cosine_similarity(text_features, image_features).item()
                
                matches.append({
                    'box': box,
                    'similarity': similarity
                })
            except Exception as e:
                # Skip problematic regions
                print(f"Error processing region: {e}")
                continue
        
        # Sort by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches
    
    def draw_results(self, image, matches, text_query, top_n=3):
        """Draw bounding boxes around the top matches"""
        result_img = image.copy()
        draw = ImageDraw.Draw(result_img)
        
        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            try:
                # Try another common font on different OS
                font = ImageFont.truetype("DejaVuSans.ttf", 20)
            except:
                font = ImageFont.load_default()
        
        colors = ["red", "orange", "yellow", "green", "blue"]
        
        # Draw top N matches
        for i, match in enumerate(matches[:top_n]):
            box = match['box']
            color = colors[i % len(colors)]
            
            # Draw rectangle
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color, width=3)
            
            # Add label with confidence
            confidence = int(match['similarity'] * 100)
            label_text = f"{confidence}% match"
            
            # Draw label background
            text_width, text_height = draw.textbbox((0, 0), label_text, font=font)[2:]
            draw.rectangle(
                [(box[0], box[1] - text_height - 4), (box[0] + text_width + 4, box[1])],
                fill=color
            )
            
            # Draw text
            draw.text((box[0] + 2, box[1] - text_height - 2), label_text, fill="white", font=font)
        
        # Add query text at the bottom
        query_text = f"Query: {text_query}"
        draw.text((10, result_img.height - 30), query_text, fill="black", font=font)
        
        return result_img
    
    def process_image(self, image_path, text_query, output_path=None):
        """Process an image with a text query to find matching regions"""
        print(f"Processing image with query: '{text_query}'")
        
        # Load and preprocess image
        image = self.preprocess_image(image_path)
        
        # Analyze image (for debugging)
        description = self.analyze_image(image)
        print(f"Image description: {description}")
        
        # Get regions using segmentation and color clustering
        print("Extracting regions...")
        regions = self.extract_salient_regions(image)
        print(f"Found {len(regions)} regions")
        
        # Match text to regions
        print("Matching text to regions...")
        matches = self.match_text_to_regions(image, regions, text_query)
        
        # Draw results
        print("Drawing results...")
        result_img = self.draw_results(image, matches, text_query)
        
        # Save or return the result
        if output_path:
            result_img.save(output_path)
            print(f"Results saved to {output_path}")
        
        return result_img, matches


# Example usage
if __name__ == "__main__":
    detector = TextGuidedObjectDetector()
    
    # Get input image path and text query from user or use defaults
    input_path = input("Enter path to input image: ")
    text_query = input("Enter text query : ")
    
    # Create output path
    filename = os.path.basename(input_path)
    output_path = f"result_{filename}"
    
    # Process the image
    result_img, matches = detector.process_image(input_path, text_query, output_path)
    
    if matches:
        print(f"Found {len(matches)} matches. Top match confidence: {int(matches[0]['similarity'] * 100)}%")
    else:
        print("No matches found")