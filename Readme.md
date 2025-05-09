# Lenna: Visual Grounding for RefCOCOg Dataset

<p align="center">
  <img src="assets\architecture.png" alt="Lenna Model Architecture" />
</p>

This repository implements the Lenna model for visual grounding on the RefCOCOg dataset. Lenna combines powerful vision-language models (LLaVA) with object detection (Grounding DINO) to accurately locate objects in images based on natural language referring expressions.

## Table of Contents
- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Implementation Details](#implementation-details)
- [References](#references)

## Overview

Visual grounding is the task of localizing a specific object in an image based on a textual description. Lenna excels at this task by leveraging:

1. **LLaVA**: A powerful vision-language model that understands the relationship between visual content and natural language
2. **Grounding DINO**: A state-of-the-art zero-shot object detection model with strong grounding capabilities
3. **Integrated Pipeline**: A carefully designed pipeline that combines these models for optimal performance


## Model Architecture

The Lenna model architecture consists of two main components that work together:

### 1. LLaVA (Language-and-Vision Assistant)
- Processes both image and text inputs
- Creates rich visual-linguistic representations
- Understands complex referring expressions and their visual manifestations

### 2. Grounding DINO
- Zero-shot object detection model
- Takes text descriptions and images as input
- Produces bounding boxes with confidence scores


## Dataset

### RefCOCOg Dataset
This implementation works with the RefCOCOg dataset, which contains:
- Images from the COCO dataset
- Natural language referring expressions for objects
- Ground truth bounding boxes

### Dataset Structure
The RefCOCOg dataset should be organized as follows:

```
refcocog_small/
├── instances.json
├── train2014/
│   └── COCO_train2014_*.jpg
├── val2014/
│   └── COCO_val2014_*.jpg
└── test2014/
    └── COCO_test2014_*.jpg
```

### Dataset Statistics
This implementation is designed for a small subset of RefCOCOg with:
- **Train**: 1,000 instances
- **Validation**: 200 instances
- **Test**: 200 instances

### JSON Format
The `instances.json` file should contain a list of annotation instances with the following structure:

```json
[
  {
    "image_id": 123456,
    "ann_id": 789012,
    "sentence": "the man wearing blue jeans",
    "bbox": [x, y, width, height],
    "split": "val2014"
  },
  ...
]
```

## Installation

### Requirements
Before running the code, make sure you have the following dependencies installed:

```bash
# Clone the repository
git clone https://github.com/yourusername/lenna-refcocog.git
cd lenna-refcocog

# Install dependencies
pip install -r requirements.txt
```

### Required Packages
```bash
torch>=1.10.0
torchvision>=0.11.1
transformers>=4.28.0
Pillow>=9.0.0
numpy>=1.20.0
tqdm>=4.62.0
opencv-python>=4.5.0
groundingdino @ git+https://github.com/IDEA-Research/GroundingDINO.git
```

### Model Weights
To run Lenna, you need to download the following model weights:

```bash
# Create directories
mkdir -p groundingdino/config
mkdir -p groundingdino/weights

# Download Grounding DINO config
wget -O groundingdino/config/GroundingDINO_SwinT_OGC.py https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py

# Download Grounding DINO weights
wget -O groundingdino/weights/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

## Usage

### Processing the Dataset

To process the RefCOCOg dataset with Lenna:

```bash
# Process validation set
python run_lenna.py --data_dir refcocog_small --output_dir results --split val2014

# Process test set
python run_lenna.py --data_dir refcocog_small --output_dir results --split test2014

# Process train set
python run_lenna.py --data_dir refcocog_small --output_dir results --split train2014
```

### Evaluating Results

To evaluate the model's performance:

```bash
python evaluate.py --data_dir refcocog_small --results_file results/results_val2014.json --output_file results/metrics.json
```

Or process and evaluate in one step:

```bash
python run_lenna.py --data_dir refcocog_small --output_dir results --split val2014 --evaluate
```

### Processing a Single Image

To process a single image with a specific referring expression:

```python
from lenna import Lenna
from PIL import Image

# Initialize Lenna model
lenna = Lenna()

# Process an image with a referring expression
image_path = "path/to/image.jpg"
query = "the woman in a red dress"
boxes, scores, image_with_boxes = lenna.process_image_with_query(image_path, query)

# Save the result
image_with_boxes.save("result.jpg")
```

### Command-Line Arguments

#### run_lenna.py
```
--data_dir: Directory containing the RefCOCOg dataset (default: refcocog_small)
--output_dir: Directory to save results (default: results)
--split: Dataset split to process (choices: train2014, val2014, test2014, default: val2014)
--device: Device to run the model on (default: cuda if available, otherwise cpu)
--llava_model_path: Path to the LLaVA model (default: llava-hf/llava-1.5-7b-hf)
--grounding_dino_config_path: Path to the Grounding DINO config (default: groundingdino/config/GroundingDINO_SwinT_OGC.py)
--grounding_dino_checkpoint_path: Path to the Grounding DINO checkpoint (default: groundingdino/weights/groundingdino_swint_ogc.pth)
--evaluate: Whether to evaluate the results (flag)
```

#### evaluate.py
```
--data_dir: Directory containing the RefCOCOg dataset (default: refcocog_small)
--results_file: Path to the results file (default: results/results_val2014.json)
--output_file: Path to save metrics (default: results/metrics.json)
```

## Results
### Qualitative Results

<p align="center">
  <img src="assets\results.png" alt="Qualitative Results" />
  <br>
  <em>Examples of successful visual grounding by Lenna on the RefCOCOg dataset</em>
</p>

### Analysis

*[This section will be populated with analysis of the model's performance, strengths, and weaknesses after running experiments]*

## Implementation Details

The repository consists of the following main components:

### 1. lenna.py
The core implementation of the Lenna model:
- Integrates LLaVA and Grounding DINO
- Processes images with natural language queries
- Provides methods for working with the RefCOCOg dataset
- Includes evaluation functionality

```python
# Example usage
from lenna import Lenna

# Initialize model
lenna = Lenna()

# Process an image with a referring expression
boxes, scores, image_with_boxes = lenna.process_image_with_query("image.jpg", "the cat on the sofa")
```

### 2. data_loader.py
Utilities for loading and processing the RefCOCOg dataset:
- `RefCOCOgDataset` class for dataset handling
- DataLoader creation for batch processing
- Helper functions for data processing

```python
# Example usage
from data_loader import get_dataloader

# Get dataloader for validation set
dataloader = get_dataloader("refcocog_small", split="val2014", batch_size=1)

# Process batch
for batch in dataloader:
    images = batch["images"]
    queries = batch["queries"]
    gt_boxes = batch["gt_boxes"]
    # Process batch...
```

### 3. evaluate.py
Script for evaluating model performance:
- Calculates accuracy at various IoU thresholds
- Loads ground truth and predictions
- Generates detailed metrics

### 4. run_lenna.py
Main script for running the Lenna model:
- Command-line interface for model execution
- Supports different dataset splits
- Integrates evaluation capabilities

## References

- [Lenna: Large Language and Vision Assistant](https://github.com/Meituan-AutoML/Lenna)
- [LLaVA: Large Language and Vision Assistant](https://github.com/haotian-liu/LLaVA)
- [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://github.com/IDEA-Research/GroundingDINO)
- [RefCOCOg: Referring Expressions Dataset](https://github.com/lichengunc/refer)

## Citation

If you find this implementation useful, please consider citing the original papers:

```
@article{lenna2023,
  title={Lenna: Language-Enhanced Visual Grounding},
  author={Authors},
  journal={arXiv preprint},
  year={2023}
}

@article{llava2023,
  title={Visual Instruction Tuning},
  author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
  journal={arXiv preprint arXiv:2304.08485},
  year={2023}
}

@article{groundingdino2023,
  title={Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}
```


## Acknowledgements

This implementation is based on the Lenna model by Meituan-AutoML and relies on the LLaVA and Grounding DINO models. We thank the authors for making their code and models available.