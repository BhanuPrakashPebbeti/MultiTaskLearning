# Multi-Task Learning for Cost-Effective Autonomous Driving

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

## Overview

MultiTaskLearning is a human brain-inspired neural network architecture that performs multiple computer vision tasks in a single pass to enable cost-effective autonomous driving. Our system achieves real-time performance (>60 FPS).
## Key Features

- **Multi-Task Learning (MTL)**: Combines 2D object detection, semantic segmentation, and monocular depth estimation in a single neural network.
- **Efficient Architecture**: Shared encoder with task-specific decoders for real-time processing.
- **Cost-Effective**: Eliminates the need for expensive LiDAR sensors and HD maps.
- **Real-time Performance**: Achieves >60 FPS.
- **Indian Street Adaptation**: Specifically trained and optimized for the complexity of Indian traffic scenarios.

## Architecture

The BrainDrive architecture is inspired by the human visual cortex, which processes different visual cues simultaneously through a shared early visual pathway followed by specialized regions:

```
┌───────────┐    ┌───────────────┐
│           │    │ Segmentation  │
│           │───▶│    Decoder    │───▶ Pixel-level Scene Understanding
│           │    └───────────────┘
│  Shared   │    ┌───────────────┐
│  Encoder  │───▶│     Depth     │───▶ 3D Structure Estimation
│ (ResNet)  │    │    Decoder    │
│           │    └───────────────┘
│           │    ┌───────────────┐
│           │───▶│   Detection   │───▶ Object Localization
└───────────┘    │    Decoder    │
                 └───────────────┘
```

### Components:

1. **Encoder**: A ResNet-based backbone that processes raw camera input.
2. **Segmentation Decoder**: A U-Net style architecture for pixel-wise classification of road elements.
3. **Depth Decoder**: Predicts monocular depth using a specialized loss function.
4. **Detection Decoder**: An anchor-free object detection head that detects and classifies traffic participants.

## Performance

| Task | Metric | Performance |
|------|--------|-------------|
| Segmentation | Dice Score | 0.85 |
| Depth Estimation | RMSE | 0.41m |
| Object Detection | mAP @ 0.5 IOU | 0.78 |
| Overall | FPS (RTX 3080) | >60 |

## Applications

- **Cost-effective autonomous driving** for emerging markets
- **Advanced driver assistance systems (ADAS)** with reduced hardware requirements
- **Traffic analytics** and smart city infrastructure
- **Mobile robotics** in unstructured environments

## Challenges Addressed

### Indian Traffic Scenario
Our system is specifically designed to handle the unique challenges of Indian traffic:

- Dense and heterogeneous traffic with various types of vehicles
- Unstructured road conditions with unclear lane markings
- Complex interactions between traffic participants
- Varied lighting and weather conditions

<p align="center">
  <img src="docs/assets/indian_traffic.png" alt="Indian Traffic Scenario" width="800"/>
</p>

### Technical Challenges
- **Task Interference**: We implemented novel loss balancing techniques to ensure each task benefits from shared representation learning without negative transfer.
- **Computational Efficiency**: The architecture is optimized for parallel computation on GPUs with minimal memory footprint.
- **Real-time Processing**: Special attention to inference speed while maintaining accuracy.

## Dataset

The model is trained on a custom dataset comprising:
- 50,000+ annotated images of Indian street scenes
- Annotations for 15 object classes relevant to autonomous driving
- Pixel-level semantic segmentation masks with 35 classes
- Depth ground truth from calibrated sensor setup

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/braindrive.git
cd braindrive

# Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```python
from braindrive.model import MTL_Model
from braindrive.dataset import MTL
from torch.utils.data import DataLoader

# Initialize model
model = MTL_Model(n_classes=35, device='cuda')

# Load datasets
train_dataset = MTL("path/to/train_dataset.csv")
val_dataset = MTL("path/to/val_dataset.csv")

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Train the model
# See training script for implementation details
```

### Inference

```python
import torch
import cv2
import numpy as np
from braindrive.model import MTL_Model
from braindrive.utils import labels_to_cityscapes_palette

# Load model
model = MTL_Model(n_classes=35, device='cuda')
model.load_state_dict(torch.load("path/to/model_weights.pth"))
model.eval()

# Process image
image = cv2.imread("test_image.jpg")
image = cv2.resize(image, (640, 480))
# Convert to tensor and normalize
# ...

# Forward pass
with torch.no_grad():
    seg_maps, depth_maps, detection_maps = model(image_tensor)

# Visualize results
# ...
```

## Acknowledgements

- This work was inspired by the multi-task learning approaches in computer vision
- Special thanks to [Gen Intelligence](https://www.linkedin.com/company/gen-intelligence/) for their support and collaboration
- We acknowledge the use of PyTorch, OpenCV, and other open-source libraries
