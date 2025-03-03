# Multi-Task Learning for Cost-Effective Autonomous Driving

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-red.svg)](https://pytorch.org/)

## Overview

This repository implements a Multi-Task Learning (MTL) architecture that performs multiple computer vision tasks in a single pass to enable cost-effective autonomous driving perception. Our system achieves real-time performance with a single forward pass taking only 0.007944s.

> **🚀 Featured Insight:** Discover how our model handles real-world traffic scenarios in India! [Check out this insightful LinkedIn post](https://www.linkedin.com/posts/gen-intelligence_indian-street-activity-6834375620462489600-Theg?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC0zn_AB35C2DW6c0JrixmdBygc2o4dtvbc).

## Key Features

- **Multi-Task Learning (MTL)**: Combines 2D object detection, semantic segmentation, and monocular depth estimation in a single neural network
- **Efficient Architecture**: Shared encoder with task-specific decoders for real-time processing
- **Cost-Effective**: Eliminates the need for expensive hardware by performing multiple tasks with a single model
- **Real-time Performance**: Achieves >60 FPS on V100 GPU
- **Indian Street Adaptation**: Trained and tested on the India Driving Dataset from IIIT Hyderabad

## Architecture

The MTL architecture uses a shared encoder-decoder design:

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

1. **Encoder**: A ResNet-based backbone that processes raw camera input
2. **Segmentation Decoder**: U-Net style architecture for pixel-wise classification (35 classes)
3. **Depth Decoder**: Predicts monocular depth using a specialized loss function
4. **Detection Decoder**: Anchor-free object detection for 15 object classes

## Performance

| Task | Metric | Performance |
|------|--------|-------------|
| Segmentation | IOU | 0.979 |
| Segmentation | Pixel Accuracy | 0.943 |
| Depth Estimation | A1 | 0.852 |
| Depth Estimation | RMSE | 0.031 |
| Object Detection | mAP @ 0.5 IOU | 0.256 |
| Object Detection | Average IOU | 0.726 |

## Training Methodology

Our training approach includes several specialized techniques:

- **Knowledge Distillation for Depth**: We used distillation to train our depth decoder, leveraging results from a state-of-the-art depth estimation model
- **PCGrad Optimizer**: Used for managing task interference between the different learning objectives
- **Custom Loss Functions**: 
  - Dice-Focal loss for segmentation
  - Specialized depth loss including gradient terms
  - Detection loss combining heatmap, regression, and dimension losses

## Dataset

The model is trained on the India Driving Dataset from IIIT Hyderabad, which includes:
- Images of Indian street scenes with diverse traffic patterns
- Annotations for object detection (15 classes)
- Pixel-level semantic segmentation (35 classes)
- Depth ground truth

## MTL 2.0 Improvements

The latest version of our MTL architecture (MTL 2.0) includes several improvements:
- Significantly improved depth estimation performance
- Temporally consistent results without explicit temporal monitoring
- Reduced computational cost while maintaining accuracy
- Tested on V100 GPU with 32 GB VRAM

## Usage

### Model Definition

```python
from encoder import Encoder
from decoders import seg_decoder, objdet_Decoder

class MTL_Model(nn.Module):
    def __init__(self, n_classes=35, device='cuda'):
        super(MTL_Model, self).__init__()
        self.encoder = Encoder(device=device)
        self.seg_decoder = seg_decoder(n_classes, device=device)
        self.dep_decoder = seg_decoder(n_classes=1, device=device)
        self.obj_decoder = objdet_Decoder(n_classes=15, device=device)
        self.to(device)
        
    def forward(self, X):
        outputs = self.encoder(X)
        seg_maps = self.seg_decoder(outputs)
        depth_maps = self.dep_decoder(outputs)
        detection_maps = self.obj_decoder(outputs)
        return (seg_maps, torch.sigmoid(depth_maps), detection_maps)
```

### Inference

```python
model = MTL_Model(device=device)
model.load_state_dict(torch.load("model_path.pth", map_location=device))
model.eval()

# Process a single image
with torch.no_grad():
    rgb = preprocess(image).unsqueeze(0).to(device)
    seg_maps, depth_maps, detection_maps = model(rgb)
    
    # Extract results
    seg_pred = torch.argmax(torch.softmax(seg_maps, dim=1).squeeze(0), dim=0)
    depth_pred = depth_maps.squeeze().cpu().numpy()
    
    # For detection, decode the heatmap
    hmap, regs, w_h_ = zip(*detection_maps)
    detections = ctdet_decode(hmap[0], regs[0], w_h_[0])
```

## Acknowledgements

- This work was built using PyTorch and related computer vision libraries
- We acknowledge the India Driving Dataset from IIIT Hyderabad
