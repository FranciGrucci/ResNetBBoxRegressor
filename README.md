# ResNetBBoxRegressor

A PyTorch-based deep learning project for object detection and bounding box regression using ResNet architecture.

## Overview

This project implements a bounding box regression model built on top of ResNet, designed for object detection tasks. The model combines the powerful feature extraction capabilities of ResNet with custom regression heads for precise bounding box prediction.

## Features

- **ResNet-based Architecture**: Leverages pre-trained ResNet models for robust feature extraction
- **Bounding Box Regression**: Accurate prediction of object locations and dimensions
- **Data Augmentation**: Comprehensive augmentation pipeline including resize, color jitter, and normalization
- **Flexible Training**: Configurable training parameters and optimization strategies
- **Visualization Tools**: Built-in utilities for visualizing predictions and training progress

## Project Structure

```
ResNetBBoxRegressor/
├── data.yaml              # Data configuration and paths
├── model.py               # Core model architecture
├── train.py               # Training script
├── train_vecchio.py       # Legacy training implementation
├── visualize_bbox.py      # Bounding box visualization utilities
├── dataset.py             # Dataset loading and preprocessing
├── check_labels.py        # Label validation and verification
├── main.py                # Main execution script
├── landmark.py            # Landmark detection utilities
└── training_config.json   # Training configuration parameters
```

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- NumPy
- PIL (Pillow)
- Additional dependencies as specified in requirements

## Installation

1. Clone the repository:
```bash
git clone https://github.com/FranciGrucci/ResNetBBoxRegressor.git
cd ResNetBBoxRegressor
```

2. Install dependencies:
```bash
pip install torch torchvision numpy pillow
```

## Usage

### Training

To train the model, run:

```bash
python train.py
```

The training script includes:
- Data augmentation with resize, color jitter, and normalization
- Custom loss function combining coordinate loss and confidence loss
- Learning rate scheduling with ReduceLROnPlateau
- Model checkpointing and progress tracking

### Configuration

Modify `training_config.json` to adjust:
- Learning rate and optimization parameters
- Batch size and number of epochs
- Data augmentation settings
- Model architecture parameters

### Data Format

The model expects data in the following format:
- Images in standard formats (JPEG, PNG)
- Bounding box annotations with coordinates and labels
- Dataset configuration specified in `data.yaml`

## Model Architecture

The model combines:
- **Backbone**: ResNet feature extractor
- **Regression Head**: Custom layers for bounding box coordinate prediction
- **Loss Function**: Combination of L1 loss for coordinates and confidence loss

## Key Features

### Data Augmentation
```python
transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])  # ImageNet normalization
])
```

### Training Loop
- Comprehensive training function with debugging capabilities
- Subset training option for development and testing
- Progress tracking and loss monitoring

## Visualization

Use `visualize_bbox.py` to:
- Visualize training data with bounding boxes
- Display model predictions
- Compare ground truth vs predicted boxes

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Built with PyTorch framework
- ResNet architecture from torchvision
- Inspired by modern object detection methodologies

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: Make sure to update the repository URL, add proper license file, and include any additional dependencies in a `requirements.txt` file.
