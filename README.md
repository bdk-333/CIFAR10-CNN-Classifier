# CNN Architecture

![CNN Architecture](image.png)


A Convolutional Neural Network designed for CIFAR-10 image classification, implemented in PyTorch.

## Architecture Overview

This CNN follows a classic architecture pattern with three convolutional blocks followed by fully connected layers. The network progressively reduces spatial dimensions while increasing feature depth, making it suitable for the 32×32 RGB images in the CIFAR-10 dataset.

## Network Architecture

### Input Layer
- **Input Shape**: `3 × 32 × 32`
- **Data Type**: RGB images
- **Batch Size**: 64
- **Dataset**: CIFAR-10 (10 classes)

### Convolutional Layers

#### Conv1 Block
```python
nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
nn.ReLU()
nn.MaxPool2d(kernel_size=2, stride=2)
```
- **Input**: `3 × 32 × 32`
- **After Conv**: `6 × 32 × 32`
- **After MaxPool**: `6 × 16 × 16`
- **Parameters**: `(3 × 3 × 3 + 1) × 6 = 162`

#### Conv2 Block
```python
nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1)
nn.ReLU()
nn.MaxPool2d(kernel_size=2, stride=2)
```
- **Input**: `6 × 16 × 16`
- **After Conv**: `12 × 16 × 16`
- **After MaxPool**: `12 × 8 × 8`
- **Parameters**: `(3 × 3 × 6 + 1) × 12 = 660`

#### Conv3 Block
```python
nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
nn.ReLU()
nn.MaxPool2d(kernel_size=2, stride=2)
```
- **Input**: `12 × 8 × 8`
- **After Conv**: `24 × 8 × 8`
- **After MaxPool**: `24 × 4 × 4`
- **Parameters**: `(3 × 3 × 12 + 1) × 24 = 2,616`

### Fully Connected Layers

#### Flatten Layer
- **Input**: `24 × 4 × 4`
- **Output**: `384` (flattened vector)
- **Operation**: Reshapes 3D feature maps to 1D vector

#### FC1 Layer
```python
nn.Linear(in_features=384, out_features=500)
nn.ReLU()
nn.Dropout(p=0.5)
```
- **Input**: `384`
- **Output**: `500`
- **Parameters**: `384 × 500 + 500 = 192,500`
- **Regularization**: 50% Dropout

#### Output Layer
```python
nn.Linear(in_features=500, out_features=10)
```
- **Input**: `500`
- **Output**: `10` (CIFAR-10 classes)
- **Parameters**: `500 × 10 + 10 = 5,010`

## Information Flow

The network processes images through the following stages:

1. **Input Processing**: 32×32×3 RGB images are fed into the network
2. **Feature Extraction**: Three convolutional blocks extract hierarchical features
3. **Spatial Reduction**: MaxPooling reduces spatial dimensions by half at each stage
4. **Feature Depth Increase**: Number of channels grows from 3 → 6 → 12 → 24
5. **Vectorization**: 3D feature maps are flattened to 1D vector (384 elements)
6. **Classification**: Fully connected layers map features to class probabilities

### Dimension Progression

| Stage | Operation | Input Shape | Output Shape | Spatial Size | Channels |
|-------|-----------|-------------|--------------|--------------|----------|
| Input | - | - | `3×32×32` | 1024 | 3 |
| Conv1+Pool | Conv+ReLU+MaxPool | `3×32×32` | `6×16×16` | 256 | 6 |
| Conv2+Pool | Conv+ReLU+MaxPool | `6×16×16` | `12×8×8` | 64 | 12 |
| Conv3+Pool | Conv+ReLU+MaxPool | `12×8×8` | `24×4×4` | 16 | 24 |
| Flatten | Reshape | `24×4×4` | `384×1` | 384 | 1 |
| FC1 | Linear+ReLU+Dropout | `384×1` | `500×1` | 500 | 1 |
| Output | Linear | `500×1` | `10×1` | 10 | 1 |

## Parameter Count

### Layer-wise Parameter Breakdown

| Layer | Type | Parameters | Calculation |
|-------|------|------------|-------------|
| Conv1 | Convolutional | 162 | `(3×3×3 + 1) × 6 = 162` |
| Conv2 | Convolutional | 660 | `(3×3×6 + 1) × 12 = 660` |
| Conv3 | Convolutional | 2,616 | `(3×3×12 + 1) × 24 = 2,616` |
| FC1 | Linear | 192,500 | `384 × 500 + 500 = 192,500` |
| Output | Linear | 5,010 | `500 × 10 + 10 = 5,010` |

### **Total Parameters: 200,948**

#### Parameter Distribution
- **Convolutional Layers**: 3,438 parameters (1.7%)
- **Fully Connected Layers**: 197,510 parameters (98.3%)

## Key Design Features

### Architectural Choices
- **Progressive Channel Increase**: Features become more abstract at deeper layers
- **Spatial Dimension Reduction**: Focuses on important features while reducing computation
- **Same Padding**: Preserves spatial dimensions before pooling
- **ReLU Activation**: Introduces non-linearity and prevents vanishing gradients

### Regularization Techniques
- **Dropout**: 50% dropout in FC1 layer prevents overfitting
- **MaxPooling**: Provides translation invariance and reduces parameters

### Optimization Considerations
- **Parameter Efficiency**: Majority of parameters are in FC layers
- **Feature Hierarchy**: Convolutional layers extract low-to-high level features
- **Computational Efficiency**: Small network suitable for CIFAR-10's 32×32 images

## Forward Pass Example

```python
# Input: batch_size=64, channels=3, height=32, width=32
x = torch.randn(64, 3, 32, 32)

# Conv1 Block: (64, 3, 32, 32) → (64, 6, 16, 16)
x = self.conv1(x)

# Conv2 Block: (64, 6, 16, 16) → (64, 12, 8, 8)
x = self.conv2(x)

# Conv3 Block: (64, 12, 8, 8) → (64, 24, 4, 4)
x = self.conv3(x)

# Flatten: (64, 24, 4, 4) → (64, 384)
x = x.view(-1, 24*4*4)

# FC layers: (64, 384) → (64, 10)
x = self.fct1(x)

# Output: (64, 10) - logits for 10 classes
```

## Usage

```python
# Initialize the model
model = CIFAR_CNN()

# Check total parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
```

## Model Summary

- **Task**: Multi-class image classification (CIFAR-10)
- **Input**: 32×32×3 RGB images
- **Output**: 10-class probability distribution
- **Architecture**: 3 Conv blocks + 2 FC layers
- **Total Parameters**: 200,948
- **Key Features**: Progressive feature extraction, dropout regularization, efficient design

This architecture balances model complexity with performance, making it suitable for the CIFAR-10 dataset while maintaining computational efficiency.