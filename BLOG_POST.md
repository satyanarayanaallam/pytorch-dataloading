# Training ResNet18 on Document Classification: A Complete PyTorch Guide

## Introduction

Document classification is a fundamental task in machine learning, with applications ranging from email filtering to automated document organization. In this guide, we'll walk through a practical implementation of training a **ResNet18 convolutional neural network** on the **RVL-CDIP dataset** — a large-scale benchmark containing 400,000 images of scanned documents classified into 16 categories.

This tutorial covers the complete machine learning pipeline: data preparation, model adaptation, training, validation, checkpointing, and inference using PyTorch.

---

## What is the RVL-CDIP Dataset?

The **Ryerson Vision Lab Complex Document Image Processing (RVL-CDIP)** dataset contains document images in 16 categories:

- Advertisement
- Budget
- Email
- File Folder
- Form
- Handwritten
- Invoice
- Letter
- Memo
- News Article
- Presentation
- Questionnaire
- Resume
- Scientific Publication
- Scientific Report
- Specification

Each document type has distinct visual characteristics — invoices have structured layouts with numbers, letters have formal formatting, handwritten documents show natural pen strokes. This diversity makes it an excellent testbed for transfer learning.

---

## Architecture Overview: Transfer Learning with ResNet18

Rather than training a CNN from scratch, we leverage **transfer learning** using ResNet18, a proven 18-layer residual network pre-trained on ImageNet. Here's why this approach is powerful:

1. **Pre-trained weights**: The network already understands basic visual features (edges, textures, shapes)
2. **Fine-tuning**: We replace only the final classification layer to adapt the model to 16 document classes
3. **Efficiency**: Training time and computational requirements are dramatically reduced
4. **Performance**: Pre-trained models often outperform models trained from scratch on smaller datasets

---

## Step 1: Setting Up the Environment & Dependencies

Before diving into code, install required packages:

```python
! pip3 install certifi
```

Then import essential PyTorch and torchvision modules:

```python
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image, UnidentifiedImageError
import ssl
import os

# SSL fix for secure downloads
ssl._create_default_https_context = ssl._create_unverified_context
```

---

## Step 2: Building a Robust Data Pipeline

### Handling Corrupted Images

In real-world datasets, corrupted or unreadable files are common. We implement a **safe loader function** that gracefully handles these cases:

```python
def safe_loader(path):
    try:
        # Try to open with PIL
        return Image.open(path).convert("RGB")
    except UnidentifiedImageError:
        print(f"Skipping bad file: {path}")
        # Return a dummy image (black 224x224) so DataLoader doesn't crash
        return Image.new("RGB", (224, 224))
```

This ensures that corrupted files don't crash the training loop — a critical consideration for large-scale datasets.

### Data Transformations

Image transformations prepare raw images for the CNN:

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize to CNN input size
    transforms.ToTensor(),            # Convert PIL Image to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize pixel values
])
```

Why these transformations?
- **Resize(224, 224)**: ResNet18 expects 224×224 input images
- **ToTensor()**: Converts pixel values from [0, 255] to [0, 1]
- **Normalize()**: Standardizes pixel distributions for better gradient flow

### Loading Data with ImageFolder

PyTorch's `ImageFolder` dataset automatically infers class labels from directory structure:

```python
dataset = datasets.ImageFolder(root="data/rvl_cdip", loader=safe_loader, transform=transform)

print(len(dataset))          # Total number of images
print(dataset.classes)       # List of class names
print(dataset.class_to_idx)  # Mapping class → label
```

### Train-Validation-Test Split

We follow best practices by splitting data into three subsets:

```python
train_size = int(0.7 * len(dataset))   # 70% for training
test_size = int(0.2 * len(dataset))    # 20% for testing
val_size = len(dataset) - train_size - test_size  # 10% for validation

train_dataset, test_dataset, val_dataset = random_split(
    dataset, [train_size, test_size, val_size]
)
```

**Why this split?**
- **Training set (70%)**: Used to update model weights
- **Validation set (10%)**: Used to tune hyperparameters and select the best model
- **Test set (20%)**: Used for final evaluation on unseen data

### Creating DataLoaders

DataLoaders enable efficient batch processing:

```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
```

Key parameters:
- **batch_size=32**: Process 32 images simultaneously for GPU efficiency
- **shuffle=True** (training): Randomize batch order to prevent overfitting
- **shuffle=False** (validation/test): Consistent evaluation results

---

## Step 3: Loading and Adapting ResNet18

### Loading Pre-trained Weights

We load ResNet18 without automatic ImageNet weight download:

```python
model = models.resnet18(weights=None)  # No auto-download
state_dict = torch.load("resnet18-5c106cde.pth", weights_only=False)
model.load_state_dict(state_dict)
```

### Adapting for 16-Class Classification

ResNet18 comes with 1000-class output (ImageNet). We replace the final layer for 16 document classes:

```python
num_features = model.fc.in_features  # 512 for ResNet18
model.fc = nn.Linear(num_features, 16)  # 16 document classes
```

This replacement is the essence of **transfer learning** — we keep 99% of the model unchanged and only train the final layer heavily while fine-tuning the rest.

---

## Step 4: Setting Up Training Infrastructure

### Loss Function and Optimizer

For multi-class classification, we use **CrossEntropyLoss** and the **Adam optimizer**:

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

**Why Adam?** It adapts learning rates per parameter, converging faster than standard SGD.

### Device Handling (CPU/GPU/MPS)

Modern PyTorch supports multiple accelerators:

```python
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")

print("Using device:", device)
model.to(device)
```

This code automatically:
- Uses **CUDA** on NVIDIA GPUs
- Falls back to **MPS** (Metal Performance Shaders) on Apple Silicon
- Uses **CPU** as last resort

---

## Step 5: Training Loop with Checkpointing

### The Complete Training Pipeline

```python
os.makedirs("checkpoints", exist_ok=True)

best_acc = 0.0
for epoch in range(5):  # 5 epochs
    # Training phase
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()           # Clear old gradients
        outputs = model(images)         # Forward pass
        loss = criterion(outputs, labels)
        loss.backward()                 # Backpropagation
        optimizer.step()                # Update weights
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # Validation phase
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")

    # Checkpointing
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "accuracy": acc,
        "loss": loss.item()
    }
    torch.save(checkpoint, f"checkpoints/epoch_{epoch+1}.pth")
    
    # Save best model
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "checkpoints/best_model.pth")
        print("✓ Best model updated and saved!")
```

### Understanding the Training Loop

1. **model.train()**: Enables dropout and batch normalization training behavior
2. **optimizer.zero_grad()**: Prevents gradient accumulation
3. **loss.backward()**: Computes gradients via backpropagation
4. **optimizer.step()**: Updates model parameters using computed gradients
5. **model.eval()**: Disables dropout/batch norm, runs in inference mode
6. **torch.no_grad()**: Disables gradient computation (saves memory)

### Why Checkpointing?

Saving model states after each epoch allows us to:
- Resume training if interrupted
- Track performance progression
- Recover the best model based on validation accuracy
- Avoid re-training if experiments fail later

---

## Step 6: Evaluating on Test Set

After training completes, evaluate on the held-out test set:

```python
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")
```

The test set provides an unbiased estimate of real-world performance since the model has never seen these images during training.

---

## Step 7: Inference on New Documents

Once satisfied with performance, use the best model for inference:

```python
# Load the best saved model
state_dict = torch.load("checkpoints/best_model.pth", map_location=device)
model.load_state_dict(state_dict)

# Run inference
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        # preds now contains predicted class indices
        print(f"Predictions: {preds}")
```

---

## Step 8: Resuming Training from Checkpoints

If you want to continue training from a saved checkpoint:

```python
# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']

# Continue training
for epoch in range(start_epoch, num_epochs + 5):
    # [same training code as before]
    pass
```

This is particularly useful for:
- Extending training beyond initial plan
- Fine-tuning hyperparameters
- Avoiding loss of progress due to system failures

---

## Key Takeaways

### Data Handling Best Practices
✓ Implement robust error handling for corrupted files  
✓ Use appropriate image transformations (resize, normalize)  
✓ Properly split data into train/validation/test sets  
✓ Leverage DataLoaders for efficient batch processing  

### Model Training Best Practices
✓ Use transfer learning to reduce training time  
✓ Implement checkpointing to save progress  
✓ Monitor validation accuracy to detect overfitting  
✓ Use device-agnostic code (CPU/GPU/MPS compatibility)  

### Production Considerations
✓ Save complete checkpoints (model + optimizer state)  
✓ Evaluate on held-out test set for unbiased performance estimate  
✓ Track metrics across epochs to understand learning dynamics  
✓ Use model.eval() during inference to disable dropout  

---

## Conclusion

We've built a complete document classification system using ResNet18 and PyTorch. This approach—combining transfer learning, robust data handling, and proper evaluation—forms the foundation for many real-world deep learning applications.

The techniques here extend beyond document classification: the same pipeline works for image classification, medical imaging, satellite imagery analysis, and more. By mastering these fundamentals, you have a template for tackling a wide range of computer vision problems.

**Happy training!**

---

## References

- [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html)
- [RVL-CDIP Dataset Paper](https://arxiv.org/abs/1502.08963)
- [ResNet Paper: Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Transfer Learning in Computer Vision - PyTorch Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
