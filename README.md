# DeepLearning-MiniProject1
Deep Learning Mini Project - ResNet for CIFAR-10
Project Overview
This project implements a custom ResNet-inspired convolutional neural network for image classification on the CIFAR-10 dataset. The model is designed to achieve high accuracy while maintaining a constraint of ≤5 million parameters. The training pipeline incorporates batch normalization, residual connections, data augmentation, and adaptive learning rate scheduling for improved performance.

What is ResNet?
ResNet (Residual Network) is a deep learning architecture introduced by He et al. (2015) that allows training very deep neural networks by addressing the vanishing gradient problem. The key innovation in ResNet is the residual block, which uses skip connections to bypass certain layers, enabling better gradient flow and improving convergence.

About the Dataset
CIFAR-10 is a widely used benchmark dataset for image classification, consisting of 60,000 color images (32x32 pixels) across 10 classes:

Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.
The dataset is split into 50,000 training and 10,000 test images.
Model Architecture
The model follows a ResNet-like architecture with four residual blocks, downsampling at strategic layers, and global average pooling before classification.

Input: CIFAR-10 images (32x32x3)
Initial Convolutional Layer:
7×7 convolution, 64 filters, stride=2
Batch Normalization + ReLU Activation
Residual Blocks:
Block 1: Conv(3x3, 64 → 64) → Conv(3x3, 64 → 64)
Block 2: Conv(3x3, 64 → 128, stride=2) → Conv(3x3, 128 → 128)
Block 3: Conv(3x3, 128 → 256, stride=2) → Conv(3x3, 256 → 256)
Block 4: Conv(3x3, 256 → 512, stride=2) → Conv(3x3, 512 → 512)
Global Average Pooling (1x1)
Fully Connected Layer: 512 → 10 (CIFAR-10 classes)
Output Layer: Softmax
Results and Performance
Achieved Accuracy: >90% on CIFAR-10
Model Parameters: ≤5 million
Optimization Techniques:
Data augmentation (random flip, crop, normalization)
Learning rate scheduling (StepLR / Cosine Annealing)
Regularization (Dropout, Weight Decay)
