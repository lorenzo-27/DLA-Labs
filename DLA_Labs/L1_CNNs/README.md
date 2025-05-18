# CNN and MLP Models Implementation

This repository contains implementations of Convolutional Neural Networks (CNN) and Multi-Layer Perceptrons (MLP) for image classification tasks, with support for both standard and residual architectures.

## Table of Contents

- [Theory](#theory)
  - [Convolutional Neural Networks](#convolutional-neural-networks)
  - [Multi-Layer Perceptrons](#multi-layer-perceptrons)
  - [Residual Connections](#residual-connections)
- [Implementation](#implementation)
  - [Network Architectures](#network-architectures)
  - [Datasets](#datasets)
- [Usage](#usage)
  - [Command Line Arguments](#command-line-arguments)
  - [Configuration](#configuration)
  - [Example Commands](#example-commands)
- [Tools](#tools)
  - [MLP Depth Comparison Tool](#mlp-depth-comparison-tool)
  - [Network Visualization Tool](#network-visualization-tool)
- [Installation](#installation)
- [Results Interpretation](#results-interpretation)
- [Project Structure](#project-structure)
- [Common Issues](#common-issues)
- [References](#references)

## Theory

### Convolutional Neural Networks

Convolutional Neural Networks (CNNs) are specialized deep learning architectures designed primarily for processing structured grid-like data, such as images. They employ the mathematical operation of convolution, which allows the network to extract spatial hierarchies of features. The key components include:

1. **Convolutional layers**: Apply learnable filters to input data, producing feature maps that highlight important patterns
2. **Pooling layers**: Reduce the spatial dimensions through downsampling operations (typically max or average pooling)
3. **Fully connected layers**: Connect all neurons to all activations in the previous layer to produce final predictions

The architecture of a CNN naturally preserves spatial information, making these networks especially effective for computer vision tasks like image classification, object detection, and segmentation.

### Multi-Layer Perceptrons

Multi-Layer Perceptrons (MLPs) are the standard fully-connected neural network architecture. They consist of:

1. **Input layer**: Receives the flattened input data
2. **Hidden layers**: Multiple layers of neurons where each neuron connects to all neurons in the previous layer
3. **Output layer**: Produces the final classification scores

While MLPs work on flattened inputs and lose spatial information, they can still perform reasonably well on simpler image classification tasks. They serve as useful baselines for comparing with more advanced architectures.

### Residual Connections

Residual connections (or "skip connections") were introduced in the ResNet architecture to address the degradation problem in very deep networks. They work by:

1. Adding shortcut connections that skip one or more layers
2. Creating residual blocks where the output is the sum of a transformed input and the original input
3. Allowing the network to learn residual mappings instead of direct mappings

The formula for a residual connection is:

$$y = F(x) + x$$

Where:
- $x$ is the input to the residual block
- $F(x)$ is the output of the transformation layers
- $y$ is the final output of the residual block

This approach allows for effective training of much deeper networks by providing alternative pathways for gradient flow during backpropagation.

## Implementation

### Network Architectures

The implementation provides both standard and residual variants for CNNs and MLPs:

#### CNN Architecture:
- **Standard CNN**: Consists of convolutional blocks (Conv2D + BatchNorm + ReLU) followed by max pooling layers and final fully connected layers
- **Residual CNN**: Employs residual blocks where each block contains two convolutional layers with a skip connection

#### MLP Architecture:
- **Standard MLP**: Multiple fully connected layers with batch normalization and ReLU activations
- **Residual MLP**: MLP blocks with skip connections between layers

Both architectures are flexible and can be configured with different layer sizes, filter counts, and other hyperparameters through the configuration files.

### Datasets

The implementation supports two popular image classification datasets:

1. **MNIST**: Handwritten digit recognition (10 classes, 28x28 grayscale images)
   - 60,000 training images
   - 10,000 test images

2. **CIFAR-10**: Object recognition (10 classes, 32x32 color images)
   - 50,000 training images
   - 10,000 test images

The datasets are automatically downloaded and processed with appropriate transformations:

- **Training images**: Data augmentation including random crops, horizontal flips, and normalization
- **Validation and test images**: Center crops and normalization

## Usage

### Command Line Arguments

To train a model:

```
python -m DLA_Labs.L1_CNNs.train --config PATH_TO_CONFIG_FILE
```

Options:
- `--config`: Path to the configuration YAML file (required)

### Configuration

The configuration files (`m_cnn.yaml` and `m_mlp.yaml`) allow customizing various aspects:

#### CNN Configuration (`m_cnn.yaml`):

```yaml
model:
  name: cnn
  input_shape: [3, 32, 32]  # [channels, height, width]
  num_classes: 10
  filters: [32, 64, 128, 256]  # Filters for each layer
  kernel_size: 3
  residual: false

training:
  learning_rate: 0.001
  batch_size: 256
  num_epochs: 50
  weight_decay: 0.0001
  checkpoint_dir: checkpoints/
  log_every: 10
  save_every: 5
  early_stopping_patience: 7
  optimizer: adam  # [adam, sgd]
  scheduler: reduce_lr  # [reduce_lr, cosine]

dataset:
  name: cifar10  # [mnist, cifar10]
  data_dir: data/
  val_size: 5000  # Validation set size
```

#### MLP Configuration (`m_mlp.yaml`):

```yaml
model:
  name: mlp
  input_shape: [1, 28, 28]  # [channels, height, width]
  layer_sizes: [128, 64, 32, 32, 10]  # Layer sizes
  residual: false

training:
  learning_rate: 0.001
  batch_size: 128
  num_epochs: 20
  weight_decay: 0.0001
  checkpoint_dir: checkpoints/
  log_every: 10
  save_every: 5
  early_stopping_patience: 5
  optimizer: adam  # [adam, sgd]
  scheduler: reduce_lr  # [reduce_lr, cosine]

dataset:
  name: mnist  # [mnist, cifar10]
  data_dir: data/
  val_size: 5000  # Validation set size
```

### Example Commands

1. **Train a standard CNN on CIFAR-10:**

   ```
   python -m DLA_Labs.L1_CNNs.train --config DLA_Labs/L1_CNNs/models/m_cnn.yaml
   ```

2. **Train a residual MLP on MNIST:**
   
   ```
   # First modify the config to set residual: true
   python -m DLA_Labs.L1_CNNs.train --config DLA_Labs/L1_CNNs/models/m_mlp.yaml
   ```

## Tools

### MLP Depth Comparison Tool

The `compare_mlp_depth.py` script is designed to systematically investigate the impact of network depth and residual connections on MLP performance. This tool allows to understand the degradation problem in deep networks and how residual connections can help address it.

#### Overview of Functionality

The script runs experiments with MLPs of various depths, both with and without residual connections, and compares their performance on the MNIST dataset. It generates visualizations to help understand:

1. How validation loss evolves during training
2. The relationship between network depth and test accuracy
3. Gradient magnitudes across epochs and network layers
4. The vanishing/exploding gradient problem

#### Key Components

##### Configuration Generation
- **Function**: `create_configs(depths, residual_options, epochs)`
- **Purpose**: Creates training configurations for each combination of network depth and residual option
- **Details**: 
  - Automatically generates appropriate layer sizes based on depth
  - Starts with 128 neurons, then 64, then 32 for deeper layers
  - Creates identical configurations for both standard and residual variants

##### Training Process
- **Functions**: `train_epoch()`, `evaluate()`, `train_model()`
- **Purpose**: Handle the training loop, evaluation, and model monitoring
- **Key Features**:
  - Early stopping to prevent overfitting
  - Learning rate scheduling for better convergence
  - Gradient magnitude tracking during training
  - Recording of training/validation metrics

##### Gradient Analysis
- **Function**: `analyze_gradients()`
- **Purpose**: Examines gradient flow through different layers of the network
- **Implementation**: 
  - Performs forward and backward passes on a single batch
  - Calculates gradient norms for each layer's weights
  - Handles both standard MLP and ResidualMLP architectures

##### Experiment Execution
- **Function**: `run_experiments()`
- **Purpose**: Orchestrates the entire experimental process
- **Features**:
  - Creates models with specified depths and residual configurations
  - Trains each model and collects performance metrics
  - Saves intermediate results to prevent data loss
  - Performs gradient analysis for each model

##### Visualization
- **Function**: `plot_results()`
- **Purpose**: Generates informative plots from experimental results
- **Output**:
  - Validation loss curves for different network configurations
  - Test accuracy vs. network depth comparison
  - Gradient magnitude trends during training
  - Layer-wise gradient distribution for deep networks

#### Usage

```
python -m DLA_Labs.L1_CNNs.compare_mlp_depth --depths 5 10 20 30 50 100 --epochs 20
```

Options:
- `--depths`: Network depths to experiment with (default: 5, 10, 20, 30, 50, 100)
- `--load`: Load previously saved results instead of running new experiments
- `--epochs`: Number of training epochs for each experiment (default: 20)

#### Results and Interpretation

The script generates several plots in the `mlp_results` directory:

1. **validation_losses.png**: Shows how validation loss evolves during training for different network configurations
2. **test_accuracies.png**: Plots test accuracy against network depth for both standard and residual MLPs
3. **gradient_magnitudes.png**: Tracks average gradient magnitude throughout training
4. **layer_gradient_comparison.png**: Compares gradient magnitudes across layers in deep networks

<p align="center">
<img  src="https://github.com/lorenzo-27/DLA-Labs/blob/master/assets/test_accuracies.png" width="90%" height="90%"/>
</p>

These visualizations help identify:
- The depth at which standard MLPs begin to suffer from degradation
- How effectively residual connections mitigate this problem
- Patterns in gradient flow that explain performance differences

The script also generates a `final_results.pkl` and an `intermediate_results.pkl`, both of which contain all the numeric results. A Markdown table of the data in `final_results.pkl` is presented here:

| Depth | Residual | Train Loss | Val Loss | Train Acc | Val Acc | Test Acc | Epochs |
|------------|-----------|---------------------|-------------------|--------------------|------------------|----------|-------------------|
| 5          | False     | 0.0040              | 0.0701            | 99.89              | 98.24            | 98.25    | 20                |
| 5          | True      | 0.0030              | 0.0640            | 99.92              | **98.34**            | 98.28    | 20                |
| 10         | False     | 0.0105              | 0.0817            | 99.73              | 98.30            | 98.17    | 20                |
| 10         | True      | 0.0281              | 0.0816            | 99.08              | 97.78            | 97.62    | 13                |
| 20         | False     | 0.0681              | 0.1029            | 98.23              | 97.62            | 97.53    | 20                |
| 20         | True      | 0.0369              | 0.0817            | 98.83              | 97.84            | 97.75    | 14                |
| 30         | False     | 0.1675              | 0.1961            | 95.61              | 95.66            | 95.69    | 20                |
| 30         | True      | 0.0104              | 0.0944            | 99.68              | 98.16            | 98.14    | 20                |
| 50         | False     | 1.2274              | 1.2568            | 58.39              | 58.74            | 59.90    | 14                |
| 50         | True      | 0.0142              | 0.0865            | 99.59              | 98.12            | 97.75    | 20                |
| 100        | False     | 2.3013              | 2.3086            | 11.27              | 10.70            | 11.35    | 17                |
| 100        | True      | 0.0422              | 0.1002            | 98.79              | 97.96            | 97.68    | 20                |

### Network Visualization Tool

The `visualize_network.py` module provides functionality for analyzing and visualizing neural network architectures.

#### Features:
- Uses the `torchinfo` library to generate detailed summaries of model structure
- Displays layer-by-layer information including input/output shapes and parameter counts
- Logs model output shapes for verification

#### Usage:
```python
from DLA_Labs.L1_CNNs.visualize_network import visualize

# Example usage
visualize(model, "model_name", input_tensor)
```

The function produces a formatted console output showing the network's structure and important statistics, making it easier to debug architecture issues and understand model complexity.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/username/DLA-Labs.git
   cd DLA-Labs
   ```

2. Install the dependencies with pip or conda. The `requirements.txt` file contains all the necessary packages:

    ```
    pip install -r requirements.txt
    ``` 
    or
    ```
    conda install --file requirements.txt
    ```

3. Run the setup to install the package in development mode:

   ```
   pip install -e .
   ```

4. When first run, the script will automatically download the dataset (MNIST or CIFAR-10) to the specified data directory (default: `./data/`).

## Results Interpretation

During and after training, you can monitor the following metrics:

1. **Training Loss and Accuracy**: Displayed in the console and logged to TensorBoard
2. **Validation Loss and Accuracy**: Displayed after each epoch
3. **Test Accuracy**: Displayed after training is complete

The implementation includes:

- **Early stopping**: Training stops when validation performance doesn't improve for a specified number of epochs
- **Learning rate scheduling**: Reduces the learning rate when performance plateaus
- **Model checkpoints**: Saves model states at regular intervals and the best-performing model

TensorBoard logs can be viewed with:

```
tensorboard --logdir=runs/
```

---

Best results averaged over 10 runs on the validation set. The depth and number of parameters are the same between residual and non-residual implementations. Every result has a variance of ±0.4%.
- MNIST
  - CNN without residuals: 98.8
  - **CNN with residuals: 99.2**
  - MLP without residuals: 97.4
  - MLP with residuals: 98.2

- CIFAR 10
  - CNN without residuals: 84.7
  - **CNN with residuals: 85.3**
  - MLP without residuals: 55.8
  - MLP with residuals: 57.2
 

## Project Structure

```
DLA_Labs/L1_CNNs/
├── dataset.py                  # Dataset loading and preprocessing
├── models/
│   ├── __init__.py             # Models package initialization
│   ├── m_cnn.py                # CNN model implementation
│   ├── m_cnn.yaml              # CNN model configuration
│   ├── m_mlp.py                # MLP model implementation
│   └── m_mlp.yaml              # MLP model configuration
├── train.py                    # Training functionality
├── compare_mlp_depth.py        # MLP depth comparison tool
└── visualize_network.py        # Network visualization utility
```

## Common Issues

- **CUDA Out of Memory**: Reduce the batch size in the configuration file
- **Slow convergence**: Try adjusting the learning rate or using a different optimizer
- **Overfitting**: Add more regularization (e.g., dropout, weight decay) or data augmentation
- **Dataset download issues**: If automatic download fails, manually download MNIST or CIFAR-10 and place in the data directory

## References

1. Bagdanov, A. D., DLA course material (2025)

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. _IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
