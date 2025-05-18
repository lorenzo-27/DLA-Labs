# Class Activation Mapping (CAM) Implementation

This repository contains an implementation of the Class Activation Mapping (CAM) technique for visualizing which regions of an image are important for deep convolutional neural networks' classification decisions.

## Table of Contents

- [Theory](#theory)
- [Implementation](#implementation)
  - [Network Architecture](#network-architecture)
  - [Dataset](#dataset)
- [Usage](#usage)
  - [Command Line Arguments](#command-line-arguments)
  - [Configuration](#configuration)
  - [Example Commands](#example-commands)
- [Installation](#installation)
- [Results Interpretation](#results-interpretation)
- [Project Structure](#project-structure)
- [Common Issues](#common-issues)
- [References](#references)

## Theory

Class Activation Mapping (CAM) is a technique introduced in the following paper

> Learning Deep Features for Discriminative Localization" by Zhou et al. (CVPR'16, arXiv:1512.04150, 2015).

The key insight of CAM is that the convolutional layers of a CNN naturally preserve spatial information, which is then used by the fully connected (FC) layers for classification. By examining the weighted activations just before the global average pooling layer, we can visualize which regions of an input image are most important for the network's classification decision.

The technique works as follows:

1. Replace the fully connected layers at the end of the network with a Global Average Pooling (GAP) layer followed by a single fully connected layer
2. Train the network for the classification task
3. For visualization, compute a weighted sum of the feature maps in the final convolutional layer:
   - The weights are taken from the FC layer for the target class
   - This weighted sum creates a heatmap highlighting the discriminative regions

Mathematically, the class activation map for a particular class c is given by:

$$\text{CAM}_c(x, y) = \sum_k w^c_k f_k(x, y)$$

Where:

- $f_k(x, y)$ is the activation of unit $k$ in the last convolutional layer at spatial location $(x, y)$
- $w^c_k$ is the weight corresponding to class $c$ for unit $k$ in the fully connected layer

This approach allows us to visualize which parts of an image "activate" the neurons related to a specific class, providing visual explanations for the model's decisions without requiring any architectural changes during training.

## Implementation

### Network Architecture

The implementation uses a modified ResNet-18 architecture:

- Base model: ResNet-18 pre-trained on ImageNet
- Final fully connected layer replaced with a new one for Imagenette (10 classes)
- Class activation maps are computed using the weights of the final fully connected layer

The `ResNetCAM` class provides functionality for:

- Training the model
- Making predictions
- Extracting features for CAM computation
- Generating class activation maps for visualization

The model maintains the original ResNet architecture but adds functionality to extract intermediate features needed for CAM computation.

### Dataset

The implementation uses the **Imagenette** dataset, which is a smaller subset of ImageNet consisting of 10 easily distinguishable classes:

- tench (a type of fish)
- English springer (dog breed)
- cassette player
- chain saw
- church
- French horn
- golf ball
- gas pump
- parachute
- garbage truck

The dataset is automatically downloaded and prepared with appropriate transformations:

- Training images: random resized crops, horizontal flips, normalization
- Validation images: center crops, normalization

## Usage

### Command Line Arguments

The main script supports various arguments:

```
python -m DLA_Labs.L1_CNNs.CAM.main [OPTIONS]
```

Options:

- `--config PATH`: Path to configuration file (default: "cam_config.yaml")
- `--model PATH`: Path to pretrained model
- `--class_idx INDEX`: Target class index for CAM
- `--train`: Train a new model
- `--image PATH`: Path to a specific image file to visualize
- `--random N`: Number of random Imagenette samples to visualize
- `--index INDEX`: Index of a specific Imagenette sample to visualize
- `--dataset {train,val}`: Dataset to use (train or val) - (default: "val")
- `--output_dir DIR`: Directory to save visualizations (default: "cam_results/")

### Configuration

The configuration file (`cam_config.yaml`) allows customizing various aspects:

```yaml
model:
  name: resnet_cam
  input_shape: [3, 160, 160] # [channels, height, width] for Imagenette (160x160)
  num_classes: 10
  residual: true

training:
  learning_rate: 0.001
  batch_size: 64
  num_epochs: 50
  weight_decay: 0.0001
  checkpoint_dir: checkpoints/
  log_every: 10
  save_every: 2
  early_stopping_patience: 5
  optimizer: adam # [adam, sgd]
  scheduler: reduce_lr # [reduce_lr, cosine]

dataset:
  name: imagenette
  data_dir: data/
  img_size: 160 # Imagenette image resolution
  val_size: 1000 # Validation set size
```

### Example Commands

1. **Train a new model:**

   ```
   python -m DLA_Labs.L1_CNNs.CAM.main --train --config cam_config.yaml
   ```

2. **Visualize CAM for a specific image:**

   ```
   python -m DLA_Labs.L1_CNNs.CAM.main --model checkpoints/resnet_cam.pth --image path/to/image.jpg --output_dir cam_results/
   ```

3. **Visualize CAM for random samples from the validation set:**

   ```
   python -m DLA_Labs.L1_CNNs.CAM.main --model checkpoints/resnet_cam.pth --random 5 --dataset val --output_dir cam_results/
   ```

4. **Visualize CAM for a specific image at a given index in the dataset:**

   ```
   python -m DLA_Labs.L1_CNNs.CAM.main --model checkpoints/resnet_cam.pth --index 42 --dataset val --output_dir cam_results/
   ```

5. **Target a specific class for CAM visualization:**
   ```
   python -m DLA_Labs.L1_CNNs.CAM.main --model checkpoints/resnet_cam.pth --image path/to/image.jpg --class_idx 3 --output_dir cam_results/
   ```

## Installation


1. Clone the repository:

   ```
   git clone https://github.com/lorenzo-27/DLA-Labs.git
   cd DLA_Labs
   ```

2. Install the dependencies with pip or conda. The `requirements.txt` file contains all the necessary packages:

    ```
    pip install -r requirements.txt
    ``` 
    ```
    conda install --file requirements.txt
    ```

3. Run the setup to install the package in development mode:

   ```
   pip install -e .
   ```

4. When first run, the script will automatically download the Imagenette dataset to the specified data directory (default: `./data/`).

## Results Interpretation

The CAM visualization produces three images:

1. **Original Image**: The input image
2. **CAM Heatmap**: A heat map where warmer colors (red, yellow) indicate regions that strongly activated the target class
3. **Overlay**: The CAM heatmap superimposed on the original image

The visualization also displays:

- The predicted class (or the specified target class)
- For dataset visualizations, both the true class and predicted class

<p align="center">
<img  src="https://github.com/lorenzo-27/DLA-Labs/blob/master/assets/cam_results.png" width="90%" height="90%"/>
</p>

### Understanding the Heatmap

- **Red/Yellow regions**: Areas most important for the classification decision
- **Blue/Green regions**: Areas with less influence on the classification
- **Dark blue regions**: Areas with minimal influence on the classification

## Project Structure

```
DLA_Labs/L1_CNNs/CAM/
├── cam_config.yaml      # Default configuration file
├── data.py              # Dataset loading and preprocessing
├── main.py              # Main script for running the application
├── model.py             # ResNetCAM model implementation
├── utils.py             # Helper functions and configurations
└── visualization.py     # Functions for generating CAM visualizations
```

## Common Issues

- **CUDA Out of Memory**: Reduce batch size in the configuration file
- **Dataset download issues**: If automatic download fails, manually download from [fast.ai's Imagenette](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz) and extract to the data directory
- **Loading pretrained model**: Ensure model path is correct and contains a compatible model state dictionary

## References

1. Bagdanov, A. D., DLA course material (2024)

2. Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., & Torralba, A. (2016). Learning Deep Features for Discriminative Localization. _IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_. [arXiv:1512.04150](https://arxiv.org/abs/1512.04150)

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. _IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

3. ImageNette dataset by Jeremy Howard: [https://github.com/fastai/imagenette](https://github.com/fastai/imagenette)
