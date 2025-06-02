# Transformer Fine-tuning for Sentiment Analysis

This repository contains implementations of different approaches for fine-tuning transformer models on sentiment analysis tasks, focusing on efficient parameter training techniques including LoRA (Low-Rank Adaptation), freezing strategies, and adapter modules.

## Table of Contents

- [Theory](#theory)
  - [Transformer Models](#transformer-models)
  - [Fine-tuning Approaches](#fine-tuning-approaches)
  - [Efficient Fine-tuning Methods](#efficient-fine-tuning-methods)
- [Implementation](#implementation)
  - [Exercise 1: SVM Baseline](#exercise-1-svm-baseline)
  - [Exercise 2: Full Fine-tuning](#exercise-2-full-fine-tuning)
  - [Exercise 3: Efficient Fine-tuning](#exercise-3-efficient-fine-tuning)
- [Usage](#usage)
  - [Command Line Arguments](#command-line-arguments)
  - [Example Commands](#example-commands)
- [Installation](#installation)
- [Results](#results)
- [Known Issues](#known-issues)
- [Project Structure](#project-structure)
- [References](#references)

## Theory

### Transformer Models

Transformer models, particularly BERT (Bidirectional Encoder Representations from Transformers) and its variants like DistilBERT, have revolutionized natural language processing. These models are pre-trained on large text datasets using self-supervised learning objectives and can be fine-tuned for downstream tasks like sentiment analysis.

Key components of transformer architectures include:

1. **Self-Attention Mechanism**: Allows the model to weigh the importance of different words in a sequence when processing each word
2. **Multi-Head Attention**: Multiple attention heads that can focus on different types of relationships
3. **Feed-Forward Networks**: Dense layers that process the attention outputs
4. **Layer Normalization**: Stabilizes training and improves convergence
5. **Positional Encoding**: Provides information about word positions in sequences

### Fine-tuning Approaches

Fine-tuning pre-trained models involves adapting them to specific tasks by continuing training on task-specific data. Traditional approaches include:

1. **Full Fine-tuning**: Update all model parameters during training
2. **Feature Extraction**: Freeze pre-trained weights and only train new classifier layers
3. **Gradual Unfreezing**: Progressively unfreeze layers during training

### Efficient Fine-tuning Methods

Recent advances have introduced parameter-efficient fine-tuning methods that achieve comparable performance while training significantly fewer parameters:

#### Low-Rank Adapters (LoRA)

LoRA introduces trainable rank decomposition matrices into transformer layers while keeping the original weights frozen. For a pre-trained weight matrix W, LoRA represents the weight update as:

$$W' = W + BA$$

Where:
- $W$ is the original frozen weight matrix
- $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ are trainable matrices
- $r$ is the rank (much smaller than the original dimensions)

This approach dramatically reduces the number of trainable parameters while maintaining model performance.

#### Adapter Modules

Adapters are small neural networks inserted between transformer layers. They consist of:

1. **Down-projection**: Reduces dimensionality to a bottleneck
2. **Non-linearity**: Typically GELU activation
3. **Up-projection**: Returns to original dimensionality
4. **Residual connection**: Adds the adapter output to the input

The adapter transformation can be expressed as:

$$h_{adapter} = h + f(h)$$

Where $f(h) = W_{up}(\text{GELU}(W_{down}(h)))$ and $h$ is the input hidden state.

#### Freezing Strategies

This approach involves freezing the entire pre-trained backbone and only fine-tuning the task-specific classification head. While simple, it can be effective for tasks where the pre-trained representations are already well-suited.

## Implementation

### Exercise 1: SVM Baseline

The first exercise establishes a baseline using Support Vector Machines (SVM) with BERT-extracted features:

1. **Feature Extraction**: Uses a pre-trained BERT model to extract [CLS] token representations
2. **SVM Training**: Trains a linear SVM classifier on the extracted features
3. **Evaluation**: Measures performance on validation and test sets

This approach provides a strong baseline while being computationally efficient, as it doesn't require fine-tuning the transformer model.

### Exercise 2: Full Fine-tuning

The second exercise implements traditional full fine-tuning:

1. **Model Loading**: Loads a pre-trained model with a classification head
2. **End-to-End Training**: Updates all model parameters during training
3. **Optimization**: Uses the Hugging Face Trainer with appropriate hyperparameters

This approach typically achieves the best performance but requires training all model parameters.

### Exercise 3: Efficient Fine-tuning

The third exercise focuses on parameter-efficient methods and includes three different approaches:

#### LoRA Implementation

The LoRA implementation uses the PEFT (Parameter-Efficient Fine-Tuning) library:

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,  # Rank dimension
    lora_alpha=16,  # Scaling parameter
    lora_dropout=0.1,
    target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],  # Target attention layers
    bias="none",
)

model = get_peft_model(base_model, lora_config)
```

#### Freeze Implementation

The freeze approach is implemented in the `FreezeBackboneModel` class:

```python
class FreezeBackboneModel(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 2):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        
        # Freeze all parameters in the base model
        for param in self.model.base_model.parameters():
            param.requires_grad = False
```

#### Adapter Implementation

The adapter implementation includes a custom `AdapterModule` class:

```python
class AdapterModule(nn.Module):
    def __init__(self, input_dim: int, adapter_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, adapter_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.up_proj = nn.Linear(adapter_dim, input_dim)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        x = self.up_proj(self.dropout(self.activation(self.down_proj(hidden_states))))
        return x + residual
```

The `AdapterTransformer` class inserts these modules into the transformer layers and handles the integration with the pre-trained model.

## Usage

### Command Line Arguments

The main script supports various configuration options:

```bash
python -m DLA_Labs.L3_Transformers.main [OPTIONS]
```

**General Parameters:**
- `--project`: WandB project name (default: 'BERT-Sentiment-Analysis')
- `--dataset`: HuggingFace dataset to use (default: 'cornell-movie-review-data/rotten_tomatoes')
- `--model`: Pre-trained model to use (default: 'distilbert/distilbert-base-uncased')
- `--seed`: Random seed for reproducibility (default: 42)
- `--device`: Device to run on (default: 'cuda' if available, else 'cpu')

**Training Parameters:**
- `--epochs`: Number of training epochs (default: 3)
- `--batch_size`: Batch size for training (default: 16)
- `--lr`: Learning rate (default: 5e-5)
- `--exercise`: Exercise to run (1=baseline, 2=finetuning, 3=efficient finetuning)
- `--max_length`: Maximum sequence length for tokenization (default: 512)

**Efficient Fine-tuning Parameters:**
- `--method`: Method for efficient fine-tuning ('lora', 'freeze', 'adapters')
- `--lora_r`: LoRA rank dimension (default: 8)
- `--lora_alpha`: LoRA alpha parameter (default: 16)
- `--lora_dropout`: LoRA dropout rate (default: 0.1)

### Example Commands

1. **Run SVM baseline (Exercise 1):**
   ```bash
   python -m DLA_Labs.L3_Transformers.main --exercise 1
   ```

2. **Run full fine-tuning (Exercise 2):**
   ```bash
   python -m DLA_Labs.L3_Transformers.main --exercise 2
   ```

3. **Run LoRA fine-tuning (Exercise 3):**
   ```bash
   python -m DLA_Labs.L3_Transformers.main --exercise 3 --method lora
   ```

4. **Run freeze fine-tuning (Exercise 3):**
   ```bash
   python -m DLA_Labs.L3_Transformers.main --exercise 3 --method freeze
   ```

5. **Run adapter fine-tuning (Exercise 3):**
   ```bash
   python -m DLA_Labs.L3_Transformers.main --exercise 3 --method adapters
   ```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/DLA-Labs.git
   cd DLA-Labs
   ```

2. Install the dependencies using the requirements file:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

4. Set up Weights & Biases (optional):
   ```bash
   wandb login
   ```

## Results

The following results were obtained on the Rotten Tomatoes sentiment analysis dataset using DistilBERT as the base model:

### Performance Comparison

| Method | Accuracy | F1-Score | Loss |
|--------|----------|----------|------|
| LoRA | **0.831** | 0.831 | 0.403 |
| Full Fine-tuning | 0.820 | 0.819 | 0.425 |
| Freeze | 0.778 | 0.777 | 0.474 |

### Parameter Efficiency

| Method | Trainable Parameters | Percentage of Total |
|--------|---------------------|---------------------|
| LoRA | ~0.3M | ~0.5% |
| Full Fine-tuning | ~67M | 100% |
| Freeze | ~1.5M | ~2.2% |

### Key Findings

1. **LoRA achieves the best performance** while training only 0.5% of the total parameters
2. **Full fine-tuning** provides strong performance but requires updating all parameters
3. **Freezing** is the quite parameter-efficient but shows lower performance
4. **Parameter efficiency vs. performance trade-off**: LoRA provides an excellent balance

## Known Issues

### Adapter Implementation Issue

The adapter implementation currently has a compatibility issue with the Hugging Face training pipeline. During the evaluation phase, the following error occurs:

```
KeyError: 'eval_loss'
KeyError: "The `metric_for_best_model` training argument is set to 'eval_loss', which is not found in the evaluation metrics. The available evaluation metrics are: ['eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second', 'epoch']."
```

This appears to be a pipeline compatibility issue between the custom adapter implementation and the Hugging Face Trainer's metric handling. The training completes successfully, but the evaluation phase fails when trying to determine the best model based on metrics.

## Project Structure

```
DLA_Labs/L3_Transformers/
├── main.py                     # Main training script
├── efficient_finetuning.py     # Efficient fine-tuning implementations
└── README.md                   # This file
```

## References

1. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *arXiv preprint arXiv:1810.04805*.

2. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv preprint arXiv:2106.09685*.

3. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *arXiv preprint arXiv:1910.01108*.
