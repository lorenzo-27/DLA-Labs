import torch
from torch import nn
from transformers import AutoModelForSequenceClassification
from typing import List, Optional, Union


class FreezeBackboneModel(nn.Module):
    """
    Wrapper class to freeze the backbone of a Hugging Face model and only fine-tune
    the classification head.
    """

    def __init__(self, model_name: str, num_labels: int = 2):
        """
        Initialize the model with frozen backbone layers.

        Args:
            model_name (str): Name of the pretrained model from HuggingFace.
            num_labels (int): Number of output classes.
        """
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

        # Freeze all parameters in the base model
        for param in self.model.base_model.parameters():
            param.requires_grad = False

    def forward(self, **kwargs):
        """Forward pass, directly using the wrapped model's forward."""
        return self.model(**kwargs)

    def print_trainable_parameters(self):
        """Print the number and percentage of trainable parameters."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable Parameters: {trainable_params} ({trainable_params / total_params:.2%})")


class AdapterModule(nn.Module):
    """A simple bottleneck adapter module that can be inserted between transformer layers."""

    def __init__(self, input_dim: int, adapter_dim: int, dropout_rate: float = 0.1):
        """
        Initialize adapter with down and up projections.

        Args:
            input_dim (int): Dimension of the input features
            adapter_dim (int): Dimension of the adapter bottleneck
            dropout_rate (float): Dropout rate to apply
        """
        super().__init__()
        self.down_proj = nn.Linear(input_dim, adapter_dim)
        self.activation = nn.GELU() # same as BERT and DistilBERT
        self.dropout = nn.Dropout(dropout_rate)
        self.up_proj = nn.Linear(adapter_dim, input_dim)

        # Initialize to near-identity function
        nn.init.normal_(self.down_proj.weight, std=1e-3)
        nn.init.normal_(self.up_proj.weight, std=1e-3)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the adapter.

        Args:
            hidden_states (torch.Tensor): Input hidden states from transformer

        Returns:
            torch.Tensor: Output hidden states with residual connection
        """
        residual = hidden_states

        # Down projection
        x = self.down_proj(hidden_states)
        x = self.activation(x)
        x = self.dropout(x)

        # Up projection
        x = self.up_proj(x)

        # Add residual connection
        output = x + residual

        return output


class AdapterTransformer(nn.Module):
    """
    Model with adapters inserted into a pre-trained transformer.
    """

    def __init__(
            self,
            model_name: str,
            num_labels: int = 2,
            adapter_dim: int = 64,
            dropout_rate: float = 0.1,
            target_modules: Optional[List[str]] = None
    ):
        """
        Initialize the adapter-based transformer model.

        Args:
            model_name (str): Name of the pretrained model from HuggingFace
            num_labels (int): Number of output classes
            adapter_dim (int): Dimension of the adapter bottleneck
            dropout_rate (float): Dropout rate for adapters
            target_modules (List[str], optional): List of module names where adapters should be inserted
        """
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

        # Freeze all parameters in the pretrained model
        for param in self.model.parameters():
            param.requires_grad = False

        # For DistilBERT, we insert adapters into each transformer block
        if target_modules is None:
            # By default, add adapters after attention and FFN in each transformer block
            if "distilbert" in model_name.lower():
                target_modules = ["attention", "ffn"]

        # Insert adapters
        self._insert_adapters(target_modules, adapter_dim, dropout_rate)

        # Unfreeze classification head
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def _insert_adapters(self, target_modules: List[str], adapter_dim: int, dropout_rate: float):
        """
        Insert adapter modules after specific target modules in the transformer.

        Args:
            target_modules (List[str]): List of module names where adapters should be inserted
            adapter_dim (int): Dimension of the adapter bottleneck
            dropout_rate (float): Dropout rate for adapters
        """
        # For DistilBERT
        if hasattr(self.model, "distilbert"):
            # Add adapters to each transformer block
            for layer in self.model.distilbert.transformer.layer:
                # Get hidden dimension from the model
                hidden_size = layer.attention.out_lin.out_features

                # Add adapter after attention
                if "attention" in target_modules:
                    attention_adapter = AdapterModule(hidden_size, adapter_dim, dropout_rate)
                    # Store original forward
                    original_attn_forward = layer.attention.forward

                    # Define new forward with adapter
                    def make_attention_forward(original_forward, adapter):
                        def new_forward(*args, **kwargs):
                            output = original_forward(*args, **kwargs)
                            if isinstance(output, tuple):
                                output = (adapter(output[0]),) + output[1:]
                            else:
                                output = adapter(output)
                            return output

                        return new_forward

                    # Replace forward
                    layer.attention.forward = make_attention_forward(original_attn_forward, attention_adapter)

                # Add adapter after FFN (Feed Forward Network)
                if "ffn" in target_modules:
                    ffn_adapter = AdapterModule(hidden_size, adapter_dim, dropout_rate)
                    original_ffn_forward = layer.ffn.forward

                    def make_ffn_forward(original_forward, adapter):
                        def new_forward(*args, **kwargs):
                            output = original_forward(*args, **kwargs)
                            output = adapter(output)
                            return output

                        return new_forward

                    layer.ffn.forward = make_ffn_forward(original_ffn_forward, ffn_adapter)

    def forward(self, **kwargs):
        """Forward pass, directly using the wrapped model's forward."""
        return self.model(**kwargs)

    def print_trainable_parameters(self):
        """Print the number and percentage of trainable parameters."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable Parameters: {trainable_params} ({trainable_params / total_params:.2%})")


def get_efficient_model(
        model_name: str,
        method: str = "lora",
        num_labels: int = 2,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        adapter_dim: int = 64,
        target_modules: Optional[List[str]] = None
):
    """
    Get an efficiently fine-tunable model based on the specified method.

    Args:
        model_name (str): Name of the pretrained model from HuggingFace
        method (str): Method for efficient fine-tuning ("lora", "freeze", "adapters")
        num_labels (int): Number of output classes
        lora_r (int): LoRA rank dimension
        lora_alpha (int): LoRA alpha parameter for scaling
        lora_dropout (float): Dropout rate for LoRA modules
        adapter_dim (int): Dimension of adapter bottlenecks
        target_modules (List[str], optional): List of module names where adapters/LoRA should be inserted

    Returns:
        nn.Module: Efficiently fine-tunable model
    """
    if method == "lora":
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            # Load base model for sequence classification
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )

            # Set default target modules for DistilBERT if not specified
            if target_modules is None and "distilbert" in model_name.lower():
                target_modules = ["q_lin", "k_lin", "v_lin", "out_lin"]

            # Configure LoRA
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
            )

            # Get PEFT model
            model = get_peft_model(model, lora_config)
            return model
        except ImportError:
            print("PEFT library not found. Please install with: pip install peft")
            # Fall back to freeze method
            return FreezeBackboneModel(model_name, num_labels)

    elif method == "freeze":
        return FreezeBackboneModel(model_name, num_labels)

    elif method == "adapters":
        return AdapterTransformer(
            model_name=model_name,
            num_labels=num_labels,
            adapter_dim=adapter_dim,
            target_modules=target_modules
        )

    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'lora', 'freeze', or 'adapters'.")