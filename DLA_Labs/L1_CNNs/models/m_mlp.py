import torch
from torch import nn


class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features, activation=True):
        """MLP block with optional activation function."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU() if activation else nn.Identity()
        )

    def forward(self, x):
        return self.layers(x)


class ResidualMLPBlock(nn.Module):
    def __init__(self, in_features, out_features):
        """MLP residual block with skip connection."""
        super().__init__()
        self.same_dim = (in_features == out_features)
        self.block = nn.Sequential(
            MLPBlock(in_features, out_features, activation=True),
            MLPBlock(out_features, out_features, activation=False)
        )
        # if dimensions are different, apply a linear projection to map the residual to the target dimension
        if not self.same_dim:
            self.projection = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        if not self.same_dim:
            residual = self.projection(residual)
        # add the residual to the output and apply the activation function
        return self.activation(out + residual)


class MLP(nn.Module):
    def __init__(self, layer_sizes, input_shape):
        """
        MLP model without residual connections.

        Args:
            layer_sizes: list of integers representing the size of each layer
            input_shape: input shape (e.g. (1, 28, 28) for MNIST - (3, 32, 32) for CIFAR-10)
        """
        super().__init__()
        self.input_shape = input_shape
        input_size = input_shape[0] * input_shape[1] * input_shape[2]
        layers = []

        for i in range(len(layer_sizes) - 1):
            in_features = input_size if i == 0 else layer_sizes[i - 1]
            out_features = layer_sizes[i]
            layers.append(MLPBlock(in_features, out_features))

        # Output layer
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        return self.model(x)


class ResidualMLP(nn.Module):
    def __init__(self, layer_sizes, input_shape):
        """
        MLP model with residual connections.

        Args:
            layer_sizes: list of integers representing the size of each layer
            input_shape: input shape (e.g. (1, 28, 28) for MNIST - (3, 32, 32) for CIFAR-10)
        """
        super().__init__()
        self.input_shape = input_shape
        input_size = input_shape[0] * input_shape[1] * input_shape[2]

        # First layer without residual connection
        self.first_layer = MLPBlock(input_size, layer_sizes[0])

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            in_features = layer_sizes[i]
            out_features = layer_sizes[i + 1]
            self.res_blocks.append(ResidualMLPBlock(in_features, out_features))

        # Output layer
        self.output_layer = nn.Linear(layer_sizes[-1], layer_sizes[-1])

    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)

        # Pass through first layer
        x = self.first_layer(x)

        # Pass through residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Output layer
        x = self.output_layer(x)
        return x


def load_model(config):
    """Load the MLP model based on the provided parameters."""
    model_params = config["model"]
    input_shape = model_params["input_shape"]

    if model_params["residual"]:
        model = ResidualMLP(
            layer_sizes=model_params["layer_sizes"],
            input_shape=input_shape
        )
    else:
        model = MLP(
            layer_sizes=model_params["layer_sizes"],
            input_shape=input_shape
        )

    return model


def main():
    """Test the MLP model."""
    from rich.console import Console
    from torchinfo import summary
    import yaml

    console = Console()

    with open("m_mlp.yaml", "r") as file:
        config = yaml.safe_load(file)

    batch_size = 32
    input_shape = tuple(config["model"]["input_shape"])

    # Prepare input data based on the input shape
    input_data = torch.randn(batch_size, *input_shape)

    model = load_model(config)

    _ = model(input_data)
    model_stats = summary(
        model,
        input_data=input_data,
        col_names=["input_size", "output_size", "num_params"],
        row_settings=("var_names",),
        col_width=18,
        depth=8,
        verbose=0,
    )
    console.print(model_stats)


if __name__ == "__main__":
    main()