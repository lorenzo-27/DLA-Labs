import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation=True):
        """Blocco convoluzionale di base."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if activation else nn.Identity()
        )

    def forward(self, x):
        return self.layers(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        """Blocco convoluzionale residuale con connessione skip."""
        super().__init__()
        self.same_dim = (in_channels == out_channels)
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size, activation=True),
            ConvBlock(out_channels, out_channels, kernel_size, activation=False)
        )
        if not self.same_dim:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        if not self.same_dim:
            residual = self.projection(residual)
        return self.activation(out + residual)


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes, filters, kernel_size, input_shape):
        """
        Modello CNN senza connessioni residuali.

        Args:
            in_channels: Numero di canali dell'input
            num_classes: Numero di classi per l'output
            filters: Lista di numeri di filtri per ciascun layer
            kernel_size: Dimensione del kernel per le convoluzioni
            input_shape: Dimensione dell'input (es. (3, 32, 32) per CIFAR-10)
        """
        super().__init__()
        self.input_shape = input_shape

        # Layers convoluzionali
        self.conv_layers = nn.ModuleList()
        for i in range(len(filters)):
            in_c = in_channels if i == 0 else filters[i - 1]
            out_c = filters[i]
            self.conv_layers.append(ConvBlock(in_c, out_c, kernel_size))
            # Aggiunge un layer di MaxPooling ogni 2 layers
            if i % 2 == 1:
                self.conv_layers.append(nn.MaxPool2d(2))

        # Calcolo della dimensione dell'output dopo le convoluzioni
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            for layer in self.conv_layers:
                x = layer(x)
            output_size = x.view(1, -1).size(1)

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return self.fc_layers(x)


class ResidualCNN(nn.Module):
    def __init__(self, in_channels, num_classes, filters, kernel_size, input_shape):
        """
        Modello CNN con connessioni residuali.

        Args:
            in_channels: Numero di canali dell'input
            num_classes: Numero di classi per l'output
            filters: Lista di numeri di filtri per ciascun layer
            kernel_size: Dimensione del kernel per le convoluzioni
            input_shape: Dimensione dell'input (es. (3, 32, 32) per CIFAR-10)
        """
        super().__init__()
        self.input_shape = input_shape

        # Primo layer convoluzionale senza connessione residuale
        self.first_conv = ConvBlock(in_channels, filters[0], kernel_size)

        # Layers residuali
        self.res_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        for i in range(len(filters) - 1):
            in_c = filters[i]
            out_c = filters[i + 1]
            self.res_layers.append(ResidualConvBlock(in_c, out_c, kernel_size))
            # Aggiunge un layer di MaxPooling ogni 2 blocchi residuali
            if i % 2 == 1:
                self.pool_layers.append(nn.MaxPool2d(2))

        # Calcolo della dimensione dell'output dopo le convoluzioni
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            x = self.first_conv(x)

            for i, layer in enumerate(self.res_layers):
                x = layer(x)
                if i % 2 == 1 and i // 2 < len(self.pool_layers):
                    x = self.pool_layers[i // 2](x)

            output_size = x.view(1, -1).size(1)

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.first_conv(x)

        for i, layer in enumerate(self.res_layers):
            x = layer(x)
            if i % 2 == 1 and i // 2 < len(self.pool_layers):
                x = self.pool_layers[i // 2](x)

        return self.fc_layers(x)


def load_model(config):
    """Carica il modello CNN basato sui parametri forniti."""
    model_params = config["model"]
    input_shape = tuple(model_params["input_shape"])

    if model_params["residual"]:
        model = ResidualCNN(
            in_channels=input_shape[0],
            num_classes=model_params["num_classes"],
            filters=model_params["filters"],
            kernel_size=model_params["kernel_size"],
            input_shape=input_shape
        )
    else:
        model = CNN(
            in_channels=input_shape[0],
            num_classes=model_params["num_classes"],
            filters=model_params["filters"],
            kernel_size=model_params["kernel_size"],
            input_shape=input_shape
        )

    return model


def main():
    """Funzione di test per verificare l'architettura del modello."""
    from rich.console import Console
    from torchinfo import summary
    import yaml

    console = Console()

    with open("m_cnn.yaml", "r") as file:
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