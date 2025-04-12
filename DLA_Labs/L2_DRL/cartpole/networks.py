import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    """
    A simple policy network with variable number of hidden layers.

    Attributes:
        hidden (nn.Sequential): Sequential container for hidden layers.
        out (nn.Linear): Output layer mapping to action space.
    """

    def __init__(self, input_dim, output_dim, n_hidden=1, width=128):
        """
        Initialize the PolicyNet.

        Args:
            input_dim (int): Dimension of the input (state space).
            output_dim (int): Dimension of the output (action space).
            n_hidden (int): Number of hidden layers.
            width (int): Number of neurons in each hidden layer.
        """
        super().__init__()

        # Create the hidden layers
        hidden_layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for _ in range(n_hidden - 1):
            hidden_layers.extend([nn.Linear(width, width), nn.ReLU()])

        self.hidden = nn.Sequential(*hidden_layers)
        self.out = nn.Linear(width, output_dim)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Probability distribution over actions.
        """
        x = self.hidden(x)
        x = self.out(x)
        return x


class ValueNet(nn.Module):
    """
    Network for estimating the state value function V(s).

    The Value Network shares the same architecture as the Policy Network but has a single output
    (the value of the state) instead of a probability distribution over actions.

    Attributes:
        hidden (nn.Sequential): Sequential container for hidden layers.
        out (nn.Linear): Output layer mapping to a single value.
    """

    def __init__(self, input_dim, n_hidden=1, width=128):
        """
        Initialize the ValueNet.

        Args:
            input_dim (int): Dimension of the input (state space).
            n_hidden (int): Number of hidden layers.
            width (int): Number of neurons in each hidden layer.
        """
        super().__init__()

        # Create the hidden layers with the same architecture as PolicyNet
        hidden_layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for _ in range(n_hidden - 1):
            hidden_layers.extend([nn.Linear(width, width), nn.ReLU()])

        self.hidden = nn.Sequential(*hidden_layers)
        # Output a single value (the state value)
        self.out = nn.Linear(width, 1)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Scalar value estimate for the state.
        """
        x = self.hidden(x)
        x = self.out(x)
        return x