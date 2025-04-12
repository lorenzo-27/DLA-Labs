import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNNFeatureExtractor(nn.Module):
    """
    CNN feature extractor for processing image observations.
    Takes stacked grayscale frames as input.
    """

    def __init__(self, input_channels=4):
        super().__init__()

        # CNN architecture
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate output dimensions
        # Assuming input is 84x84xN after preprocessing
        self.feature_dim = 64 * 7 * 7

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        return x


class ActorCNN(nn.Module):
    """
    Actor network for PPO with CNN feature extraction.
    Maps image observations to action probabilities.
    """

    def __init__(self, input_channels=4, action_dim=5, hidden_dim=256):
        super().__init__()

        # Feature extraction
        self.features = CNNFeatureExtractor(input_channels)

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(self.features.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        features = self.features(x)
        action_logits = self.action_head(features)
        return action_logits


class CriticCNN(nn.Module):
    """
    Critic network for PPO with CNN feature extraction.
    Maps image observations to state value predictions.
    """

    def __init__(self, input_channels=4, hidden_dim=256):
        super().__init__()

        # Feature extraction
        self.features = CNNFeatureExtractor(input_channels)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.features.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        features = self.features(x)
        state_values = self.value_head(features)
        return state_values
